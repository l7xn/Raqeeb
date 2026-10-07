"""Raqeeb backend: runs the YOLO + TrackTrack detector and serves the dashboard.

Live camera (default webcam 0):
    python server.py
Another camera, video file or RTSP stream:
    python server.py --source rtsp://user:pass@host/stream
Replay a recorded detections.csv (no camera or ultralytics needed):
    python server.py --replay samples/detections.csv

Then open http://localhost:8000 and go to "AI Detector".
"""

import argparse
import collections
import csv
import json
import re
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
PAGES = {"/": "raqeeb.html", "/raqeeb.html": "raqeeb.html",
         "/sentinelops-dashboard.html": "sentinelops-dashboard.html"}
LOST_AFTER = 2.0   # seconds without a detection before a track is reported as lost


class State:
    """Detector state shared between the tracking thread and HTTP handlers."""

    def __init__(self, mode, source):
        self.lock = threading.Lock()
        self.new_jpeg = threading.Condition(self.lock)
        self.mode, self.source = mode, source
        self.status, self.error = "starting", None
        self.video = False
        self.jpeg = None
        self.reset()

    def reset(self):
        self.entities = {}   # track_id -> summary, same fields as raqeeb.py's entities.json
        self.events = collections.deque(maxlen=200)
        self.seq = getattr(self, "seq", 0)
        self.dets, self.frame, self.size = [], 0, (640, 480)
        self.fps, self.last_t = 0.0, None

    def _event(self, kind, tid, e):
        self.seq += 1
        self.events.append({"seq": self.seq, "kind": kind, "id": tid, "class": e["class"],
                            "time": e["last_seen"], "conf": round(e["conf_sum"] / e["frames"], 3)})

    def update(self, now, frame_idx, size, dets, jpeg=None):
        """dets: list of (track_id, label, conf, [x1, y1, x2, y2])."""
        t = time.monotonic()
        with self.lock:
            if self.last_t is not None and t > self.last_t:
                inst = 1.0 / (t - self.last_t)
                self.fps = inst if not self.fps else 0.9 * self.fps + 0.1 * inst
            self.last_t, self.frame, self.size, self.status = t, frame_idx, size, "running"
            self.dets = [{"id": tid, "class": c, "conf": round(cf, 3), "box": [round(v, 1) for v in b]}
                         for tid, c, cf, b in dets]
            for tid, label, cf, _ in dets:
                e = self.entities.get(tid)
                new = e is None or not e["active"]
                if e is None:
                    e = self.entities[tid] = {"class": label, "first_seen": now, "last_seen": now,
                                              "frames": 0, "conf_sum": 0.0, "active": True, "t": t}
                e.update(last_seen=now, active=True, t=t)
                e["frames"] += 1
                e["conf_sum"] += cf
                if new:
                    self._event("new", tid, e)
            self._expire(t - LOST_AFTER)
            if jpeg is not None:
                self.jpeg, self.video = jpeg, True
                self.new_jpeg.notify_all()

    def _expire(self, before):
        for tid, e in self.entities.items():
            if e["active"] and e["t"] < before:
                e["active"] = False
                self._event("lost", tid, e)

    def finish(self, status, error=None):
        """Source ended or failed: every remaining track has left view."""
        with self.lock:
            self.status, self.error = status, error
            self._expire(float("inf"))

    def entities_json(self):
        """Entities in the same format raqeeb.py writes to entities.json."""
        with self.lock:
            return {tid: {k: e[k] for k in ("class", "first_seen", "last_seen", "frames")}
                    for tid, e in self.entities.items()}

    def snapshot(self, since):
        with self.lock:
            stale = self.last_t is None or time.monotonic() - self.last_t > LOST_AFTER
            dets = [] if stale else self.dets
            counts = collections.Counter(d["class"] for d in dets)
            return {
                "mode": self.mode, "source": self.source, "status": self.status, "error": self.error,
                "video": self.video, "frame": self.frame, "fps": round(self.fps, 1),
                "w": self.size[0], "h": self.size[1], "dets": dets, "counts": counts,
                "entities": [{"id": tid, "class": e["class"], "first_seen": e["first_seen"],
                              "last_seen": e["last_seen"], "frames": e["frames"],
                              "conf": round(e["conf_sum"] / e["frames"], 3), "active": e["active"]}
                             for tid, e in sorted(self.entities.items())],
                "seq": self.seq, "events": [ev for ev in self.events if ev["seq"] > since],
            }


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def run_live(state, args):
    """Same tracking loop as raqeeb.py, headless, feeding the dashboard."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model)
    names = model.names
    source = int(args.source) if args.source.isdigit() else args.source
    last_save = 0.0

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frame", "track_id", "class", "conf", "x1", "y1", "x2", "y2"])
        frame_idx = 0
        try:
            for r in model.track(source=source, persist=True, tracker=args.tracker,
                                 stream=True, verbose=False):
                frame_idx += 1
                h, w = r.orig_shape
                now = now_iso()
                dets = []
                boxes = r.boxes
                if boxes is not None and boxes.id is not None:
                    for tid, c, cf, b in zip(boxes.id.int().tolist(), boxes.cls.int().tolist(),
                                             boxes.conf.tolist(), boxes.xyxy.tolist()):
                        label = names[c]
                        writer.writerow([now, frame_idx, tid, label, round(cf, 3),
                                         *[round(v, 1) for v in b]])
                        dets.append((tid, label, cf, b))
                ok, buf = cv2.imencode(".jpg", r.plot(), [cv2.IMWRITE_JPEG_QUALITY, 75])
                state.update(now, frame_idx, (w, h), dets, buf.tobytes() if ok else None)
                if time.monotonic() - last_save > 2:
                    last_save = time.monotonic()
                    f.flush()
                    save_entities(state, args.entities)
        finally:
            save_entities(state, args.entities)


def save_entities(state, path):
    with open(path, "w") as jf:
        json.dump(state.entities_json(), jf, indent=2)


def run_replay(state, args):
    """Play back a detections.csv recorded by raqeeb.py as if it were live."""
    frames = collections.OrderedDict()
    with open(args.replay, newline="") as f:
        for row in csv.DictReader(f):
            box = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
            frames.setdefault(int(row["frame"]), []).append(
                (int(row["track_id"]), row["class"], float(row["conf"]), box))
    if not frames:
        raise ValueError(f"{args.replay} has no detections")
    size = tuple(int(v) for v in args.size.lower().split("x"))
    while True:
        with state.lock:
            state.reset()
        for frame_idx, dets in frames.items():
            state.update(now_iso(), frame_idx, size, dets)
            time.sleep(1 / args.fps)
        time.sleep(LOST_AFTER + 1)   # let tracks expire before looping
        state.update(now_iso(), 0, size, [])


def start_detector(state, args):
    def target():
        try:
            (run_replay if args.replay else run_live)(state, args)
            state.finish("stopped")
        except Exception as e:  # surface failures (no camera, missing ultralytics) in the dashboard
            state.finish("error", f"{type(e).__name__}: {e}")
            print("Detector stopped:", state.error)

    threading.Thread(target=target, daemon=True).start()


def make_handler(state, args):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            p = url.path
            if p in PAGES:
                return self.send(200, (ROOT / PAGES[p]).read_bytes(), "text/html; charset=utf-8")
            if p == "/api/state":
                since = int(parse_qs(url.query).get("since", ["0"])[0] or 0)
                return self.send(200, json.dumps(state.snapshot(since)).encode(), "application/json")
            if p == "/api/entities.json":
                body = json.dumps(state.entities_json(), indent=2).encode()
                return self.send(200, body, "application/json",
                                 {"Content-Disposition": "attachment; filename=entities.json"})
            if p == "/api/detections.csv":
                path = Path(args.replay or args.csv)
                if not path.exists():
                    return self.send(404, b"no detections yet", "text/plain")
                return self.send(200, path.read_bytes(), "text/csv",
                                 {"Content-Disposition": "attachment; filename=detections.csv"})
            if p == "/api/stream.mjpg":
                return self.stream()
            self.send(404, b"not found", "text/plain")

        def stream(self):
            if not state.video:
                return self.send(404, b"no video in this mode", "text/plain")
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with state.new_jpeg:
                        state.new_jpeg.wait(timeout=5)
                        jpeg = state.jpeg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, video file or stream URL (default: 0)")
    ap.add_argument("--replay", help="replay a detections.csv instead of running the model")
    ap.add_argument("--model", default=str(ROOT / "yolo26n.pt"))
    ap.add_argument("--tracker", default=str(ROOT / "tracktrac.yaml"))
    ap.add_argument("--csv", default="detections.csv", help="where live detections are logged")
    ap.add_argument("--entities", default="entities.json", help="where the track summary is saved")
    ap.add_argument("--fps", type=float, default=15, help="replay speed in frames per second")
    ap.add_argument("--size", default="640x480", help="frame size of the replayed recording")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    shown = re.sub(r"//[^/@]*@", "//", args.replay or args.source)   # hide stream credentials
    state = State("replay" if args.replay else "live", shown)
    start_detector(state, args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, args))
    server.daemon_threads = True
    print(f"Raqeeb dashboard: http://{args.host}:{args.port}  ({state.mode}: {state.source})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if not args.replay:
            save_entities(state, args.entities)
        print(f"Saved {len(state.entities)} unique objects")


if __name__ == "__main__":
    main()
