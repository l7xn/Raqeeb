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

import numpy as np # Added for color extraction

ROOT = Path(__file__).resolve().parent
PAGES = {"/": "raqeeb.html", "/raqeeb.html": "raqeeb.html"}
LOST_AFTER = 2.0   # seconds without a detection before a track is reported as lost
BAG_STATIONARY_TIME = 5.0 # seconds before an unattended bag is flagged
WEAPON_REARM_AFTER = 5.0  # rearm after the knife has been absent for five seconds

class State:
    """Detector state shared between the tracking thread and HTTP handlers."""

    def __init__(self, mode, source):
        self.lock = threading.Lock()
        self.new_jpeg = threading.Condition(self.lock)
        self.mode, self.source = mode, source
        self.status, self.error = "starting", None
        self.video = False
        self.jpeg = None
        self.tracked_bags = {} # For unattended bag detection
        self.reset()

    def reset(self):
        self.entities = {}   # track_id -> summary
        self.events = collections.deque(maxlen=200)
        self.weapon_last_seen = {}
        self.snapshots = collections.OrderedDict()
        self.seq = getattr(self, "seq", 0)
        self.dets = []
        self.frame = 0
        self.size = (640, 480)
        self.fps, self.last_t = 0.0, None

    def _event(self, kind, tid, e, event_type=None, severity=None, snapshot_jpeg=None):
        self.seq += 1
        ev_data = {"seq": self.seq, "kind": kind, "id": tid, "class": e["class"],
                   "time": e.get("last_seen", now_iso()), "conf": round(e.get("conf_sum", 0) / max(e.get("frames", 1), 1), 3)}
        
        if "color" in e:
             ev_data["color"] = e["color"]
             
        if event_type:
            ev_data["event_type"] = event_type
        if severity:
            ev_data["severity"] = severity
        if snapshot_jpeg is not None:
            self.snapshots[self.seq] = snapshot_jpeg
            ev_data["snapshot_url"] = f"/api/events/{self.seq}/snapshot.jpg"
            while len(self.snapshots) > 20:
                self.snapshots.popitem(last=False)

        self.events.append(ev_data)

    def update(self, now, frame_idx, size, dets, jpeg=None, knife_snapshot=None):
        """dets: list of (track_id, label, conf, [x1, y1, x2, y2], color)."""
        t = time.monotonic()
        with self.lock:
            if self.last_t is not None and t > self.last_t:
                inst = 1.0 / (t - self.last_t)
                self.fps = inst if not self.fps else 0.9 * self.fps + 0.1 * inst
            self.last_t, self.frame, self.size, self.status = t, frame_idx, size, "running"
            
            # Format detection for dashboard
            self.dets = [{"id": tid, "class": c, "conf": round(cf, 3), "box": [round(v, 1) for v in b]}
                         for tid, c, cf, b, _ in dets if tid != -1] # filter out untracked (weapons)
            
            for tid, label, cf, b, color in dets:
                # --- Emit once per continuous weapon appearance ---
                if tid == -1: # Convention: tid -1 is used for weapons
                    if str(label).strip().casefold() != "knife":
                        continue
                    last_seen = self.weapon_last_seen.get(label)
                    if last_seen is None or t - last_seen > WEAPON_REARM_AFTER:
                        self._event("alert", f"WPN-{label}-{int(t)}", {"class": label, "conf_sum": cf, "frames": 1},
                                    event_type="Knife Detected", severity="CRITICAL",
                                    snapshot_jpeg=knife_snapshot)
                    self.weapon_last_seen[label] = t
                    continue
                
                # --- Entity Tracking ---
                e = self.entities.get(tid)
                new = e is None or not e["active"]
                if e is None:
                    e = self.entities[tid] = {"class": label, "first_seen": now, "last_seen": now,
                                              "frames": 0, "conf_sum": 0.0, "active": True, "t": t}
                    if color: e["color"] = color

                e.update(last_seen=now, active=True, t=t)
                if color and color != "Unknown": e["color"] = color # update color if found
                e["frames"] += 1
                e["conf_sum"] += cf
                
                if new:
                    self._event("new", tid, e)
                    
                # --- Unattended Bag Logic ---
                if label in ['backpack', 'handbag', 'suitcase']:
                    bag_id = f"BAG-{tid}"
                    center_x = (b[0] + b[2]) / 2
                    center_y = (b[1] + b[3]) / 2
                    
                    if bag_id not in self.tracked_bags:
                        self.tracked_bags[bag_id] = {"first_seen": t, "last_pos": (center_x, center_y), "alerted": False}
                    else:
                        prev_pos = self.tracked_bags[bag_id]["last_pos"]
                        distance = ((center_x - prev_pos[0])**2 + (center_y - prev_pos[1])**2)**0.5
                        
                        if distance < 20: # bag is stationary
                            if t - self.tracked_bags[bag_id]["first_seen"] > BAG_STATIONARY_TIME and not self.tracked_bags[bag_id]["alerted"]:
                                self._event("alert", tid, e, event_type="Unattended Baggage", severity="HIGH")
                                self.tracked_bags[bag_id]["alerted"] = True
                        else:
                            # bag moved, reset timer
                            self.tracked_bags[bag_id]["first_seen"] = t
                            self.tracked_bags[bag_id]["last_pos"] = (center_x, center_y)

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
        with self.lock:
            self.status, self.error = status, error
            self._expire(float("inf"))

    def entities_json(self):
        with self.lock:
            return {tid: {k: e.get(k) for k in ("class", "first_seen", "last_seen", "frames", "color") if k in e}
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
                              "color": e.get("color", "Unknown"),
                              "conf": round(e["conf_sum"] / e["frames"], 3), "active": e["active"]}
                             for tid, e in sorted(self.entities.items())],
                "seq": self.seq, "events": [ev for ev in self.events if ev["seq"] > since],
            }


def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def get_dominant_color(frame, box):
    """Extracts dominant color from the upper half of a bounding box."""
    import cv2
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Upper half for shirts
    roi = frame[y1:y1 + int((y2-y1)/2), x1:x2]
    if roi.size == 0: return "Unknown"
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_color = np.average(np.average(hsv, axis=0), axis=0)
    hue = avg_color[0]
    
    if hue < 10 or hue > 160: return "Red"
    elif 10 <= hue < 35: return "Yellow"
    elif 35 <= hue < 85: return "Green"
    elif 85 <= hue < 130: return "Blue"
    else: return "Black/Dark"

def make_knife_snapshot(frame, box, confidence):
    import cv2

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    pad_x = max(40, (x2 - x1) // 2)
    pad_y = max(40, (y2 - y1) // 2)
    left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
    right, bottom = min(frame_width, x2 + pad_x), min(frame_height, y2 + pad_y)
    crop = frame[top:bottom, left:right].copy()
    if crop.size == 0:
        return None

    crop_height, crop_width = crop.shape[:2]
    box_start = (max(0, x1 - left), max(0, y1 - top))
    box_end = (min(crop_width - 1, x2 - left), min(crop_height - 1, y2 - top))
    cv2.rectangle(crop, box_start, box_end, (64, 221, 191), 2)
    cv2.putText(crop, f"KNIFE {confidence:.0%}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (64, 221, 191), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return encoded.tobytes() if ok else None

def run_live(state, args):
    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model) # General model
    try:
        model_weapon = YOLO(args.weapon_model) # Weapon model
        weapon_names = model_weapon.names
        knife_class_ids = [class_id for class_id, label in weapon_names.items()
                           if str(label).strip().casefold() == "knife"]
        if not knife_class_ids:
            raise ValueError("Weapon model has no Knife class")
        has_weapon_model = True
    except Exception as e:
        print(f"Warning: Weapon model not loaded: {e}. Running without weapons detection.")
        has_weapon_model = False

    names = model.names
    source = int(args.source) if args.source.isdigit() else args.source
    last_save = 0.0

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frame", "track_id", "class", "conf", "x1", "y1", "x2", "y2", "color"])
        frame_idx = 0
        try:
            for r in model.track(source=source, persist=True, tracker=args.tracker, stream=True, verbose=False):
                frame_idx += 1
                h, w = r.orig_shape
                orig_img = r.orig_img
                now = now_iso()
                dets = []
                knife_snapshot = None
                
                # --- 1. General Tracking (People, Bags, etc) ---
                boxes = r.boxes
                if boxes is not None and boxes.id is not None:
                    for tid, c, cf, b in zip(boxes.id.int().tolist(), boxes.cls.int().tolist(),
                                             boxes.conf.tolist(), boxes.xyxy.tolist()):
                        label = names[c]
                        color = "Unknown"
                        if label == 'person':
                            color = get_dominant_color(orig_img, b)
                            
                        writer.writerow([now, frame_idx, tid, label, round(cf, 3), *[round(v, 1) for v in b], color])
                        dets.append((tid, label, cf, b, color))
                
                # --- 2. Weapon Detection (No tracking needed, just detection) ---
                if has_weapon_model:
                    w_results = model_weapon(orig_img, verbose=False, classes=knife_class_ids)
                    if w_results[0].boxes is not None:
                        w_boxes = w_results[0].boxes
                        for c, cf, b in zip(w_boxes.cls.int().tolist(), w_boxes.conf.tolist(), w_boxes.xyxy.tolist()):
                             w_label = weapon_names[c]
                             if str(w_label).strip().casefold() != "knife":
                                 continue
                             if knife_snapshot is None:
                                 knife_snapshot = make_knife_snapshot(orig_img, b, cf)
                             # Use tid = -1 as a convention for weapons/untracked threats
                             dets.append((-1, w_label, cf, b, None))

                ok, buf = cv2.imencode(".jpg", r.plot(), [cv2.IMWRITE_JPEG_QUALITY, 75])
                state.update(now, frame_idx, (w, h), dets, buf.tobytes() if ok else None,
                             knife_snapshot=knife_snapshot)
                
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
                (int(row["track_id"]), row["class"], float(row["conf"]), box, row.get("color", "Unknown")))
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
        except Exception as e:  
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
            if p.startswith("/api/events/") and p.endswith("/snapshot.jpg"):
                parts = p.split("/")
                try:
                    event_seq = int(parts[3]) if len(parts) == 5 and parts[1:3] == ["api", "events"] else None
                except ValueError:
                    event_seq = None
                if event_seq is None:
                    return self.send(400, b"invalid event id", "text/plain")
                with state.lock:
                    image = state.snapshots.get(event_seq)
                if image is None:
                    return self.send(404, b"snapshot unavailable", "text/plain")
                return self.send(200, image, "image/jpeg",
                                 {"Content-Disposition": f"inline; filename=knife-alert-{event_seq}.jpg"})
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
            if p == "/api/frame.jpg":
                if not state.jpeg:
                    return self.send(404, b"no frame yet", "image/jpeg")
                return self.send(200, state.jpeg, "image/jpeg",
                                 {"Content-Disposition": "inline; filename=latest-frame.jpg"})
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
    ap.add_argument("--weapon_model", default=str(ROOT / "weapon.pt"), help="Path to weapon detection model") # ADDED WEAPON MODEL ARGUMENT
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
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, args))
    server.daemon_threads = True
    start_detector(state, args)
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