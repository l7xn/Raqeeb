# Raqeeb
project for hackathon expo 2030

## What's here

| File | Purpose |
| --- | --- |
| `raqeeb.html` | SentinelOps security dashboard (the website) |
| `server.py` | Backend: runs the AI detector and serves the dashboard with live detections |
| `raqeeb.py` | Standalone tracker script (webcam, OpenCV window, writes `detections.csv` / `entities.json`) |
| `yolo26n.pt` | YOLO26n detection model |
| `tracktrac.yaml` | TrackTrack multi-object tracker settings |
| `samples/` | A recorded `detections.csv` / `entities.json` for demos without a camera |

## Run the dashboard with live AI detections

```bash
pip install -r requirements.txt
python server.py                         # webcam 0
python server.py --source video.mp4      # video file, RTSP/HTTP stream, or another camera index
python server.py --replay samples/detections.csv   # replay a recording (no camera or model needed)
```

Open http://localhost:8000. The **AI Detector** page shows the camera feed with tracked boxes,
detector status and every tracked object. Live detections also appear in **Live Monitoring**,
**People Tracking**, the overview's live event feed and notifications. The rest of the dashboard
keeps running on demo data.

While the server runs it logs to `detections.csv` and `entities.json` (same format as `raqeeb.py`);
both can be downloaded from the AI Detector page.

Opening `raqeeb.html` directly still works: it shows demo data and connects to
`http://localhost:8000` when the server is running. A different detector address can be set in
Settings or with `raqeeb.html?ai=http://host:port`. Use `--host 0.0.0.0` to reach the server from
other devices.
