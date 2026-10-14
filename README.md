# Raqeeb (with chat bot)

Security presentation dashboard with recorded camera sources and actual local AI recognition.

## Start the presentation (Windows)

Double-click `start_raqeeb.cmd`, or run:

```powershell
.venv\Scripts\python.exe server.py
```

Open **http://localhost:8000**. Live Monitoring is the starting page. Wait for the camera tiles to show **AI READY**, then click **Start monitoring**. Use **Restart footage** to replay all clips and rearm presentation alerts. The first click also enables the alert sound; browser audio restrictions apply.

For a fresh installation:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe server.py
```

## Camera sources

The server discovers video files recursively in `examples/` and `samples/`. Each video becomes a camera with a persistent number. Deleting a source does not renumber the other cameras. `runtime/cameras.json` retains the assignments; `camera_sources.json` seeds CAM-01 and CAM-03 consistently on fresh installations. Add or replace videos and restart the server to refresh the list. Alternatively:

```powershell
.venv\Scripts\python.exe server.py --videos "Y:\your-videos"
```

Sources are labeled **RECORDED SOURCE**. There are no generated camera feeds, scripted incidents, random demo timers, simulated network logs, or fabricated analytics. Videos play muted and loop at their native frame rate. Other dashboard pages use these same camera sources, person tracks, and saved incidents.

## Recognition and alerts

- `yolo26n.pt` detects people and bags at 1280-pixel inference resolution. BoT-SORT uses motion and appearance features, with a longer track buffer, to retain person tracks through brief misses and occlusions.
- `weapon.pt` includes Pistol and Knife classes. These map to gun and knife critical alerts; Smartphone/Purse/Bill/Card classes do not trigger weapon alerts.
- Actual frames are analyzed locally at 12 samples per source second. Weapon detection uses a 1280-pixel full-frame pass, overlapping 960-pixel tiles, and enlarged person/hand regions in sparse scenes. All gun alerts require repeated observations of the same tracked holder; knife alerts require at least three observations and a higher score. Pose wrists and object size reject background objects and whole-body class confusion. Results are cached under `runtime/analysis/` to keep multi-camera playback smooth. Boxes and alerts follow the player's source timestamp, including after seeking.
- Person IDs such as `P-0001` are globally unique and stored in `runtime/person_ids.json`, independently of camera numbering. The same analyzed track keeps its ID across frames, replay, and server restarts. Appearance-assisted tracking reduces ID changes; these are anonymous tracking IDs, not verified identities or automatic cross-camera identity matches.
- `yolo26n-pose.pt` verifies visible wrists and assists holder association. Adjacent ambiguous hand overlaps retain a supported holder.
- People Tracking shows actual person crops with download and footage-review actions. Conservative appearance and motion matching consolidate nonoverlapping track fragments; simultaneously visible people stay separate. Boxes are smoothed and interpolated during playback.
- Weapon alerts include the associated anonymous person ID. Critical/high/medium/low severities have prominent distinct colors throughout notifications and review.
- A weapon box that substantially duplicates an independently detected whole person is rejected as a class confusion. A continuous weapon appearance produces one incident rather than a popup every frame.
- Alerts include the camera, source timestamp, confidence, and an actual annotated source-frame snapshot. Incidents and their review status persist in `runtime/incidents.json`; snapshots live in `runtime/evidence/`.
- Unsupported alerts from older analyses are archived as False Positive, preserving review history. They are excluded from live activity and notifications.
- Each weapon appearance alerts once per browser session. **Restart footage** rearms popups without duplicating saved incidents. The footage continues playing while an operator reviews an alert.
- Person Search uses Gemini to answer questions about clothing, hands visible in snapshots, person tracks, incident priority, source time, and detected weapon type. Answers include the related person and incident snapshots when available. Clothing colors are approximate automated metadata; times refer to seconds in the recorded clip, not wall-clock time.
- Copy `.env.example` to `.env` and enter your Gemini key in `.env` before starting Raqeeb. `.env` is ignored by Git; an existing server environment variable takes precedence. The key stays on the server and is never sent to the browser. Questions send track/incident metadata and up to 12 snapshots to Google's Gemini API; do not enable this for footage you are not allowed to share with Google.

Model scores are not guarantees. A small or briefly visible weapon may be missed, and false positives remain possible. Analysis samples are not every video frame. Tune sampling and confidence when needed:

```powershell
.venv\Scripts\python.exe server.py --analysis-fps 12 --weapon-conf 0.35 --imgsz 1280
```

Changing the models, source files, tracker, analysis resolution, sample rate, or threshold invalidates the corresponding analysis cache automatically.

## Files

| File | Purpose |
| --- | --- |
| `raqeeb.html` | Accessible dashboard shell and centered search |
| `ui.css` | Shared responsive dark/light design system |
| `dashboard-ui.js` | Navigation, metrics, charts, directories, and review layouts |
| `app.js` | Camera playback, synchronized boxes/alerts, search, and incident review |
| `server.py` | Server command-line entry point |
| `video_cameras.py` | Video discovery, YOLO analysis, evidence, HTTP range streaming, and persistence |
| `start_raqeeb.cmd` | Windows launch shortcut |
| `tracking_cleanup.py` | Fragment consolidation, unique track rows, and stable incident identifiers |
| `camera_sources.json` | Portable initial camera assignments |
| `yolo26n-pose.pt` | Wrist and holder validation |
| `person_tracker.yaml` | Appearance-assisted person tracking settings |
| `yolo26n-cls.pt` | Lightweight appearance encoder used by the tracker |
| `raqeeb.py` | Separate legacy webcam tracking script; not used by this dashboard |
| `samples/`, `examples/` | Input video folders (either may be absent) |
| `tests/test_video_cameras.py` | Incident provenance, persistence, source discovery, and HTTP tests |

The older root/sample CSV and JSON recordings are not loaded by the dashboard.

## Checks

```powershell
python -m unittest discover -s tests -v
node --check app.js
python -m py_compile server.py video_cameras.py tracking_cleanup.py
```

Detection uses the [Ultralytics prediction API](https://docs.ultralytics.com/modes/predict/); appearance-assisted association uses [BoT-SORT tracking](https://docs.ultralytics.com/modes/track/).

The supplied videos can also be checked in Chrome with Node.js 24+. Run the server in a separate terminal with an isolated runtime (the browser check creates test incidents):

```powershell
.venv\Scripts\python.exe server.py --port 8001 --runtime-dir runtime/validation
node tests/browser_monitoring.mjs
```

The browser check covers decoded playback, person screenshots, unique track rows, both CAM-03 gun holders, critical alerts and evidence, acknowledgement, replay deduplication, navigation, RTL, themes, search, and mobile overflow. Screenshots are saved in the system temporary directory under `raqeeb-browser-qa`. `CHROME_PATH`, `RAQEEB_QA_URL`, and `RAQEEB_QA_OUTPUT` can override the defaults.

## Professional dashboard UI

The console uses grouped navigation, a centered search field, consistent line icons, soft panel corners, and a dark palette inspired by the supplied dashboard reference. Every screen also supports a light theme and Arabic RTL layout.

Overview and Analytics show real person-track distribution, camera readiness, observed incident severity, per-camera incident counts, and sampled person counts over source time. Charts never invent activity for unobserved footage. People Tracking includes a paginated snapshot directory; alerts and incident history retain their existing review and evidence behavior. Camera playback, AI inference, tracking, IDs, weapon confirmation, and incident persistence are unchanged by this redesign.

The browser check captures each screen and verifies chart totals, all-screen mobile layout, Arabic mobile layout, and the existing camera/alert workflow. UI styling is in `ui.css`; reusable presentation components are in `dashboard-ui.js`.
