"""
ShelfHeat AI - MVP backend
--------------------------
Reads a CCTV video, detects ONLY people (YOLOv8s, class 0), tracks their
body area (smoothed bounding box), accumulates a density map over time,
overlays a JET heatmap on the frame and streams it as MJPEG through FastAPI.

Privacy (PDPA by design):
  * heads are pixelated in every output frame (toggle: POST /api/toggle-privacy)
  * heatmap overlay can be hidden without stopping analytics (toggle: POST /api/toggle-heatmap)
  * no frame, crop or face is ever written to disk
  * only anonymous (x, y) foot points and counters are kept in memory

Run:  uvicorn main:app --reload      (see README / instructions)
"""
import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

# --------------------------------------------------------------------------
# Configuration (override with environment variables)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
VIDEO_SOURCE = os.getenv("VIDEO_PATH", str(BASE_DIR / "videos" / "sample.mp4"))  # or "0" for webcam
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8s.pt")   # auto-downloads on first run
CONF = float(os.getenv("CONF", "0.40"))              # min confidence for counting / heatmap / stats
BLUR_CONF = float(os.getenv("BLUR_CONF", "0.25"))    # min confidence for PDPA head blurring (lower = safer)
IMGSZ = int(os.getenv("IMGSZ", "512"))               # lower = faster on CPU
MAX_FPS = float(os.getenv("MAX_FPS", "6"))           # processing cap: fixed frame interval for a steady stream
TRACK_BUFFER = int(os.getenv("TRACK_BUFFER", "60"))  # frames ByteTrack keeps a lost track (fewer ID switches)
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "960"))   # all frames are resized to this width
JPEG_QUALITY = 80

HEAT_SCALE = 0.25                                    # heat map is kept at 1/4 resolution (fast)
BODY_FILL_X = 0.85                                   # heat ellipse width as a share of the box width
BODY_FILL_Y = 0.95                                   # heat ellipse height as a share of the box height
MIN_BODY_AXIS = 10                                   # px (full-res) smallest ellipse half-axis
HEAT_SIGMA = 6                                       # blur in heat-map pixels (~24 px full-res)
HEAT_ALPHA = 0.62                                    # max overlay opacity
HEAT_HALF_LIFE = 12.0                               # seconds of video until old traffic counts half
HISTORY_WEIGHT = 0.8                                 # accumulated traffic tops out below the live glow
LIVE_SIGMA = 4                                       # blur in heat-map pixels for the live body glow
SMOOTH_TAU = 0.35                                    # EMA time constant (s) for boxes; higher = steadier
MIN_TRACK_SECONDS = 1.0                              # a track must last this long to count as a visitor ...
MIN_TRACK_FRAMES = 8                                 # ... and be seen in at least this many processed frames
TRACK_TTL = 10.0                                     # seconds unseen before a track is pruned
RELINK_GAP = 5.0                                     # a new track starting <= this long after a counted one was lost...
RELINK_DIST = 100                                    # ...and this close (px, for a 200 px tall person) is the same visitor

SHOW_BOXES = os.getenv("SHOW_BOXES", "1") == "1"
SHOW_HEATMAP = os.getenv("SHOW_HEATMAP", "1") == "1"
PRIVACY_BLUR = os.getenv("PRIVACY_BLUR", "1") == "1"

ORANGE_BGR = (53, 107, 255)                          # #FF6B35
AISLES = ["Beverages", "Snacks", "Dairy", "Bakery"]   # camera view split into vertical strips, left to right

log = logging.getLogger("uvicorn.error")
TRACKER_CFG = IterableSimpleNamespace(**{**YAML.load(check_yaml("bytetrack.yaml")), "track_buffer": TRACK_BUFFER})


# --------------------------------------------------------------------------
# Heatmap engine: one background thread, shared by every browser tab
# --------------------------------------------------------------------------
class HeatmapEngine:
    def __init__(self):
        self.model = None
        self.tracker = None       # ByteTrack, fed only detections >= CONF
        self._thread = None
        self._stop = threading.Event()
        self._reset_requested = False
        self._lock = threading.Lock()

        self._jpg = None
        self._frame_id = 0
        self._stats = self._empty_stats("starting", "Loading model...")

        self.heat = None
        self.aisle_heat = None
        self.tracks = {}  # (loop, track_id) -> [first_s, last_s, frames, counted, first_foot, last_foot, height, relinked]
        self._count_loop = 0      # only this pass of the clip adds visitors (no recount when it loops)
        self._visitors = 0
        self._dwell_sum = 0.0     # dwell of counted tracks that were already pruned
        self.feet = {}    # (loop, track_id) -> [x1, y1, x2, y2 (smoothed), last_frame_idx]
        self._peak = 0.0
        self._last_idx = None
        self._proc_fps = 0.0
        self._last_done = None    # time the previous frame finished (fps = frames actually delivered per second)
        self.privacy = PRIVACY_BLUR
        self.show_heatmap = SHOW_HEATMAP

    # ---- public API -------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def reset(self):
        self._reset_requested = True

    def toggle_privacy(self):
        self.privacy = not self.privacy
        return self.privacy

    def toggle_heatmap(self):
        self.show_heatmap = not self.show_heatmap
        return self.show_heatmap

    def get_frame(self):
        with self._lock:
            return self._jpg, self._frame_id

    def get_stats(self):
        with self._lock:
            stats = dict(self._stats)
        stats["privacy"] = self.privacy
        stats["heatmap"] = self.show_heatmap
        stats["model"] = Path(MODEL_NAME).stem   # read-only config, for the dashboard status bar
        stats["conf"] = CONF
        return stats

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _empty_stats(status, message=""):
        return {
            "status": status,
            "message": message,
            "total_visitors": 0,
            "current_visitors": 0,
            "top_aisle": "-",
            "top_aisle_share": 0.0,
            "avg_dwell_seconds": 0.0,
            "fps": 0.0,
            "aisles": [{"aisle": a, "share": 0.0} for a in AISLES],
        }

    def _publish_message(self, lines, status="error"):
        img = np.full((540, FRAME_WIDTH, 3), (42, 23, 15), np.uint8)  # navy
        y = 240
        for line in lines:
            cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 237, 247), 1, cv2.LINE_AA)
            y += 36
        ok, buf = cv2.imencode(".jpg", img)
        with self._lock:
            if ok:
                self._jpg = buf.tobytes()
                self._frame_id += 1
            self._stats = self._empty_stats(status, " ".join(lines))

    def _relink(self, key, tr):
        """True if this newly counted track continues a counted one that was lost moments ago nearby (ID switch)."""
        for k, c in self.tracks.items():
            if k == key or k[0] != key[0] or not c[3] or c[7]:
                continue
            if 0 <= tr[0] - c[1] <= RELINK_GAP and np.hypot(tr[4][0] - c[5][0], tr[4][1] - c[5][1]) <= RELINK_DIST * max(1.0, c[6] / 200):
                c[7] = True   # each lost track can be continued only once
                return True
        return False

    @staticmethod
    def _brackets(img, x1, y1, x2, y2, color):
        """Four rounded corner brackets instead of a full rectangle."""
        length = int(np.clip(0.22 * min(x2 - x1, y2 - y1), 6, 24))
        r = max(1, min(5, length // 2))
        # (corner x, corner y, x direction, y direction, arc start angle)
        for cx, cy, sx, sy, a0 in ((x1, y1, 1, 1, 180), (x2, y1, -1, 1, 270), (x1, y2, 1, -1, 90), (x2, y2, -1, -1, 0)):
            cv2.ellipse(img, (cx + sx * r, cy + sy * r), (r, r), 0, a0, a0 + 90, color, 2, cv2.LINE_AA)
            cv2.line(img, (cx + sx * r, cy), (cx + sx * length, cy), color, 2, cv2.LINE_AA)
            cv2.line(img, (cx, cy + sy * r), (cx, cy + sy * length), color, 2, cv2.LINE_AA)

    @staticmethod
    def _pixelate(frame, x1, y1, x2, y2):
        x1, y1 = max(0, x1), max(0, y1)
        roi = frame[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return
        small = cv2.resize(roi, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_LINEAR)
        frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # ---- main loop --------------------------------------------------------
    def _run(self):
        try:
            self.model = YOLO(MODEL_NAME)
        except Exception as exc:  # e.g. offline on first run
            self._publish_message(["Could not load YOLO model.", str(exc)[:80]])
            return

        source = int(VIDEO_SOURCE) if str(VIDEO_SOURCE).isdigit() else VIDEO_SOURCE
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self._publish_message(
                ["Video not found.", f"Put a CCTV clip at: {VIDEO_SOURCE}", "or set VIDEO_PATH=/path/to/video.mp4"]
            )
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 1 else 25.0
        frame_dt = 1.0 / fps
        loop, frame_idx, failed_reads = 0, 0, 0
        is_live = isinstance(source, int)
        start = time.time()

        while not self._stop.is_set():
            try:
                t0 = time.time()

                if self._reset_requested:
                    self.heat = None
                    self.aisle_heat = None
                    self.tracks = {}
                    self.feet = {}
                    self._peak = 0.0
                    self._count_loop = loop
                    self._visitors = 0
                    self._dwell_sum = 0.0
                    self._reset_requested = False

                # stay in sync with real time: skip frames when YOLO is slower than the video,
                # wait when it is faster (a webcam is always live, so no skipping there)
                if not is_live:
                    behind = int((time.time() - start) * fps) - frame_idx
                    if behind > 0:
                        for _ in range(min(behind, int(fps))):
                            if not cap.grab():
                                break
                            frame_idx += 1
                        if behind > fps:  # far behind: re-anchor instead of chasing forever
                            start = time.time() - frame_idx / fps
                    elif behind < 0:
                        time.sleep(-behind * frame_dt)

                ok, frame = cap.read()
                if not ok:
                    failed_reads += 1
                    if is_live or failed_reads > 3:
                        self._publish_message(["Cannot read frames from the video source."])
                        return
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the clip
                    loop += 1
                    frame_idx, start = 0, time.time()
                    self._last_idx = None
                    self.feet = {}
                    self.tracker = None                  # restart tracker IDs
                    continue
                failed_reads = 0

                self._process(frame, loop, frame_idx, fps)
                frame_idx += 1

                # fixed frame interval: wait out the rest of the 1/MAX_FPS slot so the stream is steady
                if MAX_FPS > 0:
                    rest = 1.0 / MAX_FPS - (time.time() - t0)
                    if rest > 0:
                        time.sleep(rest)
                now = time.time()
                if self._last_done is not None:
                    self._proc_fps = 0.9 * self._proc_fps + 0.1 * (1.0 / max(now - self._last_done, 1e-3))
                self._last_done = now
            except Exception as exc:  # keep the loop alive; the next good frame sets status back to "running"
                log.exception("Frame processing failed")
                with self._lock:
                    self._stats = dict(self._stats, status="error", message=f"Processing error: {exc}"[:200])
                frame_idx += 1
                time.sleep(0.5)

        cap.release()

    def _process(self, frame, loop, frame_idx, fps):
        h0, w0 = frame.shape[:2]
        if w0 != FRAME_WIDTH:
            frame = cv2.resize(frame, (FRAME_WIDTH, int(h0 * FRAME_WIDTH / w0)))
        H, W = frame.shape[:2]

        hh, hw = max(1, int(H * HEAT_SCALE)), max(1, int(W * HEAT_SCALE))
        if self.heat is None or self.heat.shape != (hh, hw):
            self.heat = np.zeros((hh, hw), np.float32)
            self.aisle_heat = np.zeros(len(AISLES), np.float64)
        dt = 1.0 / fps if self._last_idx is None else max(1, frame_idx - self._last_idx) / fps
        self._last_idx = frame_idx
        decay = 0.5 ** (dt / HEAT_HALF_LIFE)       # time-based, so skipped frames fade the same
        self.heat *= decay
        self.aisle_heat *= decay

        # 1) detect ONLY persons (COCO class 0) once at the blur threshold; track only confident ones
        det = self.model.predict(
            frame, classes=[0], conf=min(BLUR_CONF, CONF), imgsz=IMGSZ, verbose=False,
        )[0].boxes.cpu().numpy()
        if self.tracker is None:
            self.tracker = BYTETracker(TRACKER_CFG)
        tracked = self.tracker.update(det[det.conf >= CONF], frame)

        # 2) body areas (box smoothed per track) -> history + live glow
        scratch = np.zeros_like(self.heat)
        live = np.zeros_like(self.heat)
        t_now = frame_idx / fps
        people = []
        # every detection >= BLUR_CONF is pixelated (raw box, no smoothing lag), tracked or not
        blur = [tuple(int(v) for v in np.clip(b, 0, [W - 1, H - 1, W - 1, H - 1])) for b in det.xyxy]
        if len(tracked):
            for x1, y1, x2, y2, tid in tracked[:, :5]:
                x1, x2 = float(np.clip(x1, 0, W - 1)), float(np.clip(x2, 0, W - 1))
                y1, y2 = float(np.clip(y1, 0, H - 1)), float(np.clip(y2, 0, H - 1))
                key = (loop, int(tid))
                prev = self.feet.get(key)
                if prev is not None and frame_idx - prev[4] <= fps:
                    # time-based EMA: same steadiness whether we process 3 or 25 frames/s
                    k = 1.0 - np.exp(-((frame_idx - prev[4]) / fps) / SMOOTH_TAU)
                    x1, y1, x2, y2 = (p + k * (v - p) for p, v in zip(prev[:4], (x1, y1, x2, y2)))
                self.feet[key] = [x1, y1, x2, y2, frame_idx]
                foot = ((x1 + x2) / 2, y2)
                tr = self.tracks.get(key)
                if tr is None:
                    tr = self.tracks[key] = [t_now, t_now, 1, False, foot, foot, y2 - y1, False]
                else:
                    tr[1], tr[5], tr[6] = t_now, foot, max(tr[6], y2 - y1)
                    tr[2] += 1
                if (not tr[3] and tr[1] - tr[0] >= MIN_TRACK_SECONDS and tr[2] >= MIN_TRACK_FRAMES
                        and loop == self._count_loop):
                    tr[3] = True
                    if not self._relink(key, tr):
                        self._visitors += 1
                fx, fy = (x1 + x2) / 2, y2
                # sub-pixel body ellipse in heat-map coords (exact W/H ratio, so the glow lines up after upscaling)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                c = (int(round(((cx + 0.5) * hw / W - 0.5) * 16)), int(round(((cy + 0.5) * hh / H - 0.5) * 16)))
                ax = (int(round(max(MIN_BODY_AXIS, (x2 - x1) / 2 * BODY_FILL_X) * hw / W * 16)),
                      int(round(max(MIN_BODY_AXIS, (y2 - y1) / 2 * BODY_FILL_Y) * hh / H * 16)))
                cv2.ellipse(scratch, c, ax, 0, 0, 360, dt, -1, cv2.LINE_AA, 4)   # weight by time spent there
                cv2.ellipse(live, c, ax, 0, 0, 360, 1.0, -1, cv2.LINE_AA, 4)
                self.aisle_heat[min(int(fx / W * len(AISLES)), len(AISLES) - 1)] += dt
                people.append((int(x1), int(y1), int(x2), int(y2), int(fx), int(fy)))
        self.heat += scratch
        self.feet = {k: v for k, v in self.feet.items() if frame_idx - v[4] <= 2 * fps}
        for key in [k for k, t in self.tracks.items() if k[0] != loop or t_now - t[1] > TRACK_TTL]:
            first, last, _, counted, *_ = self.tracks.pop(key)
            if counted:
                self._dwell_sum += last - first

        # 3) privacy: pixelate heads BEFORE anything is drawn or streamed
        if self.privacy:
            for x1, y1, x2, y2 in blur:
                # boxes cut off by the frame edge are short, so also size the head from the box width
                head_h = min(y2 - y1, max(int((y2 - y1) * 0.28), int((x2 - x1) * 0.6)))
                self._pixelate(frame, x1, y1, x2, y1 + head_h)

        # 4) heatmap overlay: accumulated traffic + a glow that follows each person right now
        hist = cv2.GaussianBlur(self.heat, (0, 0), HEAT_SIGMA)
        self._peak = max(float(hist.max()), self._peak * 0.5 ** (dt / HEAT_HALF_LIFE))  # steady scale, fades with the heat
        if self._peak > 1e-6:
            hist /= self._peak
        live = cv2.GaussianBlur(live, (0, 0), LIVE_SIGMA)
        if live.max() > 1e-6:
            live /= live.max()
        level = np.clip(np.maximum(hist * HISTORY_WEIGHT, live), 0, 1)
        if self.show_heatmap and level.max() > 1e-3:
            color = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_JET)
            ramp = np.clip((level - 0.05) / 0.45, 0, 1)
            alpha = ramp * ramp * (3 - 2 * ramp) * HEAT_ALPHA      # smoothstep: soft edges, solid core
            color = cv2.resize(color, (W, H), interpolation=cv2.INTER_CUBIC)
            alpha = np.clip(cv2.resize(alpha, (W, H), interpolation=cv2.INTER_CUBIC), 0, 1)[..., None]
            frame = (frame * (1 - alpha) + color * alpha).astype(np.uint8)

        # 5) rounded corner brackets + foot points (no faces), drawn on a copy and blended at ~85% opacity
        if people:
            overlay = frame.copy()
            for x1, y1, x2, y2, fx, fy in people:
                if SHOW_BOXES:
                    self._brackets(overlay, x1, y1, x2, y2, ORANGE_BGR)
                cv2.circle(overlay, (fx, fy), 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, (fx, fy), 4, ORANGE_BGR, 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # 6) stats
        active = [t[1] - t[0] for t in self.tracks.values() if t[3]]
        total_aisle = float(self.aisle_heat.sum())
        top_idx = int(self.aisle_heat.argmax()) if total_aisle > 0 else -1
        aisles = [
            {"aisle": a, "share": round(100 * self.aisle_heat[i] / total_aisle, 1) if total_aisle else 0.0}
            for i, a in enumerate(AISLES)
        ]
        stats = {
            "status": "running",
            "message": "",
            "total_visitors": self._visitors,
            "current_visitors": len(people),
            "top_aisle": AISLES[top_idx] if top_idx >= 0 else "-",
            "top_aisle_share": aisles[top_idx]["share"] if top_idx >= 0 else 0.0,
            # re-linked fragments add their time to the same visitor
            "avg_dwell_seconds": round((self._dwell_sum + sum(active)) / self._visitors, 1) if self._visitors else 0.0,
            "fps": round(self._proc_fps, 1),
            "aisles": aisles,
        }

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with self._lock:
                self._jpg = buf.tobytes()
                self._frame_id += 1
                self._stats = stats


engine = HeatmapEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.start()
    yield
    engine.stop()


app = FastAPI(title="ShelfHeat AI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def dashboard():
    return FileResponse(BASE_DIR / "static" / "index.html")


async def mjpeg_generator():
    last_id = -1
    while True:
        jpg, frame_id = engine.get_frame()
        if jpg is not None and frame_id != last_id:
            last_id = frame_id
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        await asyncio.sleep(0.02)


@app.get("/video_feed")
def video_feed():
    """MJPEG stream: every frame has been through YOLO + heatmap + head pixelation."""
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/stats")
def stats():
    """Live analytics computed from the video being streamed."""
    return engine.get_stats()


@app.get("/api/stats/mock")
def stats_mock():
    """Static mock numbers (handy for UI design / offline pitch fallback)."""
    return {"total_visitors": 120, "top_aisle": "Beverages", "avg_dwell_seconds": 94}


@app.post("/api/reset")
def reset():
    engine.reset()
    return {"ok": True}


@app.post("/api/toggle-privacy")
def toggle_privacy():
    """Turn head pixelation on/off in the live stream."""
    return {"privacy": engine.toggle_privacy()}


@app.post("/api/toggle-heatmap")
def toggle_heatmap():
    """Turn the heatmap overlay on/off in the live stream (analytics keep running)."""
    return {"heatmap": engine.toggle_heatmap()}
