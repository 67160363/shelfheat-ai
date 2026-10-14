"""
ShelfHeat AI - system health check
----------------------------------
Plain-Python test runner (no pytest needed). Three groups:
  * live   - HTTP checks against a running server (BASE_URL, default http://127.0.0.1:8000)
  * model  - YOLOv8s load, CPU inference speed, box coordinate sanity on videos/sample.mp4
  * engine - HeatmapEngine math with a fake detector: body coverage, EMA, decay, toggles

Run:  python tests/health_check.py            (start uvicorn first for the live group)
"""
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import httpx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

RESULTS = []   # (group, name, ok, detail)
METRICS = {}


def check(group, name):
    def deco(fn):
        t0 = time.time()
        try:
            detail = fn() or ""
            RESULTS.append((group, name, True, detail))
        except Exception as exc:
            RESULTS.append((group, name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        print(f"[{'PASS' if RESULTS[-1][2] else 'FAIL'}] {group}/{name} ({time.time() - t0:.1f}s) {RESULTS[-1][3]}")
        return fn
    return deco


# --------------------------------------------------------------------------
# live server
# --------------------------------------------------------------------------
client = httpx.Client(base_url=BASE_URL, timeout=10)

STATS_TYPES = {
    "status": str, "message": str, "total_visitors": int, "current_visitors": int,
    "top_aisle": str, "top_aisle_share": (int, float), "avg_dwell_seconds": (int, float),
    "fps": (int, float), "aisles": list, "privacy": bool, "heatmap": bool,
}


@check("live", "GET / serves dashboard")
def _():
    r = client.get("/")
    assert r.status_code == 200, r.status_code
    assert "text/html" in r.headers["content-type"]
    for el in ("privacyBtn", "heatmapBtn", "resetBtn", "/video_feed", "/api/stats"):
        assert el in r.text, f"missing {el}"


@check("live", "GET /api/stats schema")
def _():
    t0 = time.time()
    r = client.get("/api/stats")
    METRICS["stats_latency_ms"] = round((time.time() - t0) * 1000, 1)
    assert r.status_code == 200, r.status_code
    d = r.json()
    for k, t in STATS_TYPES.items():
        assert k in d, f"missing key {k}"
        assert isinstance(d[k], t), f"{k} is {type(d[k]).__name__}"
    assert d["status"] == "running", f"status={d['status']} ({d['message']})"
    assert [a["aisle"] for a in d["aisles"]] == ["Beverages", "Snacks", "Dairy", "Bakery"]
    total = sum(a["share"] for a in d["aisles"])
    assert total == 0 or abs(total - 100) < 0.5, f"aisle shares sum to {total}"
    assert d["current_visitors"] >= 0 and d["total_visitors"] >= 0
    METRICS["server_fps"] = d["fps"]
    METRICS["server_people_in_view"] = d["current_visitors"]
    return f"fps={d['fps']} in_view={d['current_visitors']} total={d['total_visitors']}"


@check("live", "GET /api/stats/mock")
def _():
    r = client.get("/api/stats/mock")
    assert r.status_code == 200 and set(r.json()) == {"total_visitors", "top_aisle", "avg_dwell_seconds"}


def _toggle(path, key):
    before = client.get("/api/stats").json()[key]
    a = client.post(path)
    b = client.post(path)
    assert a.status_code == b.status_code == 200
    va, vb = a.json()[key], b.json()[key]
    assert isinstance(va, bool) and va == (not before) and vb == before, (before, va, vb)
    assert client.get("/api/stats").json()[key] == before, "state not restored"
    return f"{before} -> {va} -> {vb} (restored)"


@check("live", "POST /api/toggle-privacy")
def _():
    return _toggle("/api/toggle-privacy", "privacy")


@check("live", "POST /api/toggle-heatmap")
def _():
    return _toggle("/api/toggle-heatmap", "heatmap")


@check("live", "GET /api/toggle-* rejects GET")
def _():
    for p in ("/api/toggle-privacy", "/api/toggle-heatmap", "/api/reset"):
        assert client.get(p).status_code == 405, p


@check("live", "GET /video_feed MJPEG frames")
def _():
    frames, buf, t0 = [], b"", time.time()
    with client.stream("GET", "/video_feed", timeout=15) as r:
        assert r.status_code == 200, r.status_code
        assert r.headers["content-type"].startswith("multipart/x-mixed-replace; boundary=frame")
        for chunk in r.iter_bytes():
            buf += chunk
            while True:
                s = buf.find(b"\xff\xd8")
                e = buf.find(b"\xff\xd9", s + 2)
                if s < 0 or e < 0:
                    break
                frames.append(buf[s:e + 2])
                buf = buf[e + 2:]
            if len(frames) >= 15 or time.time() - t0 > 8:
                break
    dt = time.time() - t0
    assert frames, "no JPEG frames received"
    img = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[1] == 960, img.shape if img is not None else None
    METRICS["stream_fps"] = round(len(frames) / dt, 1)
    METRICS["stream_frame_kb"] = round(np.mean([len(f) for f in frames]) / 1024, 1)
    METRICS["stream_resolution"] = f"{img.shape[1]}x{img.shape[0]}"
    return f"{len(frames)} frames in {dt:.1f}s, {img.shape[1]}x{img.shape[0]}"


@check("live", "static assets")
def _():
    r = client.get("/static/index.html")
    assert r.status_code == 200 and "<!DOCTYPE html>" in r.text
    r = client.get("/static/logo.png")
    assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n", r.status_code
    assert client.get("/static/does-not-exist.png").status_code == 404
    return f"logo {len(r.content) // 1024} KB"


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
import main  # noqa: E402  (imports ultralytics; the engine thread is NOT started on import)

MODEL = {}


@check("model", "YOLOv8s loads")
def _():
    from ultralytics import YOLO
    t0 = time.time()
    MODEL["m"] = YOLO(str(ROOT / main.MODEL_NAME))
    METRICS["model_load_s"] = round(time.time() - t0, 2)
    assert MODEL["m"].names[0] == "person"
    return f"{main.MODEL_NAME} in {METRICS['model_load_s']}s, {len(MODEL['m'].names)} classes, torch {torch.__version__}"


def _sample_frames(n):
    cap = cv2.VideoCapture(main.VIDEO_SOURCE)
    assert cap.isOpened(), main.VIDEO_SOURCE
    METRICS["video"] = (f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                        f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} fps, {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        out.append(cv2.resize(f, (main.FRAME_WIDTH, int(h * main.FRAME_WIDTH / w))))
    cap.release()
    assert len(out) == n, f"only {len(out)} frames"
    return out


@check("model", "inference speed + box coordinates")
def _():
    m = MODEL["m"]
    frames = _sample_frames(33)
    for f in frames[:3]:   # warm-up
        m.predict(f, classes=[0], conf=main.CONF, imgsz=main.IMGSZ, verbose=False)
    times, counts = [], []
    for f in frames[3:]:
        t0 = time.perf_counter()
        r = m.predict(f, classes=[0], conf=main.CONF, imgsz=main.IMGSZ, verbose=False)[0]
        times.append(time.perf_counter() - t0)
        H, W = f.shape[:2]
        if len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            xyxyn = r.boxes.xyxyn.cpu().numpy()
            assert (r.boxes.cls.cpu().numpy() == 0).all(), "non-person class returned"
            assert (xyxyn >= -1e-4).all() and (xyxyn <= 1 + 1e-4).all(), "normalized coords out of [0,1]"
            assert np.allclose(xyxyn * [W, H, W, H], xyxy, atol=1.0), "xyxyn * frame size != xyxy"
            assert (xyxy[:, 2] > xyxy[:, 0]).all() and (xyxy[:, 3] > xyxy[:, 1]).all()
        counts.append(len(r.boxes))
    ms = np.array(times) * 1000
    METRICS["infer_ms_mean"] = round(float(ms.mean()), 1)
    METRICS["infer_ms_p95"] = round(float(np.percentile(ms, 95)), 1)
    METRICS["infer_fps"] = round(1000 / float(ms.mean()), 1)
    METRICS["people_per_frame"] = round(float(np.mean(counts)), 1)
    assert sum(counts) > 0, "no people detected in 30 frames of the sample video"
    return f"{METRICS['infer_fps']} FPS ({METRICS['infer_ms_mean']} ms mean, p95 {METRICS['infer_ms_p95']} ms), {METRICS['people_per_frame']} people/frame"


@check("model", "ByteTrack assigns IDs")
def _():
    m = MODEL["m"]
    m.predictor = None
    frames = _sample_frames(10)
    ids = set()
    for f in frames:
        r = m.track(f, classes=[0], conf=main.CONF, imgsz=main.IMGSZ, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        if r.boxes.id is not None:
            ids.update(r.boxes.id.int().tolist())
    m.predictor = None
    assert ids, "tracker produced no IDs"
    return f"{len(ids)} track IDs over 10 frames"


# --------------------------------------------------------------------------
# engine math (fake detector, no video)
# --------------------------------------------------------------------------
from ultralytics.engine.results import Boxes  # noqa: E402


class _Result:
    def __init__(self, boxes):
        # tracked-style rows [x1, y1, x2, y2, id, conf, cls]; conf defaults to 0.9
        rows = [[*b[:4], b[4], b[5] if len(b) > 5 else 0.9, 0] for b in boxes]
        self.boxes = Boxes(torch.tensor(rows, dtype=torch.float32).reshape(-1, 7), (540, 960))


class FakeModel:
    def __init__(self):
        self.boxes = []

    def predict(self, frame, **kw):
        return [_Result(self.boxes)]


class FakeTracker:
    """Stands in for ByteTrack: echoes the fed detections with their preset IDs, records how many it got."""
    def __init__(self):
        self.fed = []

    def update(self, det, img=None):
        self.fed.append(len(det))
        if not len(det):
            return np.empty((0, 8))
        return np.c_[det.xyxy, det.id, det.conf, det.cls, np.arange(len(det))]


def _engine(privacy=False):
    e = main.HeatmapEngine()
    e.model = FakeModel()
    e.tracker = FakeTracker()
    e.privacy = privacy
    return e


FPS = 25.0
BLANK = np.full((540, 960, 3), 90, np.uint8)
BOX = (400, 120, 520, 460, 1)   # x1, y1, x2, y2, id -> 120 x 340 px person


def _out(e):
    jpg, _ = e.get_frame()
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


@check("engine", "heat covers the body box, not just the feet")
def _():
    e = _engine()
    e.model.boxes = [BOX]
    for i in range(10):
        e._process(BLANK.copy(), 0, i, FPS)
    heat = cv2.resize(e.heat, (960, 540), interpolation=cv2.INTER_LINEAR)
    x1, y1, x2, y2, _ = BOX
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    peak = heat.max()
    inside = heat[y1 + 20:y2 - 20, x1 + 20:x2 - 20]
    assert heat[cy, cx] > 0.9 * peak, "box centre is not hot"
    assert heat[y1 + 40, cx] > 0.5 * peak and heat[y2 - 40, cx] > 0.5 * peak, "head/feet ends not covered"
    assert (inside > 0.3 * peak).mean() > 0.95, "body interior not covered"
    assert heat[cy, x1 - 60] < 0.05 * peak and heat[cy, x2 + 60] < 0.05 * peak, "heat leaks far outside the box"
    ys, xs = np.where(heat > 0.5 * peak)
    cov = ((xs.max() - xs.min()) / (x2 - x1), (ys.max() - ys.min()) / (y2 - y1))
    assert 0.7 < cov[0] < 1.1 and 0.8 < cov[1] < 1.1, cov
    return f"half-max footprint covers {cov[0]:.0%} of box width, {cov[1]:.0%} of height"


@check("engine", "heat-map coordinate alignment")
def _():
    errs = []
    for box in [(100, 80, 180, 300, 1), (700, 200, 820, 520, 2), (0, 0, 60, 200, 3)]:
        e = _engine()
        e.model.boxes = [box]
        e._process(BLANK.copy(), 0, 0, FPS)
        heat = cv2.resize(e.heat, (960, 540), interpolation=cv2.INTER_LINEAR)
        ys, xs = np.nonzero(heat > 0.5 * heat.max())
        w = heat[ys, xs]
        errs.append(np.hypot(np.average(xs, weights=w) - (box[0] + box[2]) / 2,
                             np.average(ys, weights=w) - (box[1] + box[3]) / 2))
    METRICS["heat_align_err_px"] = round(float(max(errs[:2])), 1)
    assert max(errs[:2]) < 3.0, errs   # third box is clipped by the frame edge
    return f"centroid error {max(errs[:2]):.1f}px (full-res)"


@check("engine", "EMA smoothing follows 1 - exp(-dt/tau)")
def _():
    e = _engine()
    e.model.boxes = [BOX]
    e._process(BLANK.copy(), 0, 0, FPS)
    e.model.boxes = [(BOX[0] + 100, BOX[1], BOX[2] + 100, BOX[3], 1)]   # jump 100 px right
    e._process(BLANK.copy(), 0, 5, FPS)                                 # 5 frames later (0.2 s)
    k = 1 - np.exp(-(5 / FPS) / main.SMOOTH_TAU)
    got = e.feet[(0, 1)][0] - BOX[0]
    assert abs(got - 100 * k) < 1e-3, (got, 100 * k)
    e2 = _engine()
    e2.model.boxes = [BOX]
    e2._process(BLANK.copy(), 0, 0, FPS)
    for i in range(1, 6):
        e2.model.boxes = [(BOX[0] + 100, BOX[1], BOX[2] + 100, BOX[3], 1)]
        e2._process(BLANK.copy(), 0, i, FPS)
    assert abs(e2.feet[(0, 1)][0] - e.feet[(0, 1)][0]) < 1e-3, "EMA depends on processing rate"
    e.model.boxes = [(BOX[0] + 300, BOX[1], BOX[2] + 300, BOX[3], 1)]
    e._process(BLANK.copy(), 0, 5 + int(FPS) + 1, FPS)                  # >1 s gap -> snap, no smoothing
    assert e.feet[(0, 1)][0] == BOX[0] + 300
    return f"0.2 s step moves {got:.1f}/100 px (k={k:.3f}); rate-independent; snaps after 1 s gap"


@check("engine", "time-based decay halves heat per half-life")
def _():
    e = _engine()
    e.model.boxes = [BOX]
    e._process(BLANK.copy(), 0, 0, FPS)
    e.model.boxes = []
    e._process(BLANK.copy(), 0, 1, FPS)
    h0 = float(e.heat.sum())
    n = int(main.HEAT_HALF_LIFE * FPS)
    e._process(BLANK.copy(), 0, 1 + n, FPS)          # one big skip
    ratio_skip = float(e.heat.sum()) / h0
    e2 = _engine()
    e2.model.boxes = [BOX]
    e2._process(BLANK.copy(), 0, 0, FPS)
    e2.model.boxes = []
    for i in range(1, 2 + n):                       # every frame
        e2._process(BLANK.copy(), 0, i, FPS)
    ratio_each = float(e2.heat.sum()) / h0
    assert abs(ratio_skip - 0.5) < 1e-3 and abs(ratio_each - 0.5) < 1e-3, (ratio_skip, ratio_each)
    return f"after {main.HEAT_HALF_LIFE:.0f}s: {ratio_skip:.4f} (skipped frames) vs {ratio_each:.4f} (every frame)"


@check("engine", "heatmap toggle changes overlay only")
def _():
    x1, y1, x2, y2, _ = BOX
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    e = _engine()
    e.model.boxes = [BOX]
    e._process(BLANK.copy(), 0, 0, FPS)
    on = _out(e)
    assert e.toggle_heatmap() is False
    e._process(BLANK.copy(), 0, 1, FPS)
    off = _out(e)
    heat_off = float(e.heat.sum())
    assert np.abs(on[cy, cx].astype(int) - 90).max() > 30, "overlay not drawn when ON"
    assert np.abs(off[cy, cx].astype(int) - 90).max() < 8, "overlay drawn when OFF"
    assert heat_off > 0 and e.get_stats()["current_visitors"] == 1, "analytics stopped while OFF"
    assert e.toggle_heatmap() is True and e.get_stats()["heatmap"] is True


@check("engine", "privacy pixelates head region")
def _():
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    x1, y1, x2, y2, _ = BOX
    e = _engine(privacy=True)
    e.show_heatmap = False
    e.model.boxes = [BOX]
    e._process(noisy.copy(), 0, 0, FPS)
    out = _out(e)
    head = out[y1 + 10:y1 + 80, x1 + 10:x2 - 10].astype(float)
    orig = noisy[y1 + 10:y1 + 80, x1 + 10:x2 - 10].astype(float)
    assert np.abs(head - orig).mean() > 40, "head not obscured"
    assert e.toggle_privacy() is False


@check("engine", "low-confidence people are blurred but not counted")
def _():
    rng = np.random.default_rng(1)
    noisy = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    x1, y1, x2, y2, _ = BOX
    e = _engine(privacy=True)
    e.show_heatmap = False
    e.model.boxes = [(x1, y1, x2, y2, 7, 0.30)]
    e._process(noisy.copy(), 0, 0, FPS)
    head = _out(e)[y1 + 10:y1 + 80, x1 + 10:x2 - 10].astype(float)
    assert np.abs(head - noisy[y1 + 10:y1 + 80, x1 + 10:x2 - 10]).mean() > 40, "low-conf head not blurred"
    assert e.get_stats()["current_visitors"] == 0 and float(e.heat.sum()) == 0 and not e.tracks
    assert e.tracker.fed == [0], f"tracker was fed {e.tracker.fed} low-conf detections"


@check("engine", "visitors counted once per session, tracks pruned")
def _():
    e = _engine()
    e.model.boxes = [BOX]
    for i in range(40):
        e._process(BLANK.copy(), 0, i, FPS)
    assert e.get_stats()["total_visitors"] == 1
    for i in range(40):                              # clip looped: same person, new loop key
        e._process(BLANK.copy(), 1, i, FPS)
    s = e.get_stats()
    assert s["total_visitors"] == 1, s["total_visitors"]
    assert list(e.tracks) == [(1, 1)], list(e.tracks)
    assert abs(s["avg_dwell_seconds"] - 39 / FPS) < 0.05, s["avg_dwell_seconds"]
    e.model.boxes = []
    e._process(BLANK.copy(), 1, 40 + int((main.TRACK_TTL + 1) * FPS), FPS)
    assert not e.tracks and e.get_stats()["total_visitors"] == 1
    return f"total stays 1 across loops; avg dwell {s['avg_dwell_seconds']}s kept after pruning"


@check("engine", "processing errors keep the loop alive")
def _():
    e = _engine()
    calls = {"n": 0}

    class Boom(FakeModel):
        def predict(self, frame, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            if calls["n"] >= 4:
                e._stop.set()
            return super().predict(frame, **kw)

    orig = main.HeatmapEngine._run
    e.model = Boom()
    real_yolo = main.YOLO
    main.YOLO = lambda *_: e.model
    seen = []
    real_set = main.HeatmapEngine._process

    def spy(self, *a):
        seen.append(self.get_stats()["status"])
        return real_set(self, *a)

    main.HeatmapEngine._process = spy
    try:
        orig(e)
    finally:
        main.YOLO = real_yolo
        main.HeatmapEngine._process = real_set
    assert calls["n"] >= 4 and "error" in seen, (calls, seen)
    assert e.get_stats()["status"] == "running", "did not recover after the error"
    return f"{calls['n']} frames, statuses seen: {seen}"


@check("engine", "visitor needs a track lasting >= 1.0 s")
def _():
    e = _engine()
    e.model.boxes = [BOX]
    n = int(main.MIN_TRACK_SECONDS * FPS)
    for i in range(n):                               # 0 .. 0.96 s
        e._process(BLANK.copy(), 0, i, FPS)
    assert e.get_stats()["total_visitors"] == 0
    e._process(BLANK.copy(), 0, n, FPS)              # 1.00 s
    assert e.get_stats()["total_visitors"] == 1


@check("engine", "ID switch nearby is re-linked, far new track counts")
def _():
    e = _engine()
    x1, y1, x2, y2, _ = BOX
    e.model.boxes = [(x1, y1, x2, y2, 1)]
    for i in range(50):                              # id 1 for 2 s
        e._process(BLANK.copy(), 0, i, FPS)
    e.model.boxes = []
    for i in range(50, 75):                          # lost for 1 s
        e._process(BLANK.copy(), 0, i, FPS)
    e.model.boxes = [(x1 + 20, y1, x2 + 20, y2, 2)]  # tracker hands out a new id 20 px away
    for i in range(75, 125):
        e._process(BLANK.copy(), 0, i, FPS)
    assert e.get_stats()["total_visitors"] == 1, e.get_stats()["total_visitors"]
    e.model.boxes = [(x1 + 20, y1, x2 + 20, y2, 2), (820, 100, 900, 400, 3)]   # someone new, far away
    for i in range(125, 175):
        e._process(BLANK.copy(), 0, i, FPS)
    s = e.get_stats()
    assert s["total_visitors"] == 2, s["total_visitors"]
    return f"visitors={s['total_visitors']}, avg dwell {s['avg_dwell_seconds']}s"


@check("engine", "reset clears state; video errors are reported")
def _():
    e = _engine()
    e.model.boxes = [BOX]
    for i in range(10):
        e._process(BLANK.copy(), 0, i, FPS)
    e.reset()
    assert e._reset_requested
    e._publish_message(["Video not found."])
    s = e.get_stats()
    assert s["status"] == "error" and "Video not found" in s["message"] and e.get_frame()[0]


# --------------------------------------------------------------------------
print("\n" + "=" * 60)
passed = sum(r[2] for r in RESULTS)
print(f"{passed}/{len(RESULTS)} checks passed")
for k, v in METRICS.items():
    print(f"  {k}: {v}")
sys.exit(0 if passed == len(RESULTS) else 1)
