import streamlit as st
import numpy as np
import math, datetime, io
from shared_session import dept_header, record_session

try:
    import cv2
    ARUCO_OK = hasattr(cv2, "aruco")
except ImportError:
    cv2 = None
    ARUCO_OK = False


# ── Organ zone maps (Couinaud + Kidney + Spleen + Bladder) ────────────────────
ORGAN_ZONES = {
    "Liver (Couinaud)": {
        "icon": "🫀",
        "zones": [
            ("Segment I — Caudate",  "Deep, aim toward IVC, steep angle",      (0.35, 0.35, 0.65, 0.55)),
            ("Segment II",           "Left lobe upper — subcostal flat",        (0.05, 0.05, 0.30, 0.40)),
            ("Segment III",          "Left lobe lower — near aorta",            (0.05, 0.40, 0.30, 0.70)),
            ("Segment IVa",          "Central left upper — between lobes",      (0.30, 0.05, 0.55, 0.38)),
            ("Segment IVb",          "Central left lower — near gallbladder",   (0.30, 0.38, 0.55, 0.65)),
            ("Segment V",            "Right central lower — 45° subcostal",     (0.55, 0.40, 0.80, 0.70)),
            ("Segment VI",           "Right lateral lower — flank approach",    (0.80, 0.50, 1.00, 0.90)),
            ("Segment VII",          "Right posterior upper — intercostal",     (0.80, 0.10, 1.00, 0.50)),
            ("Segment VIII",         "Right upper — near hepatic veins",        (0.55, 0.05, 0.80, 0.40)),
            ("Gallbladder Fossa",    "Between Seg IV and V — fossa view",       (0.45, 0.60, 0.65, 0.85)),
            ("Bile Duct Zone",       "Portal hepatis — look for CBD < 6mm",     (0.35, 0.55, 0.60, 0.80)),
        ],
    },
    "Kidney (Right)": {
        "icon": "🫘",
        "zones": [
            ("Upper Pole",    "Steep angle — may need intercostal",  (0.20, 0.02, 0.80, 0.20)),
            ("Cortex",        "Outer rim — echogenic in adults",     (0.05, 0.20, 0.95, 0.40)),
            ("Outer Medulla", "Hypoechoic pyramids",                 (0.10, 0.40, 0.90, 0.55)),
            ("Inner Medulla", "Renal columns between pyramids",      (0.15, 0.55, 0.85, 0.65)),
            ("Renal Pelvis",  "Central bright echo complex",         (0.25, 0.65, 0.75, 0.75)),
            ("Renal Sinus",   "Fat + collecting system",             (0.20, 0.75, 0.80, 0.85)),
            ("Hilum",         "Vessels enter/exit — use Doppler",    (0.30, 0.85, 0.70, 0.95)),
            ("Lower Pole",    "Most accessible pole",                (0.20, 0.90, 0.80, 1.00)),
        ],
    },
    "Spleen": {
        "icon": "🫁",
        "zones": [
            ("Superior Pole",    "Diaphragmatic surface",          (0.10, 0.02, 0.90, 0.25)),
            ("Anterior Surface", "Face probe toward abdomen",      (0.02, 0.25, 0.40, 0.75)),
            ("Parenchyma",       "Homogeneous — measure long axis",(0.40, 0.25, 0.90, 0.65)),
            ("Splenic Hilum",    "Vessels — use Doppler",          (0.30, 0.65, 0.70, 0.85)),
            ("Inferior Pole",    "Tip of spleen",                  (0.20, 0.85, 0.80, 1.00)),
        ],
    },
    "Bladder": {
        "icon": "💧",
        "zones": [
            ("Anterior Wall",   "Closest to probe — near-field",  (0.15, 0.02, 0.85, 0.20)),
            ("Posterior Wall",  "Far wall — assess thickness",    (0.15, 0.80, 0.85, 0.98)),
            ("Left Lateral",    "Ureteric jet visible on Doppler",(0.02, 0.20, 0.25, 0.80)),
            ("Right Lateral",   "Ureteric jet visible on Doppler",(0.75, 0.20, 0.98, 0.80)),
            ("Trigone",         "Base — ureteric orifices here",  (0.30, 0.70, 0.70, 0.90)),
            ("Bladder Neck",    "Assess for funneling",           (0.35, 0.85, 0.65, 1.00)),
            ("Dome",            "Superior — check for masses",    (0.25, 0.02, 0.75, 0.20)),
        ],
    },
}


def _detect_aruco(img_bytes):
    """Return (frame_rgb, markers_dict, warning_msg).

    Fix: always returns the raw frame so the user sees what their camera
    captured, with a clear warning overlay if marker detection can't run.
    """
    if cv2 is None:
        return None, {}, "opencv-contrib-python not installed"
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, {}, "Could not decode image"
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not ARUCO_OK:
        return frame_rgb, {}, "ArUco module unavailable — showing raw image only"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)
    markers = {}
    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            cx = int(corners[i][0][:, 0].mean())
            cy = int(corners[i][0][:, 1].mean())
            markers[int(mid)] = (cx, cy)
            cv2.circle(frame_rgb, (cx, cy), 12, (0, 255, 0), -1)
            cv2.putText(frame_rgb, f"ID{mid}", (cx - 20, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame_rgb, markers, ""


def _norm_pos(px, py, markers):
    """Map probe tip pixel to normalised 0-1 using corner markers if available."""
    if cv2 is None or not ARUCO_OK:
        return None, None
    corners = {k: markers[k] for k in [10, 11, 12, 13] if k in markers}
    if len(corners) == 4:
        src = np.array([corners[10], corners[11], corners[12], corners[13]], dtype=np.float32)
        dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        if H is not None:
            pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
            out = cv2.perspectiveTransform(pt, H)
            return float(out[0][0][0]), float(out[0][0][1])
    return None, None


def _zone_hit(nx, ny, zones):
    for name, tip, (x0, y0, x1, y1) in zones:
        if x0 <= nx <= x1 and y0 <= ny <= y1:
            return name, tip
    return None, None


def _camera_warning():
    st.warning(
        "⚠️ **Print markers on PAPER** — do not show them on a phone/tablet screen. "
        "Screens emit light that causes glare and breaks ArUco detection. "
        "Print at 100% size (~5 cm square) on plain matte paper."
    )


def _score_badge(score):
    if score >= 90: return "🏆 Excellent"
    if score >= 75: return "✅ Good"
    if score >= 55: return "⚠️ Fair"
    return "❌ Needs Practice"


# ── ULTRASOUND ────────────────────────────────────────────────────────────────
def _ultrasound(student_id):
    dept_header("Ultrasound Training Lab", "🔊", student_id)
    _camera_warning()

    organ = st.selectbox("Select organ / zone map", list(ORGAN_ZONES.keys()))
    info = ORGAN_ZONES[organ]

    st.markdown(f"### {info['icon']} {organ} — Zone Map")
    zone_names = [z[0] for z in info["zones"]]
    target = st.selectbox("Target zone to practise", zone_names)

    col1, col2 = st.columns([3, 2])
    with col1:
        img_file = st.camera_input("📸 Capture probe position")
    with col2:
        st.markdown("**Zone guide:**")
        for z in info["zones"]:
            marker = "🎯" if z[0] == target else "•"
            st.markdown(f"{marker} **{z[0]}** — {z[1]}")

    if img_file:
        frame_rgb, markers, warn = _detect_aruco(img_file.getvalue())
        if frame_rgb is not None:
            st.image(frame_rgb,
                     caption="Detected" if not warn else f"⚠️ {warn}",
                     use_column_width=True)
        if warn:
            st.warning(warn)
            return
        if 0 not in markers:
            st.warning("Probe tip (ID 0) not detected. Check marker is printed on paper and well-lit.")
            return

        px, py = markers[0]
        h, w = frame_rgb.shape[:2]
        nx, ny = px / w, py / h

        # Try homography first; fall back to raw normalised
        if ARUCO_OK and 10 in markers:
            mnx, mny = _norm_pos(px, py, markers)
            if mnx is not None:
                nx, ny = mnx, mny

        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))

        hit_name, hit_tip = _zone_hit(nx, ny, info["zones"])
        target_zone = next((z for z in info["zones"] if z[0] == target), None)

        if hit_name:
            if hit_name == target:
                score = 95.0
                st.success(f"✅ Correct zone: **{hit_name}**")
                st.info(f"Clinical tip: {hit_tip}")
            else:
                score = 45.0
                st.warning(f"⚠️ You are in: **{hit_name}** — target is **{target}**")
        else:
            score = 20.0
            st.error("❌ Probe is outside all defined zones — reposition on the phantom")

        # Angle bonus if both tip+tail detected
        if 0 in markers and 1 in markers:
            ax, ay = markers[0]
            bx, by = markers[1]
            angle = math.degrees(math.atan2(by - ay, bx - ax)) % 180
            st.metric("Probe angle", f"{angle:.1f}°")
            score = min(100, score + 5)

        st.metric("Session score", f"{score:.0f}/100", _score_badge(score))
        if st.button("💾 Save Score", key="us_save"):
            record_session(student_id, "ultrasound", score,
                           f"Organ:{organ} Target:{target} Hit:{hit_name}")
            st.success("Score saved to your profile!")


# ── SURGERY ───────────────────────────────────────────────────────────────────
def _surgery(student_id):
    dept_header("Surgical Skills Suite", "🔪", student_id)
    _camera_warning()

    drill = st.selectbox("Select drill",
        ["Straight incision", "Curved incision", "Laparoscopic circle"])
    st.markdown(f"**Drill:** {drill} — hold instrument (ID 2) and make one steady movement.")

    if "surg_history" not in st.session_state:
        st.session_state.surg_history = []

    img_file = st.camera_input("📸 Capture instrument position")
    if img_file:
        frame_rgb, markers, warn = _detect_aruco(img_file.getvalue())
        if frame_rgb is not None:
            st.image(frame_rgb,
                     caption="Detected" if not warn else f"⚠️ {warn}",
                     use_column_width=True)
        if warn:
            st.warning(warn)
            return
        if 2 not in markers:
            st.warning("Instrument tip (ID 2) not detected — check printed marker.")
            return

        pos = markers[2]
        st.session_state.surg_history.append(pos)
        hist = st.session_state.surg_history

        if len(hist) >= 2:
            diffs = [math.hypot(hist[i][0] - hist[i - 1][0],
                                hist[i][1] - hist[i - 1][1])
                     for i in range(1, len(hist))]
            tremor_px = float(np.std(diffs)) if len(diffs) > 1 else 0.0
            # Fix: normalize tremor against frame width so the score is
            # resolution-independent. On a 4K camera 10 px is tiny; on a
            # 640 px webcam it's huge.
            w = frame_rgb.shape[1]
            tremor_frac = tremor_px / max(w, 1)
            tremor_score = max(0, 100 - tremor_frac * 1000)

            c1, c2, c3 = st.columns(3)
            c1.metric("Captures",     len(hist))
            c2.metric("Tremor (RMS)", f"{tremor_px:.1f}px ({tremor_frac*100:.2f}% of frame)")
            c3.metric("Score",        f"{tremor_score:.0f}/100", _score_badge(tremor_score))

            if st.button("💾 Save Score", key="surg_save"):
                record_session(student_id, "surgery", tremor_score,
                               f"Drill:{drill} tremor:{tremor_px:.1f}px")
                st.success("Score saved!")
                st.session_state.surg_history = []
        else:
            st.info("Keep capturing frames — need at least 2 positions to score tremor.")

    if st.session_state.get("surg_history") and st.button("🔄 Reset drill", key="surg_reset"):
        st.session_state.surg_history = []


# ── FLS PEG TRANSFER ─────────────────────────────────────────────────────────
def _fls(student_id):
    dept_header("FLS Peg Transfer Lab", "🧲", student_id)
    _camera_warning()
    st.markdown("Hold **ID 3** (left) and **ID 4** (right). Bring them together to register a handoff.")

    if "fls_transfers" not in st.session_state:
        st.session_state.fls_transfers = 0
    if "fls_start" not in st.session_state:
        st.session_state.fls_start = datetime.datetime.now()

    img_file = st.camera_input("📸 Capture both instruments")
    if img_file:
        frame_rgb, markers, warn = _detect_aruco(img_file.getvalue())
        if frame_rgb is not None:
            st.image(frame_rgb,
                     caption="Detected" if not warn else f"⚠️ {warn}",
                     use_column_width=True)
        if warn:
            st.warning(warn)
            return

        has_l = 3 in markers
        has_r = 4 in markers
        if not has_l and not has_r:
            st.warning("Neither instrument detected — hold both ID 3 and ID 4 in frame.")
            return

        c1, c2 = st.columns(2)
        c1.metric("Left (ID 3)",  "✅ Detected" if has_l else "❌ Missing")
        c2.metric("Right (ID 4)", "✅ Detected" if has_r else "❌ Missing")

        if has_l and has_r:
            lx, ly = markers[3]
            rx, ry = markers[4]
            dist = math.hypot(rx - lx, ry - ly)
            w = frame_rgb.shape[1]
            # Fix: was hardcoded `dist < 60`. On a 4K camera that never
            # triggers; on a 640 px camera it always triggers.
            # Use 5% of frame width as adaptive handoff threshold.
            handoff_threshold = 0.05 * w
            st.metric("Instrument distance", f"{dist:.0f} px  (handoff < {handoff_threshold:.0f} px)")
            if dist < handoff_threshold:
                st.session_state.fls_transfers += 1
                st.success(f"🤝 Handoff! Total: {st.session_state.fls_transfers}")
            else:
                st.info(f"Move instruments closer together for handoff (< {handoff_threshold:.0f} px)")

        elapsed = (datetime.datetime.now() - st.session_state.fls_start).total_seconds()
        rate = st.session_state.fls_transfers / (elapsed / 60) if elapsed > 0 else 0
        score = min(100, st.session_state.fls_transfers * 10)

        m1, m2, m3 = st.columns(3)
        m1.metric("Handoffs", st.session_state.fls_transfers)
        m2.metric("Rate/min", f"{rate:.1f}")
        m3.metric("Score",    f"{score:.0f}/100", _score_badge(score))

        if st.button("💾 Save Score", key="fls_save"):
            record_session(student_id, "fls", score,
                           f"Handoffs:{st.session_state.fls_transfers} rate:{rate:.1f}/min")
            st.success("Score saved!")
            st.session_state.fls_transfers = 0
            st.session_state.fls_start = datetime.datetime.now()


# ── ROUTER ────────────────────────────────────────────────────────────────────
def render(dept_tab: str, student_id: str):
    if dept_tab == "ultrasound":
        _ultrasound(student_id)
    elif dept_tab == "surgery":
        _surgery(student_id)
    elif dept_tab == "fls":
        _fls(student_id)
