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


def _detect_aruco(img_bytes):
    """Return (frame_rgb, markers_dict, warning_msg).

    Fix: previously returned (None, {}) on missing opencv, leaving the user
    staring at a blank screen. Now always returns the raw frame so the user
    at least sees what their camera captured, with a clear warning if detection
    can't run.
    """
    arr = np.frombuffer(img_bytes, np.uint8)
    if cv2 is None:
        return None, {}, "opencv-contrib-python not installed"
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


ZONES = ["Left", "Centre", "Right"]
ZONE_CENTERS = {"Left": 0.17, "Centre": 0.5, "Right": 0.83}


def _camera_warning():
    st.warning("⚠️ Print markers on PAPER — screens cause glare and break detection.")


def _score_badge(s):
    if s >= 90: return "🏆 Excellent"
    if s >= 75: return "✅ Good"
    if s >= 55: return "⚠️ Fair"
    return "❌ Needs Practice"


def render(student_id: str):
    dept_header("Endoscopy Suite", "🔭", student_id)
    _camera_warning()

    drill = st.selectbox("Select drill", [
        "🗺️ Scope Navigation",
        "🎯 Target Biopsy",
        "🔄 Retroflexion",
        "🎥 CT Flythrough",
    ])

    if drill == "🗺️ Scope Navigation":
        st.markdown("""
        **Setup:** Tape ID 5 to your scope tip. Draw LEFT / CENTRE / RIGHT zones on paper.
        Move scope through each zone in order.
        """)
        target_zone = st.selectbox("Target zone", ZONES)
        img_file = st.camera_input("📸 Capture scope position")
        if img_file:
            frame_rgb, markers, warn = _detect_aruco(img_file.getvalue())
            if frame_rgb is not None:
                st.image(frame_rgb,
                         caption="Detected" if not warn else f"⚠️ {warn}",
                         use_column_width=True)
            if warn:
                st.warning(warn)
                return
            if 5 not in markers:
                st.warning("Scope tip (ID 5) not detected.")
                return
            px, py = markers[5]
            h, w = frame_rgb.shape[:2]
            nx = px / w
            # Continuous scoring based on distance to target zone centre.
            # Fix: previously 90 vs 30 — two pixels off the 0.33 line dropped
            # you 60 points. Now it's smooth.
            target_x = ZONE_CENTERS[target_zone]
            score = max(0.0, 100 - abs(nx - target_x) * 200)
            zone = "Left" if nx < 0.33 else ("Centre" if nx < 0.66 else "Right")
            st.metric("Detected zone", zone)
            st.metric("Score", f"{score:.0f}/100", _score_badge(score))
            if st.button("💾 Save", key="endo_nav_save"):
                record_session(student_id, "endoscopy", score,
                               f"Navigation zone:{zone}")
                st.success("Saved!")

    elif drill == "🎯 Target Biopsy":
        st.markdown("""
        **Setup:** Draw 3 red circles on paper. Hold scope (ID 5) as close to centre of target circle as possible.
        """)
        target_num = st.slider("Target circle number", 1, 3)
        target_positions = {1: (0.25, 0.5), 2: (0.5, 0.5), 3: (0.75, 0.5)}
        tx, ty = target_positions[target_num]

        img_file = st.camera_input("📸 Aim at target")
        if img_file:
            frame_rgb, markers, warn = _detect_aruco(img_file.getvalue())
            if frame_rgb is not None:
                st.image(frame_rgb,
                         caption="Detected" if not warn else f"⚠️ {warn}",
                         use_column_width=True)
            if warn:
                st.warning(warn)
                return
            if 5 not in markers:
                st.warning("Scope tip (ID 5) not detected.")
                return
            px, py = markers[5]
            h, w = frame_rgb.shape[:2]
            nx, ny = px / w, py / h
            dist = math.hypot(nx - tx, ny - ty)
            score = max(0, 100 - dist * 200)
            st.metric("Distance to target", f"{dist * 100:.1f}%")
            st.metric("Score", f"{score:.0f}/100", _score_badge(score))
            if st.button("💾 Save", key="endo_bx_save"):
                record_session(student_id, "endoscopy", score,
                               f"Biopsy target:{target_num} dist:{dist:.3f}")
                st.success("Saved!")

    elif drill == "🔄 Retroflexion":
        st.markdown("""
        **Setup:** Tape ID 5 to scope tip, ID 6 to handle (~10 cm back).
        Rotate scope until angle between tip and handle is in the retroflexion window.
        """)
        if "retro_history" not in st.session_state:
            st.session_state.retro_history = []

        img_file = st.camera_input("📸 Capture scope angle")
        if img_file:
            frame_rgb, markers, warn = _detect_aruco(img_file.getvalue())
            if frame_rgb is not None:
                st.image(frame_rgb,
                         caption="Detected" if not warn else f"⚠️ {warn}",
                         use_column_width=True)
            if warn:
                st.warning(warn)
                return
            if 5 not in markers or 6 not in markers:
                st.warning("Need both ID 5 (tip) and ID 6 (handle) visible.")
                return
            ax, ay = markers[6]
            bx, by = markers[5]
            angle = math.degrees(math.atan2(by - ay, bx - ax)) % 360
            st.metric("Scope angle", f"{angle:.1f}°")
            # Fix: original `angle > 120 or angle < -120` had dead code (angle is
            # already wrapped to [0, 360) by `% 360`, so never negative).
            # Retroflexion = scope tip rotated ~180° from handle → angle near 180°.
            target = 180.0
            shortest_arc = abs(((angle - target + 180) % 360) - 180)
            if shortest_arc < 60:  # within 120°–240° window
                st.success("✅ Retroflexion achieved!")
                score = 95.0
            else:
                needed = shortest_arc
                st.info(f"Rotate {needed:.0f}° more toward 180° to achieve retroflexion.")
                score = max(0, (1 - shortest_arc / 180) * 80)
            st.metric("Score", f"{score:.0f}/100", _score_badge(score))
            if st.button("💾 Save", key="endo_retro_save"):
                record_session(student_id, "endoscopy", score,
                               f"Retroflexion angle:{angle:.1f}")
                st.success("Saved!")

    elif drill == "🎥 CT Flythrough":
        st.markdown("""
        **Virtual colonoscopy flythrough** — load a CT case in the CT Reading Room first,
        then return here to fly through the volume axially.
        """)
        if "ct_vol" not in st.session_state:
            st.warning("No CT case loaded. Go to 🩻 CT Reading Room → Load Case first.")
            return
        vol = st.session_state.ct_vol
        nz = vol.shape[0]
        depth = st.slider("Flythrough depth", 0, nz - 1, nz // 2)
        sl = vol[depth]
        lo, hi = -700.0, 100.0
        clipped = np.clip(sl, lo, hi)
        scaled = ((clipped - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
        from PIL import Image, ImageDraw
        img = Image.fromarray(scaled).convert("RGB")
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        w, h = img.size
        draw.ellipse([10, 10, w - 10, h - 10], fill=255)
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=mask)
        st.image(bg, caption=f"Endoscope view — depth {depth}/{nz}",
                 use_column_width=True)
        if st.button("💾 Save flythrough session", key="endo_fly_save"):
            record_session(student_id, "endoscopy", 70.0, f"Flythrough depth:{depth}")
            st.success("Saved!")
