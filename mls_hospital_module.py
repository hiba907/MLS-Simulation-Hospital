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

# Suture Lab reference gallery + simulator (pure-Python, no Streamlit dep)
from suture_module import SUTURE_TYPES, render_schematic, score_stitch


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


# ── SURGERY (tabbed: Incision, Suture Lab, Tremor) ──────────────────────────
#
# Marker layout reminder:
#   ID 2  = instrument tip (e.g. scalpel or needle holder)
#   ID 20 = wound endpoint A (printed and taped at one end of the cut)
#   ID 21 = wound endpoint B (printed and taped at the other end)
# The line A-B defines the wound. Works for any size — 1 cm or 20 cm.
# (For tiny wounds where 2 markers would crowd the cut, place them slightly
# off the wound on either side; the line A-B still defines the wound axis.)


def _surgery(student_id):
    dept_header("Surgical Skills Suite", "🔪", student_id)
    _camera_warning()

    tab_incise, tab_suture, tab_tremor = st.tabs([
        "🔪 Incision Drill (wound markers)",
        "🪡 Suture Lab (8 patterns)",
        "📏 Tremor Analysis",
    ])
    with tab_incise:
        _incision_drill(student_id)
    with tab_suture:
        _suture_lab(student_id)
    with tab_tremor:
        _tremor_drill(student_id)


# ── Incision drill: track instrument relative to the wound line ─────────────
def _incision_drill(student_id):
    st.markdown("### 🔪 Incision Drill — Make a straight cut along the wound")

    with st.expander("ℹ️ How wound markers work (read this once)", expanded=False):
        st.markdown(r"""
**The wound is a break in the skin** — a *cut*. It can be small (1 cm) or
large (20+ cm). To teach the simulator where the cut is, you place **2
ArUco markers at the 2 ends of the cut**:

| Marker | Where to put it |
|---|---|
| **ID 20** | One end of the cut (wound endpoint A) |
| **ID 21** | The other end of the cut (wound endpoint B) |
| **ID 2**  | The instrument tip (scalpel, needle holder, …) |

The simulator draws a yellow line between markers 20 and 21 — that **is** the
wound. You then move the instrument (ID 2) along the line to make the
incision.

**Why 2 markers, not 1?** A cut is a line, and a line needs 2 points to
define it. With 2 markers, the simulator knows the wound's *position*,
*length*, and *orientation* — regardless of whether the cut is 1 cm or 20 cm.

**For a very small cut** (1-2 cm) the 2 markers will sit right next to each
other at the wound ends — that's fine, the math still works. If the cut is
so small that 2 printed markers would overlap, place them slightly off the
wound on either side, along the wound's long axis.

**No wound yet?** Draw a line on a piece of tape with a marker, then put
the 2 ArUco markers at the ends of the line. You can also use the corner
markers (ID 10/11) of the phantom as endpoints for a long wound.

**For a curved cut** (which this drill doesn't simulate yet), you'd need
3+ markers. That's coming in a future version.
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        drill = st.selectbox("Drill",
            ["Straight incision", "Long incision (>10 cm)"],
            key="incise_drill")
    with col2:
        st.markdown("**Wound markers (printed):**")
        st.markdown("- ID 20 — endpoint A")
        st.markdown("- ID 21 — endpoint B")
    with col3:
        st.markdown("**Instrument:**")
        st.markdown("- ID 2 — scalpel tip")

    if "incise_frames" not in st.session_state:
        st.session_state.incise_frames = []
    if "incise_wound" not in st.session_state:
        st.session_state.incise_wound = None  # (p1, p2, length_px) once defined

    img_file = st.camera_input("📸 Capture incision (one frame at a time)")

    if img_file:
        frame_rgb, markers, warn = _detect_aruco(img_file.getvalue())
        if warn:
            st.warning(warn)
            return

        if 20 not in markers or 21 not in markers:
            st.warning("⚠️ Wound markers (IDs **20** and **21**) not both detected. "
                       "Print & tape them at the 2 ends of your cut.")
            # Still show the frame so the user sees what was captured
            if frame_rgb is not None:
                st.image(frame_rgb, caption="Detected markers", use_column_width=True)
            return
        if 2 not in markers:
            st.warning("⚠️ Instrument (ID **2**) not detected. Hold the printed marker "
                       "on your scalpel / needle holder.")
            if frame_rgb is not None:
                # Draw the wound line even without instrument
                disp = frame_rgb.copy()
                cv2.line(disp, markers[20], markers[21], (0, 255, 255), 2)
                cv2.circle(disp, markers[20], 8, (255, 0, 255), -1)
                cv2.circle(disp, markers[21], 8, (255, 0, 255), -1)
                st.image(disp, caption="Wound (yellow) — instrument not detected", use_column_width=True)
            return

        # We have wound markers + instrument. Annotate the frame.
        p1, p2 = markers[20], markers[21]
        instr = markers[2]
        h, w = frame_rgb.shape[:2]
        # Convert to normalized [0..1] coordinates for scoring
        n_instr = (instr[0] / w, instr[1] / h)
        n_p1 = (p1[0] / w, p1[1] / h)
        n_p2 = (p2[0] / w, p2[1] / h)
        wound_len_norm = math.hypot(n_p2[0] - n_p1[0], n_p2[1] - n_p1[1])

        # Save wound definition once
        if st.session_state.incise_wound is None:
            st.session_state.incise_wound = (n_p1, n_p2, wound_len_norm)
            st.success(f"✅ Wound defined — length {wound_len_norm:.2f} of frame "
                       f"({(wound_len_norm * (w / 100) if w else 0):.0f} cm if frame is {w/100:.0f} cm wide)")

        # Distance from instrument to the wound LINE (not just the endpoints)
        # Line through p1 → p2 in normalized coords
        dx, dy = n_p2[0] - n_p1[0], n_p2[1] - n_p1[1]
        seg_len = math.hypot(dx, dy) or 1e-6
        # Project instrument onto the segment
        t = max(0.0, min(1.0, ((n_instr[0] - n_p1[0]) * dx + (n_instr[1] - n_p1[1]) * dy) / (seg_len ** 2)))
        proj_x = n_p1[0] + t * dx
        proj_y = n_p1[1] + t * dy
        dist_to_line = math.hypot(n_instr[0] - proj_x, n_instr[1] - proj_y)
        # Position along the wound (0 = at A, 1 = at B)
        along = t
        # Distance to the nearest endpoint
        d_a = math.hypot(n_instr[0] - n_p1[0], n_instr[1] - n_p1[1])
        d_b = math.hypot(n_instr[0] - n_p2[0], n_instr[1] - n_p2[1])

        # Append to history
        st.session_state.incise_frames.append({
            "instr": n_instr,
            "dist_to_line": dist_to_line,
            "along": along,
            "d_a": d_a,
            "d_b": d_b,
        })

        # Annotate and show
        disp = frame_rgb.copy()
        cv2.line(disp, p1, p2, (0, 255, 255), 2)
        cv2.circle(disp, p1, 8, (255, 0, 255), -1)
        cv2.circle(disp, p2, 8, (255, 0, 255), -1)
        cv2.circle(disp, instr, 10, (0, 255, 0), -1)
        cv2.putText(disp, f"line dist: {dist_to_line*100:.1f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp, f"along: {along*100:.0f}%",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        st.image(disp, caption=f"Frame {len(st.session_state.incise_frames)} — "
                               f"instrument is {dist_to_line*100:.1f}% off the wound line, "
                               f"{along*100:.0f}% along (A→B)", use_column_width=True)

    # Show the trajectory + scores once we have at least 2 frames
    if len(st.session_state.incise_frames) >= 2:
        frames = st.session_state.incise_frames
        # Tremor — std-dev of frame-to-frame movement
        diffs = []
        for i in range(1, len(frames)):
            a = frames[i - 1]["instr"]
            b = frames[i]["instr"]
            diffs.append(math.hypot(a[0] - b[0], a[1] - b[1]))
        tremor_frac = float(np.std(diffs)) if len(diffs) > 1 else 0.0
        tremor_score = max(0, 100 - tremor_frac * 800)

        # Path-following — average distance to the line
        avg_off = float(np.mean([f["dist_to_line"] for f in frames]))
        path_score = max(0, 100 - avg_off * 600)

        # Endpoint coverage — first frame should be near A, last near B
        if len(frames) >= 2:
            start_d = frames[0]["d_a"]   # first frame's distance to A
            end_d = frames[-1]["d_b"]     # last frame's distance to B
            end_score = max(0, 100 - (start_d + end_d) * 200)

        total = tremor_score * 0.4 + path_score * 0.4 + end_score * 0.2

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Captures",     len(frames))
        c2.metric("Tremor",       f"{tremor_frac*100:.2f}% / frame")
        c3.metric("Off-line",     f"{avg_off*100:.1f}% of wound length")
        c4.metric("Score",        f"{total:.0f}/100", _score_badge(total))

        with st.expander("Score breakdown", expanded=False):
            st.markdown(f"""
- **Tremor** (40%): std-dev of frame-to-frame movement → `{tremor_score:.0f}/100`
  - Lower = more steady hand
- **Path-following** (40%): avg distance from the wound line → `{path_score:.0f}/100`
  - Lower = straighter incision
- **Endpoints** (20%): did you start near A and end near B → `{end_score:.0f}/100`
  - Lower = missed wound ends
""")

        if st.button("💾 Save Score", key="incise_save"):
            record_session(student_id, "surgery", total,
                           f"Incision frames:{len(frames)} "
                           f"tremor:{tremor_frac*100:.2f}% off:{avg_off*100:.1f}%")
            st.success("Score saved!")
            st.session_state.incise_frames = []
            st.session_state.incise_wound = None

    if st.session_state.incise_frames and st.button("🔄 Reset drill", key="incise_reset"):
        st.session_state.incise_frames = []
        st.session_state.incise_wound = None


# ── Suture lab: pick a pattern, study the schematic, practice the bites ────
def _suture_lab(student_id):
    st.markdown("### 🪡 Suture Lab — pick a pattern, study it, then practise")

    with st.expander("ℹ️ How this works", expanded=False):
        st.markdown(r"""
**The simulator gives you:**

1. **A reference gallery** of 8 common suture patterns. Each is drawn as a
   schematic with the wound (red line), the suture path (colored), and the
   order of bites (numbered). Study the pattern before you start.
2. **A virtual wound** drawn in your camera frame. The simulator shows you
   the *next* bite target as a highlighted dot — your job is to hover the
   instrument (ID 2) over that dot and click the camera.
3. **A score for each bite** based on how close you got to the target. The
   session score is the average across all bites.

**Setup:**
- Print the **ID 2** marker and tape it to your instrument (needle holder,
  forceps, anything pen-like).
- The camera view IS the practice field — you don't need a suturing pad
  for this drill. The wound is shown as a virtual line and you hit the
  numbered dots in order.
- For "Purse-string", the wound is a small circle in the middle.
""")

    # Pick a pattern
    col_pick, col_info = st.columns([1, 1])
    with col_pick:
        pattern = st.selectbox("Suture pattern", list(SUTURE_TYPES.keys()), key="suture_pick")
    with col_info:
        info = SUTURE_TYPES[pattern]
        st.markdown(f"**{info['name']}**")
        st.markdown(f"_{info['short']}_")
        st.markdown(f"**Indications:** {info['indications']}")
        n_bites = info["n_bites_per_stitch"]
        n_st = info["n_stitches"]
        total_bites = n_bites * n_st
        st.markdown(f"**Stitches:** {n_st} × **{n_bites} bite{'s' if n_bites > 1 else ''}** "
                    f"each = **{total_bites} targets**")

    # Render the schematic
    fig = render_schematic(pattern)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.markdown("#### 🎯 Practice the pattern")
    st.caption("Each numbered dot is a bite. Hover the instrument (ID 2) over the dot and capture.")

    # Simulator state
    if "suture_bites" not in st.session_state:
        st.session_state.suture_bites = []
    if "suture_pattern" not in st.session_state or st.session_state.suture_pattern != pattern:
        # New pattern selected — reset
        st.session_state.suture_bites = []
        st.session_state.suture_pattern = pattern

    # Get target list
    targets = SUTURE_TYPES[pattern]["bites"](0.80, 1.0)
    n_bites_per = SUTURE_TYPES[pattern]["n_bites_per_stitch"]
    n_total = len(targets)
    n_done = len(st.session_state.suture_bites)
    st.progress(n_done / n_total, text=f"Bites: {n_done} / {n_total}")

    if n_done >= n_total:
        scores = [b["score"] for b in st.session_state.suture_bites]
        avg = sum(scores) / len(scores)
        st.success(f"🎉 All {n_total} bites complete — average **{avg:.0f}/100** "
                   f"({_score_badge(avg)})")
        c1, c2, c3 = st.columns(3)
        c1.metric("Average score", f"{avg:.0f}/100", _score_badge(avg))
        c2.metric("Best bite", f"{max(scores):.0f}")
        c3.metric("Worst bite", f"{min(scores):.0f}")
        if st.button("💾 Save Score", key="suture_save"):
            record_session(student_id, "surgery", avg,
                           f"Suture:{pattern} avg:{avg:.0f} bites:{n_total}")
            st.success("Score saved!")
            st.session_state.suture_bites = []
        if st.button("🔄 Try again", key="suture_again"):
            st.session_state.suture_bites = []
            st.rerun()
    else:
        # Show next target
        next_target = targets[n_done]
        stitch_idx = n_done // n_bites_per + 1
        bite_idx = n_done % n_bites_per + 1
        st.markdown(f"**Next:** Stitch {stitch_idx}, bite {bite_idx} — "
                    f"target at x={next_target[0]:.2f}, y={next_target[1]:.2f}")

        # Show all targets on a small map with the next one highlighted
        fig2, ax2 = plt.subplots(figsize=(8, 2.5))
        # Draw a virtual wound
        if pattern == "Purse-string":
            ax2.add_patch(plt.Circle((0.5, 0.5), 0.10, fill=False,
                                     edgecolor="#8b0000", linewidth=3))
        else:
            ax2.plot([0.10, 0.90], [0.5, 0.5], color="#8b0000", linewidth=3)
        # All targets in light gray
        for i, (x, y) in enumerate(targets):
            color = "lightgray"
            if i < n_done:
                color = "lightgreen"  # done
            elif i == n_done:
                color = SUTURE_TYPES[pattern]["color"]  # next
            ax2.plot(x, y, "o", color=color, markersize=10 if i == n_done else 6,
                     markeredgecolor="black" if i == n_done else "white",
                     markeredgewidth=1.5)
        # Show instrument so far
        for b in st.session_state.suture_bites:
            ax2.plot(b["actual"][0], b["actual"][1], "x", color="black", markersize=8)
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_aspect("equal"); ax2.axis("off")
        ax2.set_title("Targets (green=done, colored=next, gray=upcoming)", fontsize=10)
        st.pyplot(fig2)
        plt.close(fig2)

        img_file = st.camera_input(f"📸 Bite {n_done + 1} of {n_total}")
        if img_file:
            frame_rgb, markers, warn = _detect_aruco(img_file.getvalue())
            if warn:
                st.warning(warn)
            elif 2 in markers and frame_rgb is not None:
                h, w = frame_rgb.shape[:2]
                actual = (markers[2][0] / w, markers[2][1] / h)
                score, dist, status = score_stitch(actual, next_target)
                st.session_state.suture_bites.append({
                    "target": next_target,
                    "actual": actual,
                    "score": score,
                    "dist": dist,
                    "status": status,
                })
                # Show annotated frame
                disp = frame_rgb.copy()
                cv2.circle(disp, markers[2], 10, (0, 255, 0), -1)
                # Draw the target in wound coordinates → pixel coordinates
                tx_px = int(next_target[0] * w)
                ty_px = int(next_target[1] * h)
                cv2.circle(disp, (tx_px, ty_px), 12, (0, 0, 255), 2)
                cv2.line(disp, markers[2], (tx_px, ty_px), (255, 200, 0), 1)
                st.image(disp, caption=f"Bite {n_done + 1}: score **{score:.0f}** "
                                       f"({status}, dist {dist*100:.1f}% of frame)",
                         use_column_width=True)
                # Auto-advance
                st.rerun()
            else:
                st.warning("Instrument (ID 2) not detected. Check the printed marker.")


# ── Tremor analysis (renamed legacy drill) ──────────────────────────────────
def _tremor_drill(student_id):
    st.markdown("### 📏 Tremor Analysis — hold the instrument as still as you can")
    st.caption("Legacy drill: hold the instrument (ID 2) and capture multiple frames. "
               "Lower tremor (std-dev of frame-to-frame movement) = higher score.")

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
            w = frame_rgb.shape[1]
            tremor_frac = tremor_px / max(w, 1)
            tremor_score = max(0, 100 - tremor_frac * 1000)

            c1, c2, c3 = st.columns(3)
            c1.metric("Captures",     len(hist))
            c2.metric("Tremor (RMS)", f"{tremor_px:.1f}px ({tremor_frac*100:.2f}% of frame)")
            c3.metric("Score",        f"{tremor_score:.0f}/100", _score_badge(tremor_score))

            if st.button("💾 Save Score", key="surg_save"):
                record_session(student_id, "surgery", tremor_score,
                               f"Tremor drill tremor:{tremor_px:.1f}px")
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
