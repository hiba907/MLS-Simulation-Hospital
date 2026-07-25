
import streamlit as st
import json, os, uuid, datetime

st.set_page_config(page_title="AI MLS Simulation Hospital",
                   page_icon="🏥", layout="wide")

from shared_session import (
    make_guest_id, load_student,
    leaderboard_data, DEPT_KEYS
)

_CSS = """
<style>
body, .stApp { background:#1D1D20; color:#fbfbff; }
.dept-card {
    background:#23232a; border:1px solid #333; border-radius:12px;
    padding:1.4rem; text-align:center;
}
.dept-card:hover { border-color:#ffd400; }
.dept-card .icon  { font-size:2.6rem; }
.dept-card .label { font-weight:700; font-size:1.1rem; margin-top:.4rem; }
.dept-card .desc  { color:#909094; font-size:.82rem; margin-top:.3rem; }
.landing-btn button {
    font-size: 1.3rem !important;
    padding: 1rem 2rem !important;
    background: #ffd400 !important;
    color: #1D1D20 !important;
    border: none !important;
}
.session-note {
    color:#909094; font-size:0.85rem; text-align:center; margin-top:1rem;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

DEPTS = [
    ("ultrasound",   "🔊","Ultrasound Lab",      "Probe positioning with organ zone maps"),
    ("surgery",      "🔪","Surgery Suite",        "Tremor, path precision & incision drills"),
    ("fls",          "🧲","FLS Peg Transfer",     "Two-instrument handoff timing"),
    ("ct",           "🩻","CT Reading Room",      "Slice, segment & upload real DICOM"),
    ("endoscopy",    "🔭","Endoscopy Suite",      "Navigation, biopsy & retroflexion drills"),
    ("analytics",    "📊","Analytics Dashboard",  "Score charts, radar & leaderboard"),
    ("instructions", "📖","Instructions",         "Full setup guide & ArUco marker table"),
]


def _landing():
    """No-registration landing page. Single button → enter as guest."""
    st.markdown(
        "<h1 style='text-align:center;color:#ffd400;margin-top:3rem'>🏥 AI MLS Simulation Hospital</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#fbfbff;font-size:1.1rem;margin-top:0.5rem'>"
        "Medical Learning Simulation Platform</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#909094;margin-top:0.5rem'>"
        "Hands-on practice with ultrasound, surgery, FLS, CT, and endoscopy — "
        "using your phone or webcam and objects you already have."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("<div class='landing-btn'>", unsafe_allow_html=True)
    if st.button("🚀 Enter Hospital", type="primary", use_container_width=True):
        st.session_state.student_id   = make_guest_id()
        st.session_state.student_name = "Guest"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='session-note'>No account needed. Your scores save to this browser tab only.</p>",
        unsafe_allow_html=True,
    )


def _lobby(sid, sname):
    st.markdown("<h2 style='color:#ffd400'>🏥 Hospital Lobby</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:#909094'>Session: <b>{sname}</b> ({sid}) — "
        f"scores are saved for this browser tab</p>",
        unsafe_allow_html=True,
    )

    with st.expander("🏆 Leaderboard", expanded=False):
        rows = leaderboard_data()
        if rows:
            for i, r in enumerate(rows[:10], 1):
                bar = "⭐" * min(5, int(r["overall"]/20))
                st.markdown(f"`#{i}` **{r['name']}** — {r['overall']}  {bar}")
        else:
            st.info("No sessions yet — start training to appear here!")

    st.markdown("---")
    cols = st.columns(4)
    for idx, (key, icon, label, desc) in enumerate(DEPTS):
        with cols[idx % 4]:
            st.markdown(
                f'''<div class="dept-card">
                <div class="icon">{icon}</div>
                <div class="label">{label}</div>
                <div class="desc">{desc}</div>
                </div>''', unsafe_allow_html=True)
            st.write("")
            if st.button(f"Enter {icon}", key=f"btn_{key}"):
                st.session_state.dept = key
                st.rerun()

    st.markdown("---")
    if st.button("🔄 Start New Session", key="new_session"):
        for k in ["student_id", "student_name", "dept"]:
            st.session_state.pop(k, None)
        st.rerun()


def _route(dept, sid):
    if st.button("⬅️ Back to Lobby", key="back"):
        st.session_state.pop("dept", None)
        st.rerun()

    # NOTE: enrollment gating removed — guest mode allows all departments.
    # The is_enrolled() function in shared_session.py is kept as a stub for
    # backwards compatibility with any code that still imports it.

    if dept == "ultrasound":
        try:
            from mls_hospital_module import render as r
            r("ultrasound", sid)
        except Exception as _e:
            import traceback
            st.error(f"Ultrasound module error: {_e}")
            st.code(traceback.format_exc())
    elif dept == "surgery":
        try:
            from surgery_module import run as r
            r(sid)
        except Exception as _e:
            import traceback
            st.error(f"Surgery module error: {_e}")
            st.code(traceback.format_exc())
    elif dept == "fls":
        try:
            from mls_hospital_module import render as r
            r("fls", sid)
        except Exception as _e:
            import traceback
            st.error(f"FLS module error: {_e}")
            st.code(traceback.format_exc())
    elif dept == "ct":
        try:
            from ct_hospital_module import render as r
            r(sid)
        except Exception as _e:
            import traceback
            st.error(f"CT module error: {_e}")
            st.code(traceback.format_exc())
    elif dept == "endoscopy":
        try:
            from endoscopy_module import render as r
            r(sid)
        except Exception as _e:
            import traceback
            st.error(f"Endoscopy module error: {_e}")
            st.code(traceback.format_exc())
    elif dept == "analytics":
        try:
            from analytics_dashboard import render as r
            r(sid)
        except Exception as _e:
            import traceback
            st.error(f"Analytics module error: {_e}")
            st.code(traceback.format_exc())
    elif dept == "instructions":
        try:
            from instructions_module import render as r
            r(sid)
        except Exception as _e:
            import traceback
            st.error(f"Instructions module error: {_e}")
            st.code(traceback.format_exc())


def main():
    if "student_id" not in st.session_state:
        _landing()
        return
    sid   = st.session_state.student_id
    sname = st.session_state.student_name
    dept  = st.session_state.get("dept")
    if dept:
        _route(dept, sid)
    else:
        _lobby(sid, sname)

main()
