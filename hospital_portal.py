import streamlit as st
import json, os, uuid, datetime

st.set_page_config(page_title="AI MLS Simulation Hospital",
                   page_icon="🏥", layout="wide")

from shared_session import (
    register_student, login_student, load_student,
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
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

DEPTS = [
    ("ultrasound",   "🔊", "Ultrasound Lab",      "Probe positioning with organ zone maps"),
    ("surgery",      "🔪", "Surgery Suite",        "Tremor, path precision & incision drills"),
    ("fls",          "🧲", "FLS Peg Transfer",     "Two-instrument handoff timing"),
    ("ct",           "🩻", "CT Reading Room",      "Slice, segment & upload real DICOM"),
    ("endoscopy",    "🔭", "Endoscopy Suite",      "Navigation, biopsy & retroflexion drills"),
    ("analytics",    "📊", "Analytics Dashboard",  "Score charts, radar & leaderboard"),
    ("instructions", "📖", "Instructions",         "Full setup guide & ArUco marker table"),
]


def _login_ui():
    st.markdown("<h1 style='text-align:center;color:#ffd400'>🏥 AI MLS Simulation Hospital</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#909094'>Medical Learning Simulation Platform</p>",
                unsafe_allow_html=True)
    st.markdown("---")
    tab_r, tab_s, tab_g = st.tabs(["📝 Register", "🔑 Sign In", "👤 Guest / Demo"])

    with tab_r:
        st.subheader("Create Account")
        name = st.text_input("Your full name", key="reg_name")
        st.selectbox("Training level",
                     ["Medical Student", "Intern", "Resident", "Fellow", "Attending"],
                     key="reg_level")
        if st.button("✅ Register", type="primary", key="do_reg"):
            if name.strip():
                data = register_student(name.strip())
                st.session_state.student_id = data["id"]
                st.session_state.student_name = data["name"]
                st.success(f"Welcome {name}! Your ID: **{data['id']}** — write this down!")
                st.rerun()
            else:
                st.error("Please enter your name.")

    with tab_s:
        st.subheader("Sign In")
        sid = st.text_input("Your Student ID (e.g. s_abc123)", key="login_sid")
        if st.button("🔑 Sign In", type="primary", key="do_login"):
            data = login_student(sid.strip())
            if data:
                st.session_state.student_id = data["id"]
                st.session_state.student_name = data["name"]
                st.success(f"Welcome back, {data['name']}!")
                st.rerun()
            else:
                st.error("Student ID not found. Please register first.")

    with tab_g:
        st.info("Guest mode: explore everything — scores are not saved.")
        if st.button("🚀 Enter as Guest", key="do_guest"):
            st.session_state.student_id = "guest"
            st.session_state.student_name = "Guest"
            st.rerun()


def _lobby(sid, sname):
    st.markdown(f"<h2 style='color:#ffd400'>🏥 Hospital Lobby</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#909094'>Logged in as <b>{sname}</b> ({sid})</p>",
                unsafe_allow_html=True)

    with st.expander("🏆 Leaderboard", expanded=False):
        rows = leaderboard_data()
        if rows:
            for i, r in enumerate(rows[:10], 1):
                bar = "⭐" * min(5, int(r["overall"] / 20))
                st.markdown(f"`#{i}` **{r['name']}** — {r['overall']}  {bar}")
        else:
            st.info("No students yet — register to appear here!")

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
    if st.button("🚪 Logout", key="logout"):
        for k in ["student_id", "student_name", "dept"]:
            st.session_state.pop(k, None)
        st.rerun()


def _route(dept, sid):
    if st.button("⬅️ Back to Lobby", key="back"):
        st.session_state.pop("dept", None)
        st.rerun()
    if dept == "ultrasound":
        from mls_hospital_module import render as r; r("ultrasound", sid)
    elif dept == "surgery":
        from mls_hospital_module import render as r; r("surgery", sid)
    elif dept == "fls":
        from mls_hospital_module import render as r; r("fls", sid)
    elif dept == "ct":
        from ct_hospital_module import render as r; r(sid)
    elif dept == "endoscopy":
        from endoscopy_module import render as r; r(sid)
    elif dept == "analytics":
        from analytics_dashboard import render as r; r(sid)
    elif dept == "instructions":
        from instructions_module import render as r; r(sid)


def main():
    if "student_id" not in st.session_state:
        _login_ui()
        return
    sid = st.session_state.student_id
    sname = st.session_state.student_name
    dept = st.session_state.get("dept")
    if dept:
        _route(dept, sid)
    else:
        _lobby(sid, sname)


main()
