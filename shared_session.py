import os, json, uuid, datetime, glob, math
import streamlit as st

# Anchor data directory to this script's location, not cwd.
# (Original used cwd-relative path, which broke if launched from a different folder.)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENTS_DIR = os.path.join(_SCRIPT_DIR, "hospital_data")
os.makedirs(STUDENTS_DIR, exist_ok=True)

DEPT_KEYS = ["ultrasound", "surgery", "fls", "ct", "endoscopy", "reasoning", "pathology", "foley", "intubation", "endoscopy_scope"]


def make_guest_id() -> str:
    """Per-session guest ID. Each browser session gets a fresh ID, so scores
    don't mix between users but still save per session."""
    return "g_" + uuid.uuid4().hex[:8]


def _top_surgeon(s):
    """Top Surgeon = mastery (>=80) in every dept the student has actually attempted.

    Fix: previously required `best >= 80` in ALL 5 depts, which made the badge
    permanently locked for any student who never tried Endoscopy (best=0).
    Now the badge only requires mastery in depts where the student has any sessions.
    """
    attempted = {k for k in DEPT_KEYS if s.get("scores", {}).get(k)}
    if len(attempted) < len(DEPT_KEYS):
        return False  # must have at least one session in every dept
    return all(s.get("best", {}).get(k, 0) >= 80 for k in DEPT_KEYS)


BADGE_RULES = [
    ("First Steps",     "🎓", lambda s: s.get("total_sessions", 0) >= 1),
    ("Steady Hand",     "✋", lambda s: s.get("best", {}).get("surgery", 0) >= 80),
    ("Probe Master",    "🔊", lambda s: s.get("best", {}).get("ultrasound", 0) >= 85),
    ("CT Explorer",     "🩻", lambda s: s.get("best", {}).get("ct", 0) >= 70),
    ("FLS Bronze",      "🥉", lambda s: s.get("best", {}).get("fls", 0) >= 70),
    ("FLS Silver",      "🥈", lambda s: s.get("best", {}).get("fls", 0) >= 85),
    ("Scope Navigator", "🔭", lambda s: s.get("best", {}).get("endoscopy", 0) >= 75),
    ("Reasoner",        "🧠", lambda s: s.get("best", {}).get("reasoning", 0) >= 80),
    ("Master Reasoner", "🎓", lambda s: s.get("best", {}).get("reasoning", 0) >= 95),
    ("Pathologist",     "🔬", lambda s: s.get("best", {}).get("pathology", 0) >= 80),
    ("Catheter Confident", "🩺", lambda s: s.get("best", {}).get("foley", 0) >= 80),
    ("Airway Expert",   "🫁", lambda s: s.get("best", {}).get("intubation", 0) >= 80),
    ("Scope Explorer",  "🔭", lambda s: s.get("best", {}).get("endoscopy_scope", 0) >= 80),
    ("Top Surgeon",     "🏆", _top_surgeon),
]


def _path(sid):
    return os.path.join(STUDENTS_DIR, f"{sid}.json")


def _empty(name):
    return {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "registered": datetime.datetime.now().isoformat(),
        "total_sessions": 0,
        "scores":  {k: [] for k in DEPT_KEYS},
        "best":    {k: 0  for k in DEPT_KEYS},
        "avg":     {k: 0  for k in DEPT_KEYS},
        "badges":  [],
        "sessions": [],
    }


def register_student(name: str) -> dict:
    """Kept for backwards compatibility, but no longer called by hospital_portal.
    Guest mode generates IDs via make_guest_id() instead."""
    sid = "s_" + str(uuid.uuid4())[:6]
    data = _empty(name)
    data["id"] = sid
    save_student(data)
    return data


def login_student(sid: str):
    p = _path(sid)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupted JSON — treat as missing rather than crashing
        return None


def load_student(sid: str) -> dict:
    """Load student, defensively filling in missing keys for older JSONs."""
    data = login_student(sid)
    if data is None:
        # No local cache for this id yet. If it's a signed-in student (sid is
        # their real email) and Supabase is configured, rebuild their record
        # from the persisted vh_sim_sessions history instead of starting them
        # over as a blank Guest — this is what makes progress survive a
        # container restart/new device. Guests with no Supabase hit just
        # get the normal blank record, same as before.
        try:
            if st.session_state.get("signed_in"):
                from sim_account import get_sim_best_and_avg, get_sim_history
                hist = get_sim_best_and_avg(sid, DEPT_KEYS)
                data = _empty(st.session_state.get("student_name", "Student"))
                data["id"] = sid
                data["best"] = hist["best"]
                data["avg"] = hist["avg"]
                data["scores"] = hist["scores"]
                data["total_sessions"] = hist["total_sessions"]
                for row in get_sim_history(sid):
                    data["sessions"].append({"dept": row.get("dept"), "score": row.get("score"),
                                              "notes": row.get("notes", ""), "ts": row.get("created_at", "")})
                for bname, icon, rule in BADGE_RULES:
                    key = f"{icon}{bname}"
                    try:
                        if rule(data) and key not in data["badges"]:
                            data["badges"].append(key)
                    except Exception:
                        pass
                return data
        except Exception:
            pass
        return _empty("Guest")
    # Backfill any keys added in later versions
    data.setdefault("scores",  {k: [] for k in DEPT_KEYS})
    data.setdefault("best",    {k: 0  for k in DEPT_KEYS})
    data.setdefault("avg",     {k: 0  for k in DEPT_KEYS})
    data.setdefault("badges",  [])
    data.setdefault("sessions", [])
    data.setdefault("total_sessions", sum(len(v) for v in data["scores"].values()))
    return data


def save_student(data: dict):
    """Atomic write: temp file + os.replace.

    Survives Ctrl+C / OOM / power-loss mid-write. Without this, a crash
    during write corrupts the student JSON and the user gets logged in as Guest.
    """
    target = _path(data["id"])
    tmp = target + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, target)


def all_students() -> list:
    out = []
    for p in glob.glob(os.path.join(STUDENTS_DIR, "*.json")):
        try:
            with open(p) as f:
                out.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return out


def leaderboard_data() -> list:
    rows = []
    for s in all_students():
        overall = round(sum(s.get("best", {}).get(k, 0) for k in DEPT_KEYS) / len(DEPT_KEYS), 1)
        rows.append({
            "name": s["name"],
            "id": s["id"],
            "overall": overall,
            "sessions": s.get("total_sessions", 0),
            "badges": len(s.get("badges", [])),
        })
    return sorted(rows, key=lambda r: r["overall"], reverse=True)


def record_session(sid: str, dept: str, score: float, notes: str = "") -> dict:
    # Whitelist check — fix: previously a typo (e.g. "crt") silently created a
    # phantom column in the student JSON that never showed up on the dashboard.
    if dept not in DEPT_KEYS:
        raise ValueError(
            f"Unknown department '{dept}'. "
            f"Must be one of: {', '.join(DEPT_KEYS)}"
        )
    data = load_student(sid)
    score = round(float(score), 1)
    data["scores"].setdefault(dept, []).append(score)
    data["total_sessions"] = data.get("total_sessions", 0) + 1
    data["best"][dept] = max(data["best"].get(dept, 0), score)
    arr = data["scores"][dept]
    data["avg"][dept] = round(sum(arr) / len(arr), 1)
    data["sessions"].append({
        "dept": dept,
        "score": score,
        "notes": notes,
        "ts": datetime.datetime.now().isoformat(),
    })
    for bname, icon, rule in BADGE_RULES:
        key = f"{icon}{bname}"
        try:
            if rule(data) and key not in data["badges"]:
                data["badges"].append(key)
        except Exception:
            pass
    save_student(data)

    # If this is a signed-in student (sid is their real email, not a guest
    # id), also mirror the score into Supabase so it survives across devices
    # and sessions. Purely additive — guests and Supabase-unconfigured
    # deployments are completely unaffected, local JSON above already ran.
    try:
        if st.session_state.get("signed_in"):
            from sim_account import save_sim_session
            save_sim_session(sid, dept, score, notes)
    except Exception:
        pass

    return data


def dept_header(title: str, icon: str, student_id: str):
    st.markdown(
        f'''<div style="background:#23232a;padding:1rem 1.5rem;border-radius:10px;
        margin-bottom:1rem;border-left:4px solid #ffd400;">
        <span style="font-size:2rem">{icon}</span>
        <span style="font-size:1.4rem;font-weight:700;color:#fbfbff;margin-left:.6rem">{title}</span>
        <span style="float:right;color:#909094;font-size:.85rem">Session: {student_id}</span>
        </div>''',
        unsafe_allow_html=True
    )


# ============================================================================
# FREE-TRIAL STUB — kept for backwards compatibility
# In guest mode this is never called, but old code may still reference it.
# ============================================================================
def is_enrolled(sid: str, dept: str) -> bool:
    """STUB — always allow during free trial phase / guest mode."""
    return True
