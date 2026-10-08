"""
sim_account.py
────────────────────────────────────────────────────────────────────────────
Real, persistent student identity for the AI MLS Simulation Hospital —
reusing the EXACT SAME Supabase project and the EXACT SAME vh_users table
the Virtual Hospital already authenticates against. A student's Virtual
Hospital login (email + password) works here too, unchanged — this is the
actual integration point between the two apps, not just two link buttons.

Design, on purpose:
  - Password hashing matches app.py's _hash_pw() exactly (SHA-256 of the raw
    password) so a password set in one app verifies correctly in the other.
  - This file never writes to vh_users itself for login — only reads it, the
    same read-only check app.py's _get_user() does. Registering a brand-new
    account still happens in the Virtual Hospital (it already has the full
    register form, security questions, mentor-approval flow, etc.) — this
    file deliberately does not duplicate that. Students without an account
    yet are told to register there, or can continue as Guest (unchanged,
    original behavior, fully preserved).
  - All persistent sim scores go in a NEW table, vh_sim_sessions, keyed by
    the student's email — never written into vh_users.

Guest mode is untouched: if Supabase isn't configured, or the student picks
"Continue as Guest", everything falls back to shared_session.py's original
local-JSON behavior exactly as before. This file only adds a path for
students who choose to sign in.

Requires one new Supabase table (same project as vh_users):

    create table vh_sim_sessions (
      id            bigint generated always as identity primary key,
      student_email   text not null,
      dept              text not null,
      score               numeric not null,
      notes                 text,
      created_at              timestamp default now()
    );
"""

import hashlib
import requests
import streamlit as st

TABLE = "vh_sim_sessions"


def _sb_creds():
    try:
        return st.secrets.get("SUPABASE_URL", ""), st.secrets.get("SUPABASE_KEY", "")
    except Exception:
        return "", ""


def sb_available() -> bool:
    url, key = _sb_creds()
    return bool(url and key and not url.startswith("YOUR_"))


def _sb_headers(key: str) -> dict:
    return {"apikey": key, "Authorization": f"Bearer {key}",
            "Content-Type": "application/json", "Prefer": "return=representation"}


def _hash_pw(pw: str) -> str:
    """Identical to app.py's _hash_pw() — SHA-256 of the raw password, so a
    password set in the Virtual Hospital verifies correctly here too."""
    return hashlib.sha256(pw.encode()).hexdigest()


def login(email: str, password: str):
    """Check credentials against the SAME vh_users table the Virtual
    Hospital uses. Returns {id, name, email, role} on success, or a
    (False, reason) tuple on failure — never writes to vh_users."""
    if not sb_available():
        return False, "Supabase isn't configured for this app yet."
    url, key = _sb_creds()
    email = email.lower().strip()
    ph = _hash_pw(password)
    try:
        r = requests.get(f"{url}/rest/v1/vh_users", headers=_sb_headers(key),
                          params={"email": f"eq.{email}", "select": "*"}, timeout=8)
        if r.status_code != 200:
            return False, f"Could not reach the account system ({r.status_code})."
        rows = r.json()
        if not rows:
            return False, "No account with that email — register in the Virtual Hospital first."
        stored_hash = rows[0].get("password_hash", "")
        if stored_hash != ph:
            return False, "Incorrect password."
        u = rows[0]
        return {"id": u["id"], "name": u["name"], "email": u["email"],
                "role": u.get("role", "student")}
    except Exception as e:
        return False, f"Login failed: {e}"


def save_sim_session(email: str, dept: str, score: float, notes: str = ""):
    """Persist one score record for a signed-in student. Silently no-ops if
    Supabase isn't configured — the caller's local-JSON fallback still runs,
    so nothing breaks for guest mode."""
    if not sb_available() or not email:
        return
    url, key = _sb_creds()
    try:
        requests.post(f"{url}/rest/v1/{TABLE}", headers=_sb_headers(key),
                       json={"student_email": email.lower().strip(), "dept": dept,
                             "score": float(score), "notes": notes}, timeout=10)
    except Exception:
        pass  # never let a scoring write crash the page


def get_sim_history(email: str) -> list:
    """All persisted session rows for a signed-in student, newest first."""
    if not sb_available() or not email:
        return []
    url, key = _sb_creds()
    try:
        r = requests.get(f"{url}/rest/v1/{TABLE}", headers=_sb_headers(key),
                          params={"student_email": f"eq.{email.lower().strip()}",
                                   "select": "*", "order": "created_at.desc"}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def get_sim_best_and_avg(email: str, dept_keys: list) -> dict:
    """Recompute best/avg per department from the persisted rows — mirrors
    shared_session.py's local-JSON shape so the rest of the app (badges,
    leaderboard) can treat a signed-in student the same way as a guest."""
    history = get_sim_history(email)
    best = {k: 0 for k in dept_keys}
    scores = {k: [] for k in dept_keys}
    for row in history:
        d = row.get("dept")
        if d in scores:
            scores[d].append(row["score"])
            best[d] = max(best[d], row["score"])
    avg = {k: (round(sum(v) / len(v), 1) if v else 0) for k, v in scores.items()}
    return {"best": best, "avg": avg, "scores": scores, "total_sessions": len(history)}


def render_login_box():
    """Small sign-in form for the landing page. Returns the logged-in user
    dict on success (also stashes it in session_state), or None."""
    with st.form("sim_login_form"):
        st.caption("Have a Virtual Hospital account? Sign in to save your progress permanently "
                   "across devices. (Or skip this and continue as Guest below.)")
        email = st.text_input("Email", key="sim_login_email")
        pw = st.text_input("Password", type="password", key="sim_login_pw")
        go = st.form_submit_button("Sign in", type="primary")
        if go:
            if not email.strip() or not pw:
                st.error("Enter both email and password.")
                return None
            result = login(email, pw)
            if isinstance(result, dict):
                st.success(f"Welcome back, {result['name']}!")
                return result
            else:
                _, msg = result
                st.error(msg)
                return None
    return None
