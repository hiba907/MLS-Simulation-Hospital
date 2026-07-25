import streamlit as st
import plotly.graph_objects as go
import datetime, math, os, json, glob
from shared_session import dept_header, load_student, all_students, leaderboard_data, DEPT_KEYS

PALETTE = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF"]


def _demo_student():
    import random
    random.seed(42)
    scores = {k: [random.randint(45, 95) for _ in range(8)] for k in DEPT_KEYS}
    best = {k: max(v) for k, v in scores.items()}
    avg = {k: round(sum(v) / len(v), 1) for k, v in scores.items()}
    sessions = []
    for i in range(8):
        for k in DEPT_KEYS:
            sessions.append({
                "dept": k,
                "score": scores[k][i],
                "ts": (datetime.datetime.now() - datetime.timedelta(days=40 - i * 5)).isoformat(),
                "notes": "",
            })
    return {
        "name": "Demo Student",
        "id": "demo",
        "total_sessions": 40,
        "scores": scores,
        "best": best,
        "avg": avg,
        "badges": ["🎓First Steps", "✋Steady Hand"],
        "sessions": sessions,
    }


def render(student_id: str):
    dept_header("Analytics Dashboard", "📊", student_id)

    data = load_student(student_id)
    if data["total_sessions"] == 0:
        st.info("No real sessions yet — showing demo data. Train in any department to see your own data.")
        data = _demo_student()

    tab_mine, tab_board = st.tabs(["📈 My Analytics", "🏆 Leaderboard"])

    with tab_mine:
        # KPI cards
        # Fix: was labelled "Overall Best" but is actually an average of bests.
        # Renamed to "Avg Best Score" so students aren't confused about what
        # the number means.
        overall = round(sum(data["best"].get(k, 0) for k in DEPT_KEYS) / len(DEPT_KEYS), 1)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sessions", data["total_sessions"])
        c2.metric("Avg Best Score", f"{overall}/100")
        c3.metric("Badges Earned",  len(data["badges"]))
        c4.metric("Departments",    sum(1 for k in DEPT_KEYS if data["best"].get(k, 0) > 0))

        # Score trend
        sessions = data.get("sessions", [])
        if sessions:
            dept_traces = {}
            for s in sessions:
                dept_traces.setdefault(s["dept"], []).append(s["score"])
            fig = go.Figure()
            for i, (dept, scores) in enumerate(dept_traces.items()):
                fig.add_trace(go.Scatter(
                    y=scores, mode="lines+markers", name=dept,
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2)
                ))
            fig.update_layout(
                title="Score Trend",
                paper_bgcolor="#1D1D20", plot_bgcolor="#23232a",
                font_color="#fbfbff", legend=dict(bgcolor="#1D1D20"),
                xaxis_title="Session", yaxis_title="Score", yaxis_range=[0, 100]
            )
            st.plotly_chart(fig, use_container_width=True)

        # Skill radar
        r_vals = [data["best"].get(k, 0) for k in DEPT_KEYS]
        fig2 = go.Figure(go.Scatterpolar(
            r=r_vals + [r_vals[0]],
            theta=[k.title() for k in DEPT_KEYS] + [DEPT_KEYS[0].title()],
            fill="toself", line_color="#ffd400"
        ))
        fig2.update_layout(
            polar=dict(bgcolor="#23232a", radialaxis=dict(range=[0, 100])),
            paper_bgcolor="#1D1D20", font_color="#fbfbff", title="Skill Radar"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Badges
        st.markdown("### 🏅 Badges")
        if data["badges"]:
            st.markdown("  ".join(f"`{b}`" for b in data["badges"]))
        else:
            st.info("Complete training sessions to earn badges!")

    with tab_board:
        rows = leaderboard_data()
        if not rows:
            st.info("No students registered yet.")
            return
        names = [r["name"] for r in rows[:10]]
        overall = [r["overall"] for r in rows[:10]]
        colors = ["#ffd400" if r["id"] == student_id else "#A1C9F4" for r in rows[:10]]
        fig3 = go.Figure(go.Bar(
            x=names, y=overall, marker_color=colors,
            text=[f"{v}" for v in overall], textposition="outside"
        ))
        # Fix: y-axis label was "Overall Score" but the value is an average of bests.
        fig3.update_layout(
            title="Top 10 Students",
            paper_bgcolor="#1D1D20", plot_bgcolor="#23232a",
            font_color="#fbfbff", yaxis_range=[0, 105],
            xaxis_title="Student", yaxis_title="Avg Best Score"
        )
        st.plotly_chart(fig3, use_container_width=True)
