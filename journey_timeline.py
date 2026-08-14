"""Patient Journey Timeline — horizontal clinical-event visualization.

Renders a case's `case_journey` events as colored dots on a horizontal
time axis, with patient age in years on the x-axis (matching the Substack
'nuraxi' visualization style). Designed to slot into the existing
Reasoning Canvas UI as a separate tab ('🛏️ Patient Journey') or as a
standalone streamlit page.

This module is independent of reasoning_canvas.py — pass it any case dict
that has a `case_journey` list of events.

No ML. No model training. Pure Python + plotly. Tested with plotly>=5.18.
"""
from __future__ import annotations
import math
from typing import List, Dict, Optional

# Category → color (matches the Substack visualization palette)
CATEGORY_COLORS = {
    "critical":   "#e74c3c",   # red — must-do steps
    "severity":   "#e67e22",   # orange — abnormal findings
    "negated":    "#16a085",   # teal — things ruled out / denied
    "temporal":   "#3498db",   # blue — timing markers
    "stable":     "#27ae60",   # green — patient improving / stable
    "supportive": "#9b59b6",   # purple — adjuncts
    "discharge":  "#2c3e50",   # dark — endpoints
}

CATEGORY_ICONS = {
    "critical":   "■",
    "severity":   "●",
    "negated":    "○",
    "temporal":   "◆",
    "stable":     "▲",
    "supportive": "★",
    "discharge":  "⬛",
}


def _seconds_to_label(seconds: float) -> str:
    """Convert seconds from arrival to a human label (T+5m, T+2h, T+3d)."""
    if seconds < 0:
        return f"T{seconds/60:.0f}m" if seconds > -3600 else f"T{seconds/3600:.1f}h"
    if seconds < 60:
        return f"T+{int(seconds)}s"
    if seconds < 3600:
        return f"T+{int(seconds/60)}m"
    if seconds < 86400:
        h = seconds / 3600
        return f"T+{h:.1f}h".replace(".0h", "h")
    d = seconds / 86400
    return f"T+{d:.1f}d".replace(".0d", "d")


def _age_label(start_age_years: float, t_seconds: float) -> str:
    age = start_age_years + (t_seconds / 86400 / 365.25)
    return f"{age:.3f} yr"


def _t_to_x(t: float, t_min: float, t_max: float) -> float:
    """Map time in seconds to a 0-100 x-coordinate.

    Clinical case timelines have a brutal distribution problem: ~30 events
    happen in the first 2 hours, then a handful spread over days. Pure
    linear time crushes everything to the left; pure sqrt makes the
    follow-up look like it's right after the ED workup. We use a piecewise
    scale that gives the first hour real space, the rest of the ED workup
    moderate space, and long-term events their own region.

    Regions (% of width):
      pre-arrival  (t < 0)             : 12%
      0 - 30 min   (initial resuscitation) : 26%
      30 min - 6h                       : 22%
      6h - 24h                          : 15%
      1d - 7d                           : 15%
      > 7d                              : 10%
    """
    if t_max == t_min:
        return 50.0

    if t < 0:
        # Pre-arrival history. Compress: assume |t| <= 24h
        norm = min(1.0, max(0.0, -t / (24 * 3600)))
        return 12.0 * norm

    # Acute / subacute / chronic regions
    if t < 30 * 60:           # first 30 min
        local = t / (30 * 60)
        return 12.0 + 26.0 * local
    if t < 6 * 3600:          # 30 min - 6h
        local = (t - 30 * 60) / (6 * 3600 - 30 * 60)
        return 12.0 + 26.0 + 22.0 * local
    if t < 24 * 3600:         # 6h - 24h
        local = (t - 6 * 3600) / (24 * 3600 - 6 * 3600)
        return 12.0 + 26.0 + 22.0 + 15.0 * local
    if t < 7 * 24 * 3600:     # 1d - 7d
        local = (t - 24 * 3600) / (7 * 24 * 3600 - 24 * 3600)
        return 12.0 + 26.0 + 22.0 + 15.0 + 15.0 * local
    # > 7d — log scale up to t_max
    t7 = 7 * 24 * 3600
    if t_max <= t7:
        return 95.0
    log_norm = math.log(t / t7) / math.log(t_max / t7)
    return 12.0 + 26.0 + 22.0 + 15.0 + 15.0 + 10.0 * min(1.0, max(0.0, log_norm))


def render_journey_timeline(
    case: dict,
    start_age_years: float = 58.0,
    title: Optional[str] = None,
    height: int = 700,
    width: int = 2400,
):
    """Render a patient-journey timeline.

    case['case_journey'] = list of {
        't':         seconds from arrival (float, can be negative for pre-arrival)
        'event':     short event description (str)
        'type':      'history' | 'exam' | 'lab' | 'imaging' | 'treatment' |
                     'procedure' | 'arrival' | 'disposition' | 'follow_up' | 'decision'
        'category':  'critical' | 'severity' | 'negated' | 'temporal' |
                     'stable' | 'supportive' | 'discharge'
        'label':     optional short label for the node
    }

    Returns a plotly figure. In Streamlit, call st.plotly_chart(fig, ...).
    """
    import plotly.graph_objects as go

    journey: List[dict] = case.get("case_journey", [])
    if not journey:
        fig = go.Figure()
        fig.add_annotation(
            text="No case_journey data — author one to see the timeline",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    # Sort by time, then assign lanes and x-jitter
    journey = sorted(journey, key=lambda e: e.get("t", 0))
    t_min = min(e["t"] for e in journey)
    t_max = max(e["t"] for e in journey)
    # Pad by 1 minute on each side
    t_lo = t_min - 60
    t_hi = t_max + 60

    # Lane assignment: alternates above/below the axis, cycled through 8 lanes
    # on each side. For dense case histories this gives clean vertical separation.
    N_LANES_PER_SIDE = 8

    # Count events per exact timestamp so we can give them x-jitter when many
    # happen at the same moment (e.g. all the pre-arrival history, all the
    # immediate post-PCI events).
    ts_counts: Dict[float, int] = {}
    for ev in journey:
        ts_counts[ev["t"]] = ts_counts.get(ev["t"], 0) + 1
    ts_seen: Dict[float, int] = {}

    event_lane: Dict[int, int] = {}
    for i, ev in enumerate(journey):
        side = 1 if i % 2 == 0 else -1
        offset = ((i // 2) % N_LANES_PER_SIDE) + 1
        event_lane[id(ev)] = side * offset

    fig = go.Figure()

    # The horizontal axis line — drawn in x-units (0-100) since we're using
    # the compressed scale for layout. The age labels are drawn separately.
    fig.add_shape(
        type="line", x0=0, x1=100, y0=0, y1=0,
        line=dict(color="#7f8c8d", width=2),
    )

    # Anchor markers
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 0],
        mode="markers",
        marker=dict(color="#34495e", size=10, symbol="circle"),
        showlegend=False, hoverinfo="skip",
    ))

    # Count events per exact timestamp so we can give them x-jitter when many
    # happen at the same moment (e.g. all the pre-arrival history, all the
    # immediate post-PCI events).
    ts_counts: Dict[float, int] = {}
    for ev in journey:
        ts_counts[ev["t"]] = ts_counts.get(ev["t"], 0) + 1
    ts_seen: Dict[float, int] = {}

    # Plot each event
    for ev in journey:
        t = ev["t"]
        cat = ev.get("category", "temporal")
        color = CATEGORY_COLORS.get(cat, "#95a5a6")
        icon = CATEGORY_ICONS.get(cat, "•")
        chosen_lane = event_lane[id(ev)]
        x = _t_to_x(t, t_lo, t_hi)
        # Add x-jitter if multiple events share this exact timestamp, so
        # coincident events don't fully overlap.
        n_at_t = ts_counts.get(t, 1)
        idx_at_t = ts_seen.get(t, 0)
        ts_seen[t] = idx_at_t + 1
        if n_at_t > 1:
            if n_at_t == 2:
                offsets = [-0.5, 0.5]
            elif n_at_t == 3:
                offsets = [-0.7, 0.0, 0.7]
            elif n_at_t == 4:
                offsets = [-0.9, -0.3, 0.3, 0.9]
            else:
                offsets = [-1.2 + 2.4 * k / (n_at_t - 1) for k in range(n_at_t)]
            x = x + offsets[idx_at_t] * 1.4
        y_label = chosen_lane * 0.45

        # The dot on the axis
        fig.add_trace(go.Scatter(
            x=[x], y=[0],
            mode="markers",
            marker=dict(color=color, size=12, symbol="circle",
                        line=dict(color="white", width=1.5)),
            showlegend=False,
            hovertemplate=(
                f"<b>{ev.get('event','')}</b><br>"
                f"t = {_seconds_to_label(t)}<br>"
                f"age = {_age_label(start_age_years, t)}<br>"
                f"type = {ev.get('type','')}<br>"
                f"category = {cat}<extra></extra>"
            ),
        ))

        # The leader line
        fig.add_shape(
            type="line", x0=x, x1=x, y0=0, y1=y_label,
            line=dict(color=color, width=1, dash="dot"),
        )

        # The label text
        label_text = ev.get("event", "")
        if ev.get("label") and ev["label"] != label_text:
            label_text = ev["label"]
        # Truncate very long labels so they don't blow out the layout
        if len(label_text) > 45:
            label_text = label_text[:42] + "..."

        fig.add_annotation(
            x=x, y=y_label,
            text=f"<span style='color:{color};font-weight:600'>{icon}</span> "
                 f"<span style='color:#2c3e50;font-size:10px'>{label_text}</span>",
            showarrow=False,
            font=dict(size=10, color="#2c3e50"),
            xanchor="left" if chosen_lane > 0 else "right",
            yanchor="middle",
        )

    # X-axis age labels at evenly spaced positions on the compressed scale
    n_ticks = 7
    tick_fracs = [i / (n_ticks - 1) for i in range(n_ticks)]
    tick_seconds = [t_lo + (t_hi - t_lo) * f * f for f in tick_fracs]  # invert sqrt
    tick_ages = [_age_label(start_age_years, t) for t in tick_seconds]
    tick_short = [_seconds_to_label(t) for t in tick_seconds]

    fig.update_xaxes(
        tickvals=[_t_to_x(t, t_lo, t_hi) for t in tick_seconds],
        ticktext=[f"<b>{a}</b><br><span style='font-size:9px;color:#7f8c8d'>{t}</span>"
                  for a, t in zip(tick_ages, tick_short)],
        range=[-2, 102],
        title="",
        showgrid=False, zeroline=False, showline=False,
    )
    fig.update_yaxes(
        range=[-3.8, 3.8], visible=False,
    )

    # Legend
    legend_entries = [
        ("Critical (must-do)", "critical"),
        ("Severity (abnormal)", "severity"),
        ("Negated / ruled out", "negated"),
        ("Stable / improving", "stable"),
        ("Timing / arrival", "temporal"),
        ("Supportive", "supportive"),
        ("Discharge / end", "discharge"),
    ]
    for name, cat in legend_entries:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=CATEGORY_COLORS[cat], size=11, symbol="circle"),
            name=name, showlegend=True,
        ))

    fig.update_layout(
        title=dict(
            text=title or case.get("title", "Patient Journey"),
            x=0.02, xanchor="left", font=dict(size=14, color="#2c3e50"),
        ),
        height=height,
        width=width,
        margin=dict(l=20, r=20, t=60, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1.0, font=dict(size=10),
        ),
        hovermode="closest",
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate a sample STEMI journey (matches the Substack screenshot)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_STEMI_JOURNEY = {
    "id": "stemi_58m_journey",
    "title": "Mr. Khalil, 58M — Inferior STEMI (full hospital course)",
    "case_journey": [
        {"t": -7200, "event": "Symptom onset 2h prior to arrival",
         "type": "history", "category": "temporal", "label": "Onset 2h prior"},
        {"t": -7200, "event": "Denies syncope or palpitations",
         "type": "history", "category": "negated"},
        {"t": -7200, "event": "History of T2DM — 8yr duration",
         "type": "history", "category": "severity"},
        {"t": -7200, "event": "Family history of CAD — father, age 62",
         "type": "history", "category": "severity"},
        {"t": -7200, "event": "Current smoker — 20 pack-years",
         "type": "history", "category": "severity"},
        {"t": 0,    "event": "ED arrival",
         "type": "arrival", "category": "temporal"},
        {"t": 30,   "event": "Vitals + cardiovascular exam",
         "type": "exam", "category": "critical"},
        {"t": 45,   "event": "12-lead ECG — ST elevation II, III, aVF",
         "type": "imaging", "category": "critical", "label": "ECG — STEMI inferior wall"},
        {"t": 60,   "event": "Troponin I — 4.8 ng/mL markedly elevated",
         "type": "lab", "category": "severity"},
        {"t": 80,   "event": "Aspirin 325 mg loading dose",
         "type": "treatment", "category": "critical"},
        {"t": 90,   "event": "Ticagrelor 180 mg loading dose",
         "type": "treatment", "category": "critical"},
        {"t": 100,  "event": "BP 88/54 — hypotension, acute",
         "type": "exam", "category": "severity"},
        {"t": 105,  "event": "HR 112 — sinus tachycardia",
         "type": "exam", "category": "severity"},
        {"t": 120,  "event": "IV normal saline 500 mL bolus — cautious",
         "type": "treatment", "category": "supportive"},
        {"t": 120,  "event": "Creatinine 1.4 mg/dL — mildly elevated",
         "type": "lab", "category": "negated"},
        {"t": 120,  "event": "Heparin bolus 60 U/kg",
         "type": "treatment", "category": "critical"},
        {"t": 150,  "event": "Glucose 218 mg/dL — poorly controlled",
         "type": "lab", "category": "severity"},
        {"t": 150,  "event": "Bedside echo — EF 30%, inferior hypokinesis",
         "type": "imaging", "category": "severity"},
        {"t": 150,  "event": "Door-to-balloon time 68 min",
         "type": "procedure", "category": "temporal"},
        {"t": 180,  "event": "PCI to RCA — DES placed, TIMI 3 flow restored",
         "type": "procedure", "category": "critical"},
        {"t": 200,  "event": "Chest pain — resolved post-PCI (0/10)",
         "type": "exam", "category": "stable"},
        {"t": 240,  "event": "Moderate mitral regurgitation — new",
         "type": "exam", "category": "severity"},
        {"t": 270,  "event": "Nausea — moderate, without vomiting",
         "type": "exam", "category": "negated"},
        {"t": 300,  "event": "Morphine 4 mg IV — pain severe (8/10)",
         "type": "treatment", "category": "supportive"},
        {"t": 300,  "event": "Atorvastatin 80 mg started",
         "type": "treatment", "category": "stable"},
        {"t": 360,  "event": "Transferred to step-down unit",
         "type": "disposition", "category": "stable"},
        {"t": 400,  "event": "Metoprolol 12.5 mg initiated — BP tolerated",
         "type": "treatment", "category": "stable"},
        {"t": 480,  "event": "Ramipril 2.5 mg initiated — creatinine stable",
         "type": "treatment", "category": "stable"},
        {"t": 600,  "event": "Repeat echo — EF 35%, improving",
         "type": "imaging", "category": "stable"},
        {"t": 1440, "event": "Noradrenaline discontinued",
         "type": "treatment", "category": "stable"},
        {"t": 1440, "event": "Ambulations started — 50m with assistance",
         "type": "treatment", "category": "stable"},
        {"t": 2880, "event": "Cardiac rehab referral placed",
         "type": "disposition", "category": "stable"},
        {"t": 4320, "event": "Dietitian review — cardiac diet education",
         "type": "treatment", "category": "supportive"},
        {"t": 4320, "event": "Mood low — adjustment reaction discussed",
         "type": "exam", "category": "negated"},
        {"t": 7200, "event": "Discharged home — stable condition",
         "type": "disposition", "category": "discharge"},
        {"t": 7200, "event": "Discharge meds: ASA, ticagrelor, metoprolol, ramipril, atorvastatin",
         "type": "treatment", "category": "discharge"},
        {"t": 403200, "event": "Follow-up echo in 6 weeks",
         "type": "follow_up", "category": "temporal"},
    ],
}


if __name__ == "__main__":
    fig = render_journey_timeline(SAMPLE_STEMI_JOURNEY, start_age_years=58.0,
                                  width=2400, height=900)
    try:
        fig.show()
    except Exception:
        fig.write_html("/workspace/journey_timeline_demo.html")
        print("Wrote /workspace/journey_timeline_demo.html")
