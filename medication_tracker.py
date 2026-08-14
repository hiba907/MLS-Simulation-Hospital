"""Medication timeline + Adverse Drug Reaction (ADR) tracker.

Extends the Reasoning Canvas + Elembic pipeline with structured drug tracking:

  - medication_timeline: every drug the patient is on, with start/stop/dose/status
  - drug_reactions: any adverse drug reactions, with WHO-UMC causality assessment
  - lab_monitoring: drug-level lab tracking (e.g. creatinine on ACE-i, INR on warfarin)

Three new update types for the follow_up_timeline:
  - drug_started
  - drug_stopped
  - dose_changed
  - adverse_reaction
  - lab_monitoring
  - adherence_issue

A resident doing the case sees the initial medication list, then later updates
as new drugs are added, doses changed, or reactions occur. The system tracks
how their reasoning evolved with the new drug data.

Visualization: a "Med Timeline" view that shows each drug as a horizontal bar
on a patient-age axis, with reaction events as red dots and lab checks as
blue dots overlaid.

No model training. The drug events are authored by the case author (attending)
or extracted from the clinical note by the elembic_pipeline.py NER step.
"""
from __future__ import annotations
import json
import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Medication:
    """A single drug the patient is on (or was on)."""
    name: str                          # generic name preferred: "lisinopril"
    brand: Optional[str] = None        # brand name if relevant: "Zestril"
    rxnorm_code: Optional[str] = None  # RxNorm CUI if available
    dose: Optional[str] = None         # "10 mg", "5 mL", etc.
    route: str = "PO"                  # PO, IV, IM, SC, inhaled, topical
    frequency: Optional[str] = None    # "BID", "q8h", "once daily"
    indication: Optional[str] = None   # why prescribed: "HFrEF", "HTN"
    started_at: Optional[float] = None # seconds from admission
    stopped_at: Optional[float] = None # None = ongoing
    status: str = "active"             # active | stopped | completed | held
    stop_reason: Optional[str] = None  # "completed course", "ADR", "ineffective"
    notes: str = ""


@dataclass
class AdverseReaction:
    """An adverse drug reaction (ADR) — a harmful response to a medication."""
    drug: str                          # the suspected drug
    reaction: str                      # what happened: "dry cough", "rash"
    onset: float                       # seconds from admission (or from drug start)
    severity: str                      # mild | moderate | severe | life_threatening
    causality: str                     # WHO-UMC: certain | probable | possible | unlikely | conditional | unassessable
    action_taken: str                  # "drug stopped", "dose reduced", "switched", "none"
    lab_evidence: Optional[str] = None # e.g. "Cr 1.4 → 2.1", "K 5.8"
    notes: str = ""


@dataclass
class LabCheck:
    """A monitoring lab ordered because of a drug."""
    drug: str                          # drug being monitored
    test: str                          # "creatinine", "INR", "K+", "LFTs", "CBC"
    value: str                         # "1.4 mg/dL", "2.3"
    at: float                          # seconds from admission
    interpretation: str                # "normal", "elevated", "worsening", "improving"
    notes: str = ""


@dataclass
class AdherenceNote:
    """An adherence or patient-behavior note."""
    drug: str
    at: float
    issue: str                         # "missed 3 doses", "double-dosed", "discontinued on own"
    action: str                        # "patient educated", "switched to long-acting"
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# New update types for the existing follow_up_timeline
# ─────────────────────────────────────────────────────────────────────────────
DRUG_UPDATE_TYPES = [
    ("drug_started",      "💊 Drug started",         "New medication added to the regimen"),
    ("drug_stopped",      "🛑 Drug stopped",         "Medication discontinued (course complete, ADR, or other reason)"),
    ("dose_changed",      "📊 Dose changed",         "Up- or down-titration"),
    ("adverse_reaction",  "⚠️ Adverse reaction",     "Suspected ADR — capture for causality assessment"),
    ("lab_monitoring",    "🧪 Lab monitored",        "Drug-level lab (Cr for ACE-i, INR for warfarin, etc.)"),
    ("adherence_issue",   "🔄 Adherence issue",      "Patient not taking as prescribed"),
]


# ─────────────────────────────────────────────────────────────────────────────
# WHO-UMC causality assessment (the standard for ADR causality)
# ─────────────────────────────────────────────────────────────────────────────
WHO_UMC_CRITERIA = {
    "certain":   "Event occurs in a plausible time relation to drug intake, "
                 "cannot be explained by concurrent disease or other drugs, "
                 "and is confirmed by re-challenge.",
    "probable":  "Event occurs in a plausible time relation to drug intake, "
                 "unlikely to be explained by concurrent disease or other drugs, "
                 "and improves on de-challenge.",
    "possible":  "Event occurs in a plausible time relation to drug intake, "
                 "but could be explained by concurrent disease or other drugs.",
    "unlikely":  "Event occurs in an implausible time relation to drug intake, "
                 "and is more likely explained by concurrent disease or other drugs.",
    "conditional":"Event occurs but more data needed for proper assessment.",
    "unassessable":"Report suggests ADR but the data cannot be judged.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Case schema extension — add to your existing case JSON
# ─────────────────────────────────────────────────────────────────────────────
def make_medication_timeline_schema() -> Dict[str, Any]:
    """Return the JSON schema for a case's medication tracking. Add this to
    the case dict alongside `vignette`, `expert_nodes`, etc."""
    return {
        "medication_timeline": [
            # list of Medication dicts
        ],
        "drug_reactions": [
            # list of AdverseReaction dicts
        ],
        "lab_monitoring": [
            # list of LabCheck dicts
        ],
        "adherence_notes": [
            # list of AdherenceNote dicts
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Medication timeline renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_medication_timeline(
    case: Dict[str, Any],
    start_age_years: float = 58.0,
    title: Optional[str] = None,
    height: int = 700,
    width: int = 2400,
):
    """Render a horizontal bar chart of every medication the patient is on,
    with reaction events (red dots) and lab checks (blue dots) overlaid.
    Uses plotly. Each drug gets its own horizontal lane.

    case['medication_timeline'] = list of {
        'name', 'brand', 'dose', 'route', 'frequency', 'indication',
        'started_at' (seconds from admission, can be negative for pre-admission),
        'stopped_at' (None for ongoing),
        'status', 'stop_reason', 'notes'
    }
    case['drug_reactions'] = list of {
        'drug', 'reaction', 'onset', 'severity', 'causality',
        'action_taken', 'lab_evidence', 'notes'
    }
    case['lab_monitoring'] = list of {drug, test, value, at, interpretation, notes}
    """
    import plotly.graph_objects as go

    meds = case.get("medication_timeline", [])
    reactions = case.get("drug_reactions", [])
    labs = case.get("lab_monitoring", [])
    adherence = case.get("adherence_notes", [])

    if not meds:
        fig = go.Figure()
        fig.add_annotation(
            text="No medication_timeline data — author one to see the chart",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    # Time bounds
    all_starts = [m.get("started_at", 0) for m in meds if m.get("started_at") is not None]
    all_stops  = [m.get("stopped_at", 0) for m in meds if m.get("stopped_at") is not None]
    all_events = all_starts + all_stops + [r.get("onset", 0) for r in reactions] \
                 + [l.get("at", 0) for l in labs] + [a.get("at", 0) for a in adherence]
    if not all_events:
        all_events = [0, 86400 * 7]
    t_min = min(all_events) - 86400     # 1 day padding
    t_max = max(all_events) + 86400

    def to_x(t):
        if t_max == t_min:
            return 50.0
        return 100.0 * (t - t_min) / (t_max - t_min)

    # Assign each unique drug a Y lane
    unique_drugs = []
    for m in meds:
        if m["name"] not in unique_drugs:
            unique_drugs.append(m["name"])
    n_lanes = len(unique_drugs)
    drug_lane = {name: i for i, name in enumerate(unique_drugs)}

    # Color by status
    status_colors = {
        "active":    "#27ae60",  # green
        "stopped":   "#7f8c8d",  # gray
        "completed": "#3498db",  # blue
        "held":      "#f39c12",  # orange
    }

    fig = go.Figure()

    # Plot each med as a horizontal bar
    for m in meds:
        name = m["name"]
        lane = drug_lane[name]
        y = n_lanes - lane - 1  # invert so first med is at top

        started = m.get("started_at", 0)
        stopped = m.get("stopped_at", t_max)
        if stopped is None:
            stopped = t_max  # ongoing

        x0 = to_x(started)
        x1 = to_x(stopped)
        color = status_colors.get(m.get("status", "active"), "#27ae60")

        # Bar
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=y - 0.35, y1=y + 0.35,
            fillcolor=color, line=dict(color=color, width=1),
            opacity=0.7,
        )
        # Drug name label at start
        label = m["name"]
        if m.get("dose"):
            label += f" {m['dose']}"
        if m.get("frequency"):
            label += f" {m['frequency']}"
        fig.add_annotation(
            x=x0, y=y + 0.55, text=label, showarrow=False,
            font=dict(size=10, color="#2c3e50"),
            xanchor="left", yanchor="bottom",
        )
        # Stop reason
        if m.get("stop_reason") and stopped != t_max:
            fig.add_annotation(
                x=x1, y=y - 0.55, text=f"⛔ {m['stop_reason']}",
                showarrow=False, font=dict(size=9, color="#c0392b"),
                xanchor="right", yanchor="top",
            )

    # Plot reactions as red dots
    for r in reactions:
        drug = r.get("drug", "")
        if drug not in drug_lane:
            continue
        lane = drug_lane[drug]
        y = n_lanes - lane - 1
        x = to_x(r.get("onset", 0))
        severity_size = {
            "mild": 10, "moderate": 14, "severe": 18, "life_threatening": 24
        }.get(r.get("severity", "moderate"), 14)
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(color="#e74c3c", size=severity_size, symbol="x",
                        line=dict(color="white", width=1)),
            name="Adverse reaction",
            showlegend=False,
            hovertemplate=(
                f"<b>⚠️ ADR: {r.get('reaction','')}</b><br>"
                f"Drug: {drug}<br>"
                f"Severity: {r.get('severity','')}<br>"
                f"Causality (WHO-UMC): {r.get('causality','')}<br>"
                f"Action: {r.get('action_taken','')}<br>"
                f"Lab evidence: {r.get('lab_evidence','n/a')}<br>"
                f"Notes: {r.get('notes','')}<extra></extra>"
            ),
        ))

    # Plot lab checks as blue triangles
    for l in labs:
        drug = l.get("drug", "")
        if drug not in drug_lane:
            continue
        lane = drug_lane[drug]
        y = n_lanes - lane - 1
        x = to_x(l.get("at", 0))
        interp_color = {
            "normal": "#3498db", "improving": "#27ae60",
            "elevated": "#f39c12", "worsening": "#e74c3c",
        }.get(l.get("interpretation", "normal"), "#3498db")
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(color=interp_color, size=10, symbol="triangle-up",
                        line=dict(color="white", width=1)),
            name="Lab monitoring",
            showlegend=False,
            hovertemplate=(
                f"<b>🧪 {l.get('test','')}: {l.get('value','')}</b><br>"
                f"Drug: {drug}<br>"
                f"Interpretation: {l.get('interpretation','')}<br>"
                f"Notes: {l.get('notes','')}<extra></extra>"
            ),
        ))

    # Y axis: drug names
    y_tickvals = list(range(n_lanes))
    y_ticktext = [unique_drugs[n_lanes - i - 1] for i in y_tickvals]

    # X axis: age labels at evenly spaced positions
    n_ticks = 7
    tick_fracs = [i / (n_ticks - 1) for i in range(n_ticks)]
    tick_seconds = [t_min + (t_max - t_min) * f for f in tick_fracs]
    tick_ages = [f"{start_age_years + (t / 86400 / 365.25):.3f} yr" for t in tick_seconds]
    tick_short = []
    for t in tick_seconds:
        if t < 0:
            tick_short.append(f"T{t/3600:.0f}h" if t > -86400 else f"T{t/86400:.0f}d")
        elif t < 3600:
            tick_short.append(f"T+{int(t/60)}m")
        elif t < 86400:
            tick_short.append(f"T+{t/3600:.1f}h")
        else:
            tick_short.append(f"T+{t/86400:.1f}d")

    fig.update_xaxes(
        tickvals=[to_x(t) for t in tick_seconds],
        ticktext=[f"<b>{a}</b><br><span style='font-size:9px;color:#7f8c8d'>{t}</span>"
                  for a, t in zip(tick_ages, tick_short)],
        range=[-3, 103],
        showgrid=False, zeroline=False, showline=False,
    )
    fig.update_yaxes(
        tickvals=y_tickvals, ticktext=y_ticktext,
        title="Drug",
        range=[-0.7, n_lanes - 0.3],
    )

    # Legend
    legend_entries = [
        ("Active drug",        "active",    "#27ae60"),
        ("Stopped drug",       "stopped",   "#7f8c8d"),
        ("Course completed",   "completed", "#3498db"),
        ("Held",               "held",      "#f39c12"),
        ("Adverse reaction",   "adr",       "#e74c3c"),
        ("Lab check (normal)", "lab_ok",    "#3498db"),
        ("Lab check (concerning)", "lab_bad", "#e74c3c"),
    ]
    for name, _, color in legend_entries:
        symbol = "x" if "reaction" in name else "square" if "drug" in name or "completed" in name or "Held" in name else "triangle-up"
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, size=11, symbol=symbol),
            name=name, showlegend=True,
        ))

    fig.update_layout(
        title=dict(
            text=title or case.get("title", "Medication Timeline"),
            x=0.02, xanchor="left", font=dict(size=14, color="#2c3e50"),
        ),
        height=height, width=width,
        margin=dict(l=20, r=20, t=60, b=50),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1.0, font=dict(size=10)),
        hovermode="closest",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Worked example: a hypertension patient on ACE-i → cough → switch to ARB
# (the classic ADR teaching case)
# ─────────────────────────────────────────────────────────────────────────────
HYPERTENSION_ACE_I_TO_ARB_CASE = {
    "id":   "htn_ace_i_adr",
    "title": "58M, HTN, ACE-i → dry cough → switch to ARB (classic ADR)",
    "medication_timeline": [
        {
            "name": "Lisinopril", "brand": "Zestril",
            "rxnorm_code": "29046", "dose": "10 mg", "route": "PO",
            "frequency": "once daily", "indication": "hypertension",
            "started_at": -86400 * 30,    # started 30 days before admission
            "stopped_at": 86400 * 21,     # stopped day 21
            "status": "stopped",
            "stop_reason": "ADR (dry cough)",
            "notes": "Initial antihypertensive. ACE-inhibitor.",
        },
        {
            "name": "Amlodipine", "brand": "Norvasc",
            "rxnorm_code": "197361", "dose": "5 mg", "route": "PO",
            "frequency": "once daily", "indication": "hypertension",
            "started_at": -86400 * 90,    # chronic
            "stopped_at": None,            # ongoing
            "status": "active",
            "notes": "Calcium-channel blocker. Added to lisinopril for BP control.",
        },
        {
            "name": "Hydrochlorothiazide", "brand": "Microzide",
            "rxnorm_code": "5487", "dose": "12.5 mg", "route": "PO",
            "frequency": "once daily", "indication": "hypertension",
            "started_at": -86400 * 60,    # chronic
            "stopped_at": None,            # ongoing
            "status": "active",
            "notes": "Thiazide diuretic. Added as 3rd-line agent.",
        },
        {
            "name": "Losartan", "brand": "Cozaar",
            "rxnorm_code": "52175", "dose": "50 mg", "route": "PO",
            "frequency": "once daily", "indication": "hypertension (post-ACE-i cough)",
            "started_at": 86400 * 21,      # day 21
            "stopped_at": None,            # ongoing
            "status": "active",
            "notes": "ARB. Switched from lisinopril after cough developed.",
        },
        {
            "name": "Atorvastatin", "brand": "Lipitor",
            "rxnorm_code": "83367", "dose": "40 mg", "route": "PO",
            "frequency": "once daily (evening)", "indication": "primary prevention (high CV risk)",
            "started_at": 0,                # admission
            "stopped_at": None,            # ongoing
            "status": "active",
            "notes": "Statin. Started during admission after risk re-assessment.",
        },
    ],
    "drug_reactions": [
        {
            "drug": "Lisinopril",
            "reaction": "dry, non-productive cough",
            "onset": 86400 * 14,            # day 14
            "severity": "moderate",
            "causality": "probable",
            "action_taken": "drug stopped, switched to ARB (losartan)",
            "lab_evidence": "n/a (clinical diagnosis)",
            "notes": "Classic ACE-inhibitor cough. Bradykinin accumulation. "
                     "Onset typically within weeks to months. Resolves within "
                     "days of discontinuation. ARBs are the standard alternative.",
        },
    ],
    "lab_monitoring": [
        {
            "drug": "Lisinopril", "test": "Creatinine", "value": "1.0 mg/dL",
            "at": 0, "interpretation": "normal",
            "notes": "Baseline before ACE-i was already in place.",
        },
        {
            "drug": "Lisinopril", "test": "Potassium", "value": "4.2 mEq/L",
            "at": 0, "interpretation": "normal",
            "notes": "Baseline K+ on ACE-i.",
        },
        {
            "drug": "Lisinopril", "test": "Creatinine", "value": "1.1 mg/dL",
            "at": 86400 * 10, "interpretation": "stable",
            "notes": "Routine monitoring while on ACE-i. Mild rise is expected.",
        },
        {
            "drug": "Losartan", "test": "Creatinine", "value": "1.1 mg/dL",
            "at": 86400 * 28, "interpretation": "stable",
            "notes": "1 week after switch to ARB. Renal function preserved.",
        },
        {
            "drug": "Losartan", "test": "Potassium", "value": "4.4 mEq/L",
            "at": 86400 * 28, "interpretation": "normal",
            "notes": "K+ stable on ARB.",
        },
    ],
    "adherence_notes": [
        {
            "drug": "Hydrochlorothiazide",
            "at": 86400 * 5,
            "issue": "Patient reports occasionally forgetting morning dose due to nocturia",
            "action": "Switched to bedtime dosing, nocturia monitored",
            "notes": "Dose timing adjusted to minimize sleep disruption.",
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up timeline events for this case (the multi-month evolution)
# ─────────────────────────────────────────────────────────────────────────────
HYPERTENSION_FOLLOW_UPS = [
    {
        "at_days": 14,
        "type": "complication",
        "headline": "Dry cough develops",
        "data": {
            "reaction": "dry, non-productive cough",
            "onset_day": 14,
            "severity": "moderate",
            "drug": "Lisinopril",
            "causality": "probable (WHO-UMC)",
        },
        "prompt": (
            "Patient on lisinopril 10 mg daily for 2 weeks develops a dry, "
            "non-productive cough. No fever, no URI symptoms, CXR normal. "
            "What is your differential, and how do you manage?"
        ),
        "teaching_note": (
            "ACE-inhibitor cough occurs in 5-20% of patients, due to bradykinin "
            "accumulation. It is a class effect. The standard management is to "
            "switch to an ARB (losartan, valsartan, etc.) — same RAAS mechanism, "
            "no bradykinin effect. The cough typically resolves within 1-4 days "
            "of stopping the ACE-i. Don't forget to ask about ACE-i cough in any "
            "patient on lisinopril/enalapril/ramipril who presents with cough."
        ),
    },
    {
        "at_days": 21,
        "type": "course_correction",
        "headline": "Switched to ARB (losartan 50 mg)",
        "data": {
            "drug_stopped": "Lisinopril",
            "drug_started": "Losartan 50 mg PO daily",
            "reason": "ADR (cough)",
        },
        "prompt": (
            "How do you counsel the patient about the switch? What monitoring "
            "is needed after starting the ARB?"
        ),
        "teaching_note": (
            "Counsel the patient that the cough should resolve within days. "
            "Monitor BP, K+, and creatinine 1-2 weeks after starting the ARB "
            "(same monitoring as for ACE-i). ARBs are teratogenic — confirm "
            "not pregnant in women of childbearing age. ARBs do not cause the "
            "bradykinin-mediated cough but share the same renal/K+ risks as ACE-i."
        ),
    },
    {
        "at_days": 35,
        "type": "follow_up_visit",
        "headline": "Cough resolved; BP 128/82 on losartan + amlodipine + HCTZ",
        "data": {
            "vitals": {"bp": "128/82", "hr": 72, "weight": "82 kg"},
            "labs": {"creatinine": "1.1", "potassium": "4.4"},
            "symptoms": "Cough completely resolved. No orthostasis.",
        },
        "prompt": (
            "What is your long-term plan? Does the patient need any additional "
            "workup for secondary causes of hypertension?"
        ),
        "teaching_note": (
            "Triple therapy (ACE-i/ARB + CCB + thiazide) achieves BP control "
            "in most patients with essential HTN. Secondary HTN workup is "
            "indicated if: onset < 30 or > 55, resistant HTN (uncontrolled on "
            "3 agents), abrupt worsening, hypokalemia, abdominal bruit, etc. "
            "This patient doesn't meet criteria."
        ),
    },
    {
        "at_days": 90,
        "type": "long_term",
        "headline": "3-month follow-up — well controlled, no recurrence",
        "data": {
            "vitals": {"bp": "124/78", "hr": 70},
            "labs": {"creatinine": "1.1", "potassium": "4.5", "hba1c": "5.9"},
            "adherence": "good — patient verbalized understanding of all 3 medications",
        },
        "prompt": (
            "What are the long-term monitoring goals for this patient? "
            "When would you step down or step up therapy?"
        ),
        "teaching_note": (
            "Long-term HTN monitoring: BP at every visit, BMP every 6-12 months, "
            "lipid panel annually, urine albumin/creatinine ratio annually "
            "(end-organ damage screening). If BP well-controlled for 1+ year, "
            "step-down can be considered (reduce one agent at a time, monitor "
            "closely for rebound HTN). Patient education: DASH diet, exercise, "
            "weight management, sodium restriction, smoking cessation if applicable."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Demo runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    case = HYPERTENSION_ACE_I_TO_ARB_CASE
    print(f"Case: {case['title']}")
    print(f"  {len(case['medication_timeline'])} medications tracked")
    print(f"  {len(case['drug_reactions'])} adverse reactions")
    print(f"  {len(case['lab_monitoring'])} monitoring labs")
    print(f"  {len(case['adherence_notes'])} adherence notes")
    print(f"  {len(HYPERTENSION_FOLLOW_UPS)} follow-up timeline updates")
    print()
    print("WHO-UMC causality for the lisinopril cough:", case['drug_reactions'][0]['causality'])
    print("  ->", WHO_UMC_CRITERIA[case['drug_reactions'][0]['causality']])
    print()

    # Render the medication timeline
    try:
        fig = render_medication_timeline(case, start_age_years=58.0,
                                         width=2400, height=700)
        fig.write_image("/workspace/medication_timeline_demo.png",
                        width=2400, height=700, scale=1)
        print("Wrote /workspace/medication_timeline_demo.png")
    except Exception as e:
        print(f"(rendering skipped: {e})")

    # Save the case as JSON so it can be loaded by the existing system
    os.makedirs("/workspace/cases", exist_ok=True)
    full_case = {
        **case,
        "vignette": (
            "58-year-old male with hypertension on lisinopril 10 mg + amlodipine "
            "5 mg + hydrochlorothiazide 12.5 mg. BP 152/96. He mentions a dry "
            "cough that started 2 weeks ago. He denies fever, URI symptoms, "
            "or heartburn. CXR is normal. The cough is interfering with his "
            "sleep and he wants something done about it."
        ),
        "questions_for_student": [
            "What is your differential for the cough?",
            "Which medication is most likely the cause, and why?",
            "What is the mechanism of this adverse effect?",
            "How do you manage it? What monitoring is needed after the change?",
        ],
        "critical_steps": [
            {"label": "Recognize ACE-i cough", "target_by": 300, "done_at": None},
            {"label": "Stop lisinopril", "target_by": 600, "done_at": None},
            {"label": "Start ARB (losartan)", "target_by": 900, "done_at": None},
            {"label": "Order BMP 1-2 weeks post-switch", "target_by": 1500, "done_at": None},
        ],
        "critical_node_labels": [
            "Recognize ACE-i cough",
            "Stop lisinopril",
            "Start ARB (losartan)",
            "Order BMP 1-2 weeks post-switch",
        ],
        "vitals_initial": {"hr": 78, "sbp": 152, "dbp": 96, "spo2": 98, "rr": 14},
        "expert_nodes": [
            {"id": "n_001", "type": "history",    "label": "Ask about cough onset, character, triggers",
             "rationale": "Onset 2 weeks after starting/increasing ACE-i is classic"},
            {"id": "n_002", "type": "history",    "label": "Review current medications",
             "rationale": "Identify the suspected drug"},
            {"id": "n_003", "type": "exam",       "label": "Lung exam + vitals",
             "rationale": "Rule out other causes; document baseline"},
            {"id": "n_004", "type": "decision",   "label": "Recognize ACE-i cough",
             "rationale": "Dry, non-productive, no fever/URI, CXR normal, temporal link to ACE-i"},
            {"id": "n_005", "type": "treatment",  "label": "Stop lisinopril",
             "rationale": "Definitive management of suspected ADR"},
            {"id": "n_006", "type": "treatment",  "label": "Start ARB (losartan 50 mg PO daily)",
             "rationale": "Same RAAS mechanism, no bradykinin effect, equivalent BP control"},
            {"id": "n_007", "type": "lab",        "label": "Order BMP in 1-2 weeks",
             "rationale": "Monitor K+ and Cr after ARB initiation"},
            {"id": "n_008", "type": "disposition","label": "Follow up in 2-4 weeks",
             "rationale": "Confirm cough resolution and BP control"},
        ],
        "expert_edges": [
            {"from": "n_001", "to": "n_002"},
            {"from": "n_002", "to": "n_003"},
            {"from": "n_003", "to": "n_004"},
            {"from": "n_004", "to": "n_005"},
            {"from": "n_005", "to": "n_006"},
            {"from": "n_006", "to": "n_007"},
            {"from": "n_007", "to": "n_008"},
        ],
        "follow_up_timeline": HYPERTENSION_FOLLOW_UPS,
        "case_source": {
            "type": "composite",
            "origin": "Classic teaching case — ACE-i cough is one of the most common ADRs in primary care and cardiology",
            "deidentified": True,
            "consent_note": "Composite teaching case",
        },
        "follow_up": {
            "actual_path": "Cough resolved within 1 week of stopping lisinopril. BP 128/82 on losartan + amlodipine + HCTZ. Creatinine and K+ stable at 3-month follow-up.",
            "complications": "ACE-inhibitor cough (probable WHO-UMC causality)",
            "discharge_outcome": "Continued on losartan 50 mg, amlodipine 5 mg, HCTZ 12.5 mg, atorvastatin 40 mg",
            "long_term": "BP well-controlled at 3-month follow-up (124/78). No recurrence of cough.",
            "teaching_pearl": "ACE-i cough is a class effect due to bradykinin accumulation. Always ask about ACE-i use in any patient presenting with chronic cough. ARBs are the standard alternative — same RAAS effect, no cough.",
        },
    }
    out_path = "/workspace/cases/htn_ace_i_adr.json"
    with open(out_path, "w") as f:
        json.dump(full_case, f, indent=2, default=str)
    print(f"Wrote {out_path}")
    print()
    print("To load this in the Streamlit UI:")
    print("  - Move the case file into latest/cases/")
    print("  - Restart the app, case will appear in the picker")
    print()
    print("To extend the follow-up system with drug-specific update types,")
    print("add these to reasoning_followup.py's UPDATE_TYPES list:")
    for t, label, desc in DRUG_UPDATE_TYPES:
        print(f"  ({t!r:20s}, {label!r:35s}, {desc!r})")
