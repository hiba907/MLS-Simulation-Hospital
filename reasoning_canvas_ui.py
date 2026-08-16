"""Reasoning Canvas — Streamlit UI.

A "whiteboard" where the student BUILDS the diagnostic workup as a graph of
nodes connected by arrows. No Q-bank, no flowchart picker — the student
authors the reasoning.

Layout (3 columns):
  ┌─Left (palette)─┬──────Center (canvas)──────┬─Right (vitals + case)──┐
  │ Node type       │   Visual tree             │  Case vignette         │
  │ buttons         │   (SVG with nodes+edges)  │  Vitals panel          │
  │ [+ History]     │                           │  Timer                 │
  │ [+ Exam]        │   Click nodes to:         │  Critical steps        │
  │ [+ Lab]         │   • Edit label/rationale   │  checklist             │
  │ [+ Imaging]     │   • Connect to another    │                        │
  │ [+ Decision]    │   • Delete                │                        │
  │ [+ Treatment]   │                           │                        │
  │ [+ Diagnosis]   │   [Finish Case → X-Ray]   │                        │
  │ [+ Disposition] │                           │                        │
  └─────────────────┴───────────────────────────┴────────────────────────┘

The X-Ray (after finish):
  - Total score + 3 component scores
  - Side-by-side: student's tree | expert's tree | diff
  - Missed/extra nodes highlighted
  - "Steal this tree" — show a high-scoring tree from another resident (v2)

This is a no-React, no-JS, no-streamlit-flow implementation. We render the
graph as inline SVG inside Streamlit. It feels like a whiteboard but ships
fast. The student interacts with controls (buttons) rather than dragging.
"""
import streamlit as st
import time
import math
import html
import json
import datetime
from typing import Dict, List, Optional, Tuple

import reasoning_canvas as rc
from reasoning_canvas import (
    NODE_TYPES, NODE_TYPE_ICONS, NODE_TYPE_LABELS,
    empty_tree, add_node, add_edge, remove_node, score_tree,
    VitalsSim, get_case, DEFAULT_TIME_BUDGET,
)
from shared_session import (
    dept_header, load_student, save_student, record_session,
)
from reasoning_followup import UPDATE_TYPES, UPDATE_TYPE_LABELS  # for the follow-up authoring UI


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
_CANVAS_CSS = """
<style>
.rc-canvas-wrap {
    background: #1a1a1d; border: 1px solid #333; border-radius: 10px;
    padding: 1rem; min-height: 500px; position: relative;
    overflow-x: auto;
}
.rc-node-card {
    display: inline-block; min-width: 180px; max-width: 220px;
    background: #2a2a30; border: 2px solid #555; border-radius: 8px;
    padding: 0.5rem 0.7rem; margin: 6px; vertical-align: top;
    font-size: 0.82rem; color: #fbfbff; cursor: pointer;
    transition: border-color 0.15s;
}
.rc-node-card:hover { border-color: #ffd400; }
.rc-node-card.selected { border-color: #ffd400; background: #3a3a20; }
.rc-node-card.critical { border-color: #ff4444; }
.rc-node-card .type {
    font-size: 0.7rem; color: #ffd400; text-transform: uppercase;
    letter-spacing: 0.05em; font-weight: 700;
}
.rc-node-card .label { font-weight: 600; margin-top: 2px; }
.rc-node-card .ratio { color: #909094; font-size: 0.72rem; margin-top: 3px;
                      font-style: italic; }
.rc-palette button { width: 100%; margin: 2px 0; text-align: left; }
.rc-vitals {
    background: #2a2a30; border: 1px solid #333; border-radius: 8px;
    padding: 0.8rem; color: #fbfbff;
}
.rc-vitals .vital-row {
    display: flex; justify-content: space-between;
    border-bottom: 1px solid #1a1a1d; padding: 4px 0; font-size: 0.9rem;
}
.rc-vitals .vital-row.critical { color: #ff6b6b; font-weight: 700; }
.rc-vitals .vital-row.warn { color: #ffa500; }
.rc-timer { font-size: 1.4rem; font-weight: 700; color: #ffd400;
            text-align: center; margin: 0.5rem 0; }
.rc-timer.critical { color: #ff4444; animation: pulse 1s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
.rc-xray-matched  { border-left: 4px solid #4caf50; padding-left: 8px;
                    background: #1a3a1a; margin: 4px 0; }
.rc-xray-missed   { border-left: 4px solid #ff9800; padding-left: 8px;
                    background: #3a2a1a; margin: 4px 0; }
.rc-xray-extra    { border-left: 4px solid #9c27b0; padding-left: 8px;
                    background: #2a1a3a; margin: 4px 0; }
.rc-legend { display: flex; gap: 1rem; font-size: 0.78rem;
             color: #909094; margin: 0.5rem 0; }
.rc-legend .sw { display: inline-block; width: 12px; height: 12px;
                 margin-right: 4px; vertical-align: middle; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# SVG graph renderer
# ─────────────────────────────────────────────────────────────────────────────
def _compute_layout(tree: dict) -> Dict[str, Tuple[float, float]]:
    """Simple layered layout: assign x by topo order index, y by row.

    Returns dict: node_id -> (x, y) in pixel coords inside an SVG viewport.
    """
    if not tree["nodes"]:
        return {}
    try:
        topo = []
        from reasoning_canvas import topological_order
        topo = topological_order(tree)
    except ValueError:
        # Cycle: fall back to insertion order
        topo = [n["id"] for n in tree["nodes"]]

    # Group nodes into layers by longest-path-from-source
    incoming = {nid: 0 for nid in topo}
    for e in tree["edges"]:
        incoming[e["to"]] = incoming.get(e["to"], 0) + 1
    layer_of = {}
    layers: List[List[str]] = []
    remaining = set(topo)
    while remaining:
        layer = [nid for nid in remaining if incoming.get(nid, 0) == 0]
        if not layer:
            # Cycle residue — dump in a new layer
            layer = [next(iter(remaining))]
        for nid in layer:
            layer_of[nid] = len(layers)
            remaining.discard(nid)
            for e in tree["edges"]:
                if e["from"] == nid and e["to"] in remaining:
                    incoming[e["to"]] -= 1
        layers.append(sorted(layer))

    # Position
    NODE_W, NODE_H = 200, 90
    H_GAP, V_GAP = 80, 30
    max_per_layer = max(len(L) for L in layers)
    canvas_w = max(800, max_per_layer * (NODE_W + V_GAP) + V_GAP)
    canvas_h = (len(layers) + 1) * (NODE_H + H_GAP) + H_GAP

    positions = {}
    for i, layer in enumerate(layers):
        y = H_GAP + i * (NODE_H + H_GAP)
        n = len(layer)
        # Center the layer horizontally
        total_w = n * NODE_W + (n - 1) * V_GAP
        x_start = (canvas_w - total_w) / 2
        for j, nid in enumerate(layer):
            x = x_start + j * (NODE_W + V_GAP)
            positions[nid] = (x, y)
    return positions, canvas_w, canvas_h, NODE_W, NODE_H


def render_tree_svg(tree: dict, selected_id: Optional[str] = None,
                    highlight: Optional[Dict[str, str]] = None) -> str:
    """Render the tree as inline SVG. `highlight` is node_id -> color class
    ('matched' / 'missed' / 'extra')."""
    if not tree["nodes"]:
        return ('<div style="text-align:center;color:#666;padding:2rem">'
                'Your tree is empty. Use the palette on the left to add nodes.'
                '</div>')
    positions, cw, ch, nw, nh = _compute_layout(tree)
    nodes_by_id = {n["id"]: n for n in tree["nodes"]}

    parts = [f'<svg width="{cw}" height="{ch}" '
             'style="background:#1a1a1d;border-radius:8px" '
             'xmlns="http://www.w3.org/2000/svg">']

    # Defs: arrowhead marker
    parts.append(
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#ffd400"/></marker>'
        '<marker id="arr-dim" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#666"/></marker>'
        '</defs>'
    )

    # Edges first (so they go under nodes)
    for e in tree["edges"]:
        a, b = e["from"], e["to"]
        if a not in positions or b not in positions:
            continue
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        # Start/end at the box edges, not centers
        sx, sy = x1 + nw / 2, y1 + nh
        ex, ey = x2 + nw / 2, y2
        # Curved path
        midy = (sy + ey) / 2
        path = (f'M {sx} {sy} C {sx} {midy}, {ex} {midy}, {ex} {ey}')
        parts.append(
            f'<path d="{path}" stroke="#ffd400" stroke-width="1.5" fill="none" '
            f'marker-end="url(#arr)" opacity="0.7"/>'
        )

    # Nodes
    for nid, (x, y) in positions.items():
        n = nodes_by_id[nid]
        sel = " selected" if nid == selected_id else ""
        hl = (highlight or {}).get(nid, "")
        color = {"matched": "#4caf50", "missed": "#ff9800",
                 "extra": "#9c27b0"}.get(hl, "#555")
        icon = NODE_TYPE_ICONS.get(n["type"], "•")
        type_label = NODE_TYPE_LABELS.get(n["type"], n["type"])
        # Truncate long labels
        label = n["label"] if len(n["label"]) < 35 else n["label"][:32] + "…"
        rationale = n.get("rationale", "")
        rationale_short = (rationale if len(rationale) < 60
                          else rationale[:57] + "…")
        parts.append(
            f'<g class="rc-svg-node" data-id="{nid}">'
            f'<rect x="{x}" y="{y}" width="{nw}" height="{nh}" rx="8" '
            f'fill="#2a2a30" stroke="{color}" stroke-width="2"/>'
            f'<text x="{x+10}" y="{y+18}" font-size="11" fill="#ffd400" '
            f'font-weight="700">{icon} {html.escape(type_label)}</text>'
            f'<text x="{x+10}" y="{y+38}" font-size="13" fill="#fbfbff" '
            f'font-weight="600">{html.escape(label)}</text>'
            f'<text x="{x+10}" y="{y+58}" font-size="10" fill="#909094" '
            f'font-style="italic">{html.escape(rationale_short)}</text>'
            f'<text x="{x+10}" y="{y+78}" font-size="9" fill="#666">'
            f't+{int(n.get("t_added", 0))}s</text>'
            f'</g>'
        )
    parts.append('</svg>')
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Vitals panel
# ─────────────────────────────────────────────────────────────────────────────
def render_vitals(sim_snap: dict, baseline: dict) -> str:
    """Return HTML for the vitals panel."""
    def klass(key, val):
        if key == "hr" and val > 120: return "critical"
        if key == "hr" and val > 100: return "warn"
        if key == "sbp" and val < 90:  return "critical"
        if key == "sbp" and val < 100: return "warn"
        if key == "spo2" and val < 90: return "critical"
        if key == "spo2" and val < 94: return "warn"
        return ""

    def fmt(key, val):
        if key == "hr":  return f"{val:.0f} bpm"
        if key == "sbp": return f"{val:.0f} mmHg"
        if key == "dbp": return f"{val:.0f} mmHg"
        if key == "spo2":return f"{val:.0f}%"
        if key == "rr":  return f"{val:.0f}/min"
        return f"{val}"

    rows = []
    for key, label in [("hr", "HR"), ("sbp", "SBP"), ("dbp", "DBP"),
                        ("spo2", "SpO₂"), ("rr", "RR")]:
        v = sim_snap.get(key, baseline.get(key, 0))
        rows.append(
            f'<div class="vital-row {klass(key, v)}">'
            f'<span>{label}</span><span>{fmt(key, v)}</span></div>'
        )
    t = sim_snap.get("t", 0)
    return (
        '<div class="rc-vitals">'
        f'<div style="text-align:center;font-weight:700;color:#ffd400;'
        f'margin-bottom:6px">VITALS — t+{int(t)}s</div>'
        + "".join(rows) +
        '</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────
def _timeline_order(tree: dict) -> list:
    """Nodes in the order the student wrote them (add_node already appends
    in order, so this is just the list — kept as a named helper in case the
    ordering rule ever needs to change, e.g. to a true topological sort once
    branching gets more complex)."""
    return tree["nodes"]


def _norm(text: str) -> str:
    """Lowercase, strip punctuation — used to match a written step against a
    critical-step label so 'Mark done' can trigger automatically."""
    return "".join(c.lower() for c in text if c.isalnum() or c.isspace()).strip()


def render(sid: str):
    """Streamlit entry point for the Reasoning Canvas department."""
    st.markdown(_CANVAS_CSS, unsafe_allow_html=True)
    is_rereason = st.session_state.get("rc_rereason", False)
    title_suffix = " — Re-reasoning with new data" if is_rereason else ""
    dept_header("Reasoning Canvas — Clinical Decision Trees" + title_suffix,
                "🧠", sid)

    # ── Case selection (if not yet started) ─────────────────────────────────
    if "rc_case_id" not in st.session_state or \
       st.session_state.get("rc_finished"):
        _render_case_select()
        return

    # ── Active case ─────────────────────────────────────────────────────────
    case_id = st.session_state["rc_case_id"]
    case = get_case(case_id)
    tree = st.session_state.setdefault("rc_tree", empty_tree())

    # Re-reasoning banner — show the new data and the resident's original tree
    if is_rereason:
        from reasoning_followup import load_responses
        rec = load_responses(case_id, sid)
        update = st.session_state.get("rc_rereason_update", {})
        st.markdown("---")
        st.markdown("### 🔁 Re-reasoning Mode")
        st.info(
            f"**New data (day {update.get('at_days', '?')}):** "
            f"{update.get('headline', '')}"
        )
        if update.get("data"):
            with st.expander("Show the new clinical data", expanded=False):
                st.json(update["data"])
        if update.get("prompt"):
            st.markdown(f"**Your task:** {update['prompt']}")
        if update.get("teaching_note"):
            with st.expander("Teaching note (revealed after you submit)", expanded=False):
                st.markdown(update["teaching_note"])
        # Show the original tree in a side panel so the resident can refer to it
        if rec.get("initial") and rec["initial"].get("tree"):
            with st.expander("📜 Your original reasoning (for reference)",
                             expanded=False):
                ot = rec["initial"]["tree"]
                st.caption(f"Submitted on day 0 • Score: "
                           f"{rec['initial'].get('score', 0):.0f}/100")
                # Render the original tree as a compact plan
                for i, n in enumerate(ot.get("nodes", [])):
                    icon = NODE_TYPE_ICONS.get(n.get("type", ""), "•")
                    st.markdown(f"{i+1}. {icon} **{n.get('label','')}** — "
                               f"_{n.get('rationale','')}_")
        st.markdown("---")

    # Initialize vitals sim once — but NOT in re-reasoning mode (no time pressure)
    if "rc_sim" not in st.session_state and not is_rereason:
        st.session_state["rc_sim"] = VitalsSim(
            initial=case["vitals_initial"],
            critical_steps=[dict(s) for s in case["critical_steps"]],
        )
        st.session_state["rc_t0"] = time.time()
    sim = st.session_state.get("rc_sim")
    elapsed = time.time() - st.session_state["rc_t0"]
    if sim is not None:
        sim.t = elapsed
        sim.tick(0)  # no dt advance — sim.t is set from real clock
        snap = sim.snapshot()
    else:
        # Re-reasoning mode: no sim, no time pressure
        snap = case["vitals_initial"]

    # ── Top bar: case title + timer ─────────────────────────────────────────
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.markdown(f"### {case['title']}")
        st.caption(f"{case['specialty']} • Difficulty: {case['difficulty']} • "
                   f"~{case['estimated_min']} min"
                   + (" • **🔁 Re-reasoning**" if is_rereason else ""))
    with c2:
        if is_rereason:
            st.markdown(
                '<div class="rc-timer" style="color:#4caf50">⏱ No time limit</div>',
                unsafe_allow_html=True)
        else:
            remaining = max(0, DEFAULT_TIME_BUDGET - int(elapsed))
            klass = "critical" if remaining < 60 else ""
            st.markdown(
                f'<div class="rc-timer {klass}">⏱ {remaining//60}:{remaining%60:02d}</div>',
                unsafe_allow_html=True)
    with c3:
        if st.button("🚪 End Case", type="secondary"):
            _finish_case(sid, case, tree, sim)
            return

    st.markdown("---")

    # ── 2-column layout: plan timeline (main) + case info (side) ────────────
    center, right = st.columns([3, 1.3])

    # ── RIGHT: vignette + vitals + critical steps ───────────────────────────
    with right:
        st.markdown("#### 📋 Case")
        with st.expander("Vignette", expanded=True):
            st.write(case["vignette"])
        with st.expander("Questions to consider", expanded=False):
            for q in case["questions_for_student"]:
                st.markdown(f"- {q}")
        if is_rereason:
            # In re-reasoning mode, show the new data instead of vitals
            update = st.session_state.get("rc_rereason_update", {})
            st.markdown("#### 📋 New clinical data")
            st.warning(f"**{update.get('headline','')}**")
            if update.get("data"):
                st.json(update["data"])
            if update.get("teaching_note"):
                st.markdown("---")
                st.markdown("**💡 Teaching note (after you submit)**")
                st.caption(update["teaching_note"])
        else:
            st.markdown(render_vitals(snap, case["vitals_initial"]),
                        unsafe_allow_html=True)
            # Critical steps checklist
            st.markdown("#### 🚨 Critical Steps")
            for cs in sim.critical_steps:
                if cs["done_at"] is not None:
                    st.success(f"✓ {cs['label']} (t+{int(cs['done_at'])}s)")
                elif elapsed > cs["target_by"]:
                    st.error(f"⚠ OVERDUE: {cs['label']}")
                else:
                    remaining_cs = int(cs["target_by"] - elapsed)
                    st.warning(f"○ {cs['label']} (target: t+{cs['target_by']}s, "
                               f"{remaining_cs}s left)")

    # ── CENTER: your plan, written as a running clinical timeline ──────────
    with center:
        st.markdown("#### 🩺 Your plan")
        st.caption("Write it the way you'd think through it on the ward — "
                  "each step gets added to your running plan in order.")

        ordered = _timeline_order(tree)

        if not ordered:
            st.info("Nothing added yet — write your first step below.")
        for i, node in enumerate(ordered):
            icon = NODE_TYPE_ICONS.get(node["type"], "•")
            with st.container(border=True):
                tc1, tc2 = st.columns([5, 1])
                with tc1:
                    st.markdown(f"**{i+1}. {icon} {node['label']}**")
                    if node.get("rationale"):
                        st.caption(node["rationale"])
                with tc2:
                    if st.button("🗑", key=f"del_{node['id']}",
                                 help="Remove this step"):
                        remove_node(tree, node["id"])
                        st.rerun()
                # Mark as critical step if it matches one (synonym-aware,
                # same logic used in scoring)
                expert_by_label = {en["label"]: en for en in case["expert_tree"]["nodes"]}
                matching_cs = next(
                    (cs for cs in sim.critical_steps
                     if rc._best_similarity_with_synonyms(
                         node["label"], expert_by_label.get(cs["label"], {"label": cs["label"], "synonyms": []})
                     ) >= rc.MATCH_THRESHOLD),
                    None,
                )
                if matching_cs and matching_cs["done_at"] is None:
                    if st.button(f"✅ Mark done", key=f"mark_{node['id']}"):
                        sim.mark_done(matching_cs["label"])
                        st.success("Critical step completed!")
                        st.rerun()

        st.markdown("---")
        st.markdown("##### ✍️ Add your next step")

        type_cols = st.columns(len(NODE_TYPES))
        selected_type = st.session_state.get("rc_compose_type", NODE_TYPES[0][0])
        for i, (ntype, icon, label, desc) in enumerate(NODE_TYPES):
            with type_cols[i]:
                pressed = (ntype == selected_type)
                if st.button(f"{icon}\n{label}", key=f"type_{ntype}",
                             use_container_width=True,
                             type=("primary" if pressed else "secondary")):
                    st.session_state["rc_compose_type"] = ntype
                    st.rerun()

        step_label = st.text_input(
            "What's your next step?",
            key="rc_compose_label_widget",
            placeholder="e.g. '12-lead ECG', 'Ask about chest pain onset', 'Start aspirin 325mg'",
        )
        step_rationale = st.text_input(
            "Why? (optional, 1 line)",
            key="rc_compose_rationale_widget",
            placeholder="e.g. 'Rule out STEMI before troponin returns'",
        )

        # "Continue from" only matters once there's a fork to make — default
        # to the end of the plan so the common linear case takes zero clicks.
        continue_from = None
        if len(ordered) >= 1:
            branch_target = st.selectbox(
                "Continue from",
                options=["(end of plan)"] + [n["id"] for n in ordered],
                format_func=lambda nid: (
                    "(end of plan)" if nid == "(end of plan)"
                    else f"{ordered.index(next(n for n in ordered if n['id']==nid))+1}. "
                         f"{next(n for n in ordered if n['id']==nid)['label']}"
                ),
                index=0,
                key="rc_compose_continue_from",
                help="Leave as '(end of plan)' unless this is a decision fork "
                     "branching off an earlier step.",
            )
            continue_from = None if branch_target == "(end of plan)" else branch_target

        if st.button("➕ Add to plan", type="primary", use_container_width=True,
                     disabled=not step_label.strip()):
            n = add_node(tree, selected_type, step_label.strip(),
                        rationale=step_rationale.strip(), t_added=int(elapsed))
            parent_id = continue_from or (ordered[-1]["id"] if ordered else None)
            if parent_id:
                try:
                    add_edge(tree, parent_id, n["id"])
                except ValueError:
                    pass
            st.session_state["rc_tree"] = tree
            st.session_state["_compose_label"] = ""
            st.session_state["_compose_rationale"] = ""
            st.rerun()

        # ── Finish button ───────────────────────────────────────────────────
        st.markdown("---")
        cf1, cf2, cf3 = st.columns([1, 1, 2])
        with cf1:
            if st.button("🏁 Finish & See X-Ray", type="primary",
                         use_container_width=True):
                _finish_case(sid, case, tree, sim)
                return
        with cf2:
            if st.button("↺ Reset plan", use_container_width=True):
                st.session_state["rc_tree"] = empty_tree()
                st.rerun()
        with cf3:
            st.caption(f"Steps: {len(tree['nodes'])}")


def _render_case_select():
    """The case-picker screen shown when no case is active."""
    from reasoning_canvas import list_all_case_ids, get_case
    from reasoning_followup import load_responses, pending_updates, days_since, load_timeline

    top_l, top_r = st.columns([4, 1.4])
    with top_l:
        st.markdown("### Choose a case")
        st.caption("Each case is a clinical vignette. Build the workup as a "
                   "decision tree — see how your reasoning compares to the "
                   "expert's. Real cases evolve over time — check for "
                   "**new data** below.")
    with top_r:
        if st.button("🩺 Leader: Author a case", use_container_width=True):
            st.session_state["rc_mode"] = "author"
            st.rerun()

    nav_l, nav_m, nav_r = st.columns([1, 1, 4])
    with nav_l:
        if st.button("📊 Cohort view", use_container_width=True):
            st.session_state["rc_mode"] = "cohort_picker"
            st.rerun()
    with nav_m:
        if st.button("👥 Admin dashboard", use_container_width=True):
            st.session_state["rc_mode"] = "admin"
            st.rerun()

    if st.session_state.get("rc_mode") == "author":
        _render_case_author()
        return
    if st.session_state.get("rc_mode") == "admin":
        _render_admin_dashboard()
        return
    if st.session_state.get("rc_mode") == "cohort_picker":
        # Pick a case to view cohort stats for
        from reasoning_canvas import list_all_case_ids, get_case
        st.markdown("#### Pick a case to view cohort reasoning")
        for case_id in list_all_case_ids():
            try:
                case = get_case(case_id)
            except Exception:
                continue
            if st.button(f"📊 {case['title']}", key=f"cohort_{case_id}",
                         use_container_width=True):
                st.session_state["rc_mode"] = "cohort"
                st.session_state["rc_cohort_case"] = case_id
                st.rerun()
        if st.button("← Back"):
            st.session_state.pop("rc_mode", None)
            st.rerun()
        return
    if st.session_state.get("rc_mode") == "cohort":
        _render_cohort_view(st.session_state["rc_cohort_case"])
        return

    resident_id = st.session_state.get("student_id", "guest")
    case_ids = list_all_case_ids()
    cols = st.columns(3)
    for i, case_id in enumerate(case_ids):
        try:
            case = get_case(case_id)
        except Exception as e:
            with cols[i % 3]:
                st.error(f"Case '{case_id}' failed to load: {e}")
            continue

        # Check if the resident has done this case before and if there are
        # pending follow-up updates they haven't addressed yet.
        rec = load_responses(case_id, resident_id)
        has_initial = rec.get("initial") is not None
        timeline = load_timeline(case_id)
        pending = []
        if has_initial:
            days_in = days_since(rec["initial"].get("ts", ""))
            all_pending = pending_updates(case_id, days_in)
            # Only show updates the resident hasn't responded to yet
            responded_ids = set(rec.get("responses", {}).keys())
            pending = [u for u in all_pending if u["id"] not in responded_ids]

        with cols[i % 3]:
            full = case.get("expert_nodes") and case.get("expert_edges")
            icon = "✅" if full else "📝"
            src_type = case.get("case_source", {}).get("type", "composite")
            src_badge = {
                "real":      "🏥 Real de-identified case",
                "published": "📄 Published case report",
                "composite": "🧪 Composite teaching case",
                "stub":      "🚧 Stub",
            }.get(src_type, src_type)
            update_badge = ""
            if pending:
                update_badge = (f'<br>🔔 <b>{len(pending)} new update'
                                f'{"s" if len(pending) != 1 else ""}</b>')
            done_badge = ""
            if has_initial and not pending:
                score = rec["initial"].get("score", 0)
                done_badge = f'<br>✓ Done (score {score:.0f})'
            elif has_initial and pending:
                score = rec["initial"].get("score", 0)
                done_badge = f'<br>✓ Done ({score:.0f}) → re-reason'

            st.markdown(
                f'<div class="dept-card">'
                f'<div class="icon">🧠</div>'
                f'<div class="label">{case["title"]}</div>'
                f'<div class="desc">{case["specialty"]} • {case["difficulty"]} '
                f'• {icon}{"Full case" if full else "Stub"}<br>{src_badge}'
                f'{done_badge}{update_badge}</div>'
                f'</div>', unsafe_allow_html=True)

            if pending:
                # Show pending updates as a teaser
                for u in pending:
                    u_icon = {"lab_result": "🧪", "imaging": "🩻",
                              "complication": "⚠️", "follow_up_visit": "🏥",
                              "new_data": "📋", "discharge": "🚪",
                              "long_term": "📅",
                              "course_correction": "🔄"}.get(u.get("type"), "📌")
                    st.caption(f"  {u_icon} Day {u.get('at_days','?')}: "
                               f"{u.get('headline','')[:60]}")

            if pending and st.button(f"🔁 Re-reason with new data",
                                      key=f"rereason_{case_id}",
                                      use_container_width=True,
                                      type="primary"):
                _start_rereason(case_id, pending[0])
                st.rerun()
            elif st.button(f"{'Continue' if has_initial else 'Start'} case",
                           key=f"start_{case_id}",
                           use_container_width=True):
                # Reset state for new case
                for k in ["rc_tree", "rc_sim", "rc_t0", "rc_selected_node",
                          "rc_finished", "rc_xray_result", "rc_case_obj",
                          "rc_rereason", "rc_rereason_update"]:
                    st.session_state.pop(k, None)
                st.session_state["rc_case_id"] = case_id
                st.rerun()


def _start_rereason(case_id: str, update: dict):
    """Set up session state for a re-reasoning session on a pending update."""
    for k in ["rc_tree", "rc_sim", "rc_t0", "rc_selected_node",
              "rc_finished", "rc_xray_result", "rc_case_obj"]:
        st.session_state.pop(k, None)
    st.session_state["rc_case_id"] = case_id
    st.session_state["rc_rereason"] = True
    st.session_state["rc_rereason_update"] = update


def _render_cohort_view(case_id: str):
    """Show the cohort evolution view for a case — how the program's
    residents reasoned about it, and how their reasoning evolved with
    each follow-up update."""
    from reasoning_followup import (
        load_timeline, all_responses_for_case, cohort_evolution_stats,
    )
    case = get_case(case_id)
    timeline = load_timeline(case_id)
    all_recs = all_responses_for_case(case_id)
    stats = cohort_evolution_stats(case_id)

    st.markdown(f"### 📊 Cohort View — {case['title']}")
    st.caption(f"{case['specialty']} • {len(all_recs)} residents have done this case")

    if not all_recs:
        st.info("No residents have done this case yet. Once they do, "
                "you'll see how the cohort reasoned and how their "
                "reasoning evolved over time.")
        if st.button("← Back to case list"):
            st.session_state.pop("rc_mode", None)
            st.rerun()
        return

    # Top-level stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Residents", len(all_recs))
    with c2:
        initial_scores = [r["initial"]["score"] for r in all_recs
                          if r.get("initial")]
        avg_initial = (sum(initial_scores) / len(initial_scores)
                       if initial_scores else 0)
        st.metric("Avg initial score", f"{avg_initial:.0f}/100")
    with c3:
        n_responded = sum(1 for r in all_recs if r.get("responses"))
        st.metric("Re-reasoned at least once", n_responded)
    with c4:
        st.metric("Follow-up updates", len(timeline))

    # Timeline of updates with cohort performance
    if timeline:
        st.markdown("#### Update timeline & cohort performance")
        for u in timeline:
            uid = u["id"]
            ustats = stats["per_update_stats"].get(uid, {})
            n_resp = ustats.get("n_responded", 0)
            avg = ustats.get("avg_score")
            avg_str = f"{avg:.0f}/100" if avg is not None else "—"
            u_icon = {"lab_result": "🧪", "imaging": "🩻",
                      "complication": "⚠️", "follow_up_visit": "🏥",
                      "new_data": "📋", "discharge": "🚪",
                      "long_term": "📅",
                      "course_correction": "🔄"}.get(u.get("type"), "📌")
            with st.expander(f"{u_icon} Day {u.get('at_days','?')}: "
                            f"{u.get('headline','')} — "
                            f"{n_resp} resident{'s' if n_resp != 1 else ''} responded "
                            f"(avg {avg_str})"):
                st.markdown(f"**Type:** {u.get('type','?')}")
                st.markdown(f"**Prompt to residents:** {u.get('prompt','')}")
                if u.get("teaching_note"):
                    st.success(f"💡 Teaching note: {u['teaching_note']}")
                if u.get("data"):
                    with st.expander("New clinical data", expanded=False):
                        st.json(u["data"])
                # Per-update aggregate reasoning shifts
                added_for_update = []
                dropped_for_update = []
                from collections import Counter
                ca = Counter()
                cd = Counter()
                for r in all_recs:
                    resp = r.get("responses", {}).get(uid)
                    if not resp:
                        continue
                    from reasoning_followup import compute_delta
                    d = compute_delta(r["initial"]["tree"], resp["tree"])
                    for a in d["added_nodes"]:
                        ca[a] += 1
                    for dr in d["removed_nodes"]:
                        cd[dr] += 1
                if ca:
                    st.markdown("**Most added:**")
                    for label, c in ca.most_common(5):
                        st.markdown(f"- {label} *(+{c})*")
                if cd:
                    st.markdown("**Most dropped:**")
                    for label, c in cd.most_common(5):
                        st.markdown(f"- {label} *(-{c})*")
    else:
        st.info("This case has no follow-up updates yet. Add one with the "
                "Leader panel below.")

    if st.button("← Back to case list"):
        st.session_state.pop("rc_mode", None)
        st.rerun()


def _render_admin_dashboard():
    """Chief-resident / program-director view: who's done what, who's
    behind, who needs attention."""
    from reasoning_followup import (
        load_responses, all_responses_for_case, load_timeline,
    )
    from reasoning_canvas import list_all_case_ids, get_case

    st.markdown("### 👥 Admin Dashboard — Reasoning Canvas")
    st.caption("Track engagement, identify residents who need support, and "
               "spot the case-level patterns your program should teach next.")

    # Tabs: by case | by resident | global
    tab_case, tab_resident, tab_global = st.tabs(
        ["📋 By case", "👤 By resident", "🌍 Program-wide"])

    with tab_case:
        st.markdown("#### Per-case engagement")
        for case_id in list_all_case_ids():
            try:
                case = get_case(case_id)
            except Exception:
                continue
            recs = all_responses_for_case(case_id)
            n_done = len(recs)
            n_rereasoned = sum(1 for r in recs if r.get("responses"))
            n_pending = 0
            for r in recs:
                if not r.get("initial"):
                    continue
                from reasoning_followup import days_since, pending_updates
                d = days_since(r["initial"].get("ts", ""))
                p = pending_updates(case_id, d)
                responded = set(r.get("responses", {}).keys())
                p = [u for u in p if u["id"] not in responded]
                n_pending += len(p)
            st.markdown(
                f"**{case['title']}** — {n_done} done, {n_rereasoned} re-reasoned, "
                f"🔔 {n_pending} pending updates across cohort")

    with tab_resident:
        # Aggregate per-resident from student records
        from shared_session import all_students
        students = all_students()
        rows = []
        for s in students:
            sid = s["id"]
            name = s.get("name", "?")
            rc_score = s.get("best", {}).get("reasoning_canvas", 0)
            n_attempts = len(s.get("scores", {}).get("reasoning_canvas", []))
            # How many cases has this resident done at all
            cases_done = 0
            cases_rereasoned = 0
            for case_id in list_all_case_ids():
                rec = load_responses(case_id, sid)
                if rec.get("initial"):
                    cases_done += 1
                if rec.get("responses"):
                    cases_rereasoned += 1
            rows.append({
                "Resident": name, "ID": sid,
                "Best score": rc_score,
                "Attempts": n_attempts,
                "Cases done": cases_done,
                "Re-reasoned": cases_rereasoned,
            })
        if rows:
            # Simple HTML table
            st.markdown(
                "<table style='width:100%;font-size:.85rem;color:#fbfbff'>"
                "<tr style='background:#2a2a30'>"
                "<th>Resident</th><th>Best</th><th>Attempts</th>"
                "<th>Cases done</th><th>Re-reasoned</th></tr>",
                unsafe_allow_html=True)
            for r in rows:
                st.markdown(
                    f"<tr><td>{html.escape(str(r['Resident']))} "
                    f"<small style='color:#666'>({r['ID']})</small></td>"
                    f"<td>{r['Best score']:.0f}</td>"
                    f"<td>{r['Attempts']}</td>"
                    f"<td>{r['Cases done']}</td>"
                    f"<td>{r['Re-reasoned']}</td></tr>",
                    unsafe_allow_html=True)
            st.markdown("</table>", unsafe_allow_html=True)
        else:
            st.info("No resident data yet.")

    with tab_global:
        st.markdown("#### Program-wide stats")
        from reasoning_canvas import list_all_case_ids
        all_cases = list_all_case_ids()
        total_responses = 0
        total_rereason = 0
        for case_id in all_cases:
            recs = all_responses_for_case(case_id)
            total_responses += len(recs)
            total_rereason += sum(1 for r in recs if r.get("responses"))
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.metric("Cases in library", len(all_cases))
        with gc2:
            st.metric("Total attempts", total_responses)
        with gc3:
            st.metric("Re-reasoning responses", total_rereason)

    if st.button("← Back to case list"):
        st.session_state.pop("rc_mode", None)
        st.rerun()


def _render_case_author():
    """Leader-only screen: write a full real case (vignette through
    follow-up), save it hidden, students only ever see the vignette until
    they finish their tree.

    NOTE ON ACCESS CONTROL: this is a soft gate (a shared passphrase), not
    real authentication. It stops a casual student from wandering in, but it
    is NOT secure against someone who wants in. Before using this with real
    patient-derived cases, wire this into whatever real login/role system
    `shared_session.py` already uses for students, so only actual
    instructor accounts can reach this screen.
    """
    import reasoning_canvas as rc

    st.markdown("### 🩺 Author a case (Leader)")
    if st.button("← Back to case list"):
        st.session_state.pop("rc_mode", None)
        st.session_state.pop("rc_author_unlocked", None)
        st.rerun()

    if not st.session_state.get("rc_author_unlocked"):
        st.warning("This is a soft access gate, not real security. See the "
                   "code comment in `_render_case_author` before real use.")
        pw = st.text_input("Leader passphrase", type="password")
        if st.button("Unlock"):
            # TODO: replace with a real leader/instructor role check.
            if pw == "mls-leader-2026":
                st.session_state["rc_author_unlocked"] = True
                st.rerun()
            else:
                st.error("Incorrect passphrase.")
        return

    steps = st.session_state.setdefault("author_steps", [])

    st.markdown("#### Case details")
    c1, c2, c3 = st.columns(3)
    with c1:
        title = st.text_input("Title", key="author_title",
                              placeholder="e.g. 58M crushing chest pain")
        specialty = st.text_input("Specialty", key="author_specialty",
                                  placeholder="e.g. Internal Medicine")
    with c2:
        difficulty = st.selectbox("Difficulty", ["Intern", "Resident", "Senior"],
                                  key="author_difficulty")
        est_min = st.number_input("Est. minutes", min_value=1, max_value=30,
                                  value=8, key="author_est_min")
    with c3:
        source_type = st.selectbox(
            "Case source",
            ["real", "published", "composite"],
            format_func=lambda t: {
                "real": "🏥 Real hospital chart (needs your own de-identification + consent/ethics)",
                "published": "📄 Published case report (PubMed / BMJ Case Reports / PMC)",
                "composite": "🧪 Authored teaching vignette (not a real case)",
            }[t],
            key="author_source_type",
        )
        deidentified = st.checkbox(
            "Confirmed de-identified" if source_type == "real" else "De-identified (published cases: pre-cleared by journal)",
            value=(source_type == "published"),
            key="author_deid",
        )

    if source_type == "real":
        origin = st.text_input("Case origin",
                               placeholder="e.g. LU-affiliated hospital, IM, 2026",
                               key="author_origin")
        consent_note = st.text_input("Ethics / consent note",
                                     placeholder="e.g. IRB waiver #, ethics committee approval ref",
                                     key="author_consent")
        citation = None
    elif source_type == "published":
        st.caption("Paraphrase the vignette/follow-up in your own words below — "
                  "don't paste the article's text directly, that's a copyright "
                  "problem even for a few sentences. Facts and timeline are fine "
                  "to reuse; keep the citation attached.")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            cite_journal = st.text_input("Journal", placeholder="e.g. BMJ Case Reports", key="author_cite_journal")
            cite_year = st.text_input("Year", placeholder="e.g. 2024", key="author_cite_year")
        with cc2:
            cite_pmid = st.text_input("PMID", placeholder="e.g. 38123456", key="author_cite_pmid")
            cite_doi = st.text_input("DOI", placeholder="e.g. 10.1136/bcr-2024-...", key="author_cite_doi")
        with cc3:
            cite_authors = st.text_input("Authors", placeholder="e.g. Smith J, et al.", key="author_cite_authors")
            cite_url = st.text_input("URL", placeholder="https://pubmed.ncbi.nlm.nih.gov/...", key="author_cite_url")
        origin = cite_journal or "PubMed / BMJ Case Reports"
        consent_note = "Published case report — de-identification and consent handled by the journal prior to publication"
        citation = {
            "authors": cite_authors, "title": title, "journal": cite_journal,
            "year": cite_year, "pmid": cite_pmid, "doi": cite_doi, "url": cite_url,
        }
    else:
        origin = "Authored teaching vignette"
        consent_note = "N/A — synthetic case, no real patient involved"
        citation = None

    vignette = st.text_area("Vignette (what the student sees at the start)",
                            height=140, key="author_vignette",
                            placeholder="HPI, PMH, meds, vitals, exam findings...")
    questions_raw = st.text_area("Guiding questions (one per line, not graded)",
                                 height=80, key="author_questions")

    st.markdown("#### Vitals at presentation")
    v1, v2, v3, v4, v5 = st.columns(5)
    with v1: hr0 = st.number_input("HR", value=90, key="author_hr0")
    with v2: sbp0 = st.number_input("SBP", value=120, key="author_sbp0")
    with v3: dbp0 = st.number_input("DBP", value=80, key="author_dbp0")
    with v4: spo20 = st.number_input("SpO2", value=98, key="author_spo20")
    with v5: rr0 = st.number_input("RR", value=16, key="author_rr0")

    st.markdown("---")
    st.markdown("#### The ideal workup (this becomes the hidden expert tree)")
    st.caption("Add each step in the order it should happen. Mark the ones "
              "that are time-critical — those drive the vitals/timer during "
              "the student's attempt.")

    for i, step in enumerate(steps):
        with st.container(border=True):
            sc1, sc2, sc3, sc4 = st.columns([1, 2, 2, 1])
            with sc1:
                step["type"] = st.selectbox(
                    "Type", [t for t, _, _, _ in NODE_TYPES],
                    index=[t for t, _, _, _ in NODE_TYPES].index(step.get("type", "history")),
                    key=f"astep_type_{i}",
                    format_func=lambda t: NODE_TYPE_LABELS.get(t, t))
            with sc2:
                step["label"] = st.text_input("Label", value=step.get("label", ""),
                                              key=f"astep_label_{i}")
            with sc3:
                step["rationale"] = st.text_input("Why (1 line)",
                                                   value=step.get("rationale", ""),
                                                   key=f"astep_rationale_{i}")
            with sc4:
                step["critical"] = st.checkbox("Critical", value=step.get("critical", False),
                                               key=f"astep_crit_{i}")
            if step["critical"]:
                step["target_by"] = st.number_input(
                    "Must be done by (seconds into case)",
                    min_value=0, value=step.get("target_by", 300),
                    key=f"astep_target_{i}")
            syn_raw = st.text_input(
                "Also accept these phrasings (comma-separated — e.g. brand "
                "names, local terminology)",
                value=", ".join(step.get("synonyms", [])),
                key=f"astep_syn_{i}",
                placeholder="e.g. Brilinta, Ticagrelor 90mg",
            )
            step["synonyms"] = [s.strip() for s in syn_raw.split(",") if s.strip()]
            if st.button("🗑 Remove step", key=f"astep_del_{i}"):
                steps.pop(i)
                st.rerun()

    if st.button("➕ Add step"):
        steps.append({"type": "history", "label": "", "rationale": "",
                     "critical": False, "target_by": 300})
        st.rerun()

    st.markdown("---")
    st.markdown("#### What actually happened (follow-up — hidden until the student finishes)")
    actual_path = st.text_area("Hospital course", height=90, key="author_actual_path")
    complications = st.text_input("Complications (or 'None')", key="author_complications")
    discharge_outcome = st.text_input("Discharge outcome", key="author_discharge")
    long_term = st.text_input("Long-term follow-up", key="author_longterm")
    teaching_pearl = st.text_area("Teaching pearl", height=70, key="author_pearl")

    st.markdown("---")
    if st.button("💾 Save case & hide it", type="primary"):
        if not title or not vignette or not steps:
            st.error("Title, vignette, and at least one workup step are required.")
            return
        expert_nodes = []
        expert_edges = []
        critical_steps = []
        critical_labels = []
        prev_id = None
        for i, step in enumerate(steps):
            if not step.get("label"):
                continue
            nid = f"n_{i+1:03d}"
            expert_nodes.append(rc._build_expert_node(
                nid, step["type"], step["label"], step.get("rationale", ""),
                synonyms=step.get("synonyms", [])))
            if prev_id:
                expert_edges.append({"from": prev_id, "to": nid})
            prev_id = nid
            if step.get("critical"):
                critical_steps.append({
                    "label": step["label"],
                    "target_by": step.get("target_by", 300),
                    "done_at": None,
                })
                critical_labels.append(step["label"])

        case = {
            "id": st.session_state.get("author_case_id", ""),
            "title": title,
            "specialty": specialty or "General",
            "difficulty": difficulty,
            "estimated_min": est_min,
            "vignette": vignette,
            "questions_for_student": [q.strip() for q in questions_raw.split("\n") if q.strip()],
            "critical_steps": critical_steps,
            "critical_node_labels": critical_labels,
            "vitals_initial": {"hr": hr0, "sbp": sbp0, "dbp": dbp0, "spo2": spo20, "rr": rr0},
            "expert_nodes": expert_nodes,
            "expert_edges": expert_edges,
            "case_source": {
                "type": source_type,
                "origin": origin or "Not specified",
                "deidentified": deidentified,
                "consent_note": consent_note or "Not specified",
                "citation": citation,
            },
            "follow_up": {
                "actual_path": actual_path,
                "complications": complications,
                "discharge_outcome": discharge_outcome,
                "long_term": long_term,
                "teaching_pearl": teaching_pearl,
            },
        }
        case_id = rc.save_json_case(case)
        st.success(f"Saved as case '{case_id}'. It's hidden — students only see "
                  f"the vignette until they finish their tree.")
        for k in list(st.session_state.keys()):
            if k.startswith("author_") or k.startswith("astep_"):
                st.session_state.pop(k, None)
        st.session_state.pop("author_steps", None)

    # ── Follow-up timeline editor (multi-month case evolution) ──────────
    st.markdown("---")
    st.markdown("#### 📅 Follow-up timeline (multi-month evolution)")
    st.caption("Push new clinical data to residents who already did the case. "
              "Each update appears to them as a '🔁 Re-reason with new data' "
              "notification on the case card. The system measures how their "
              "reasoning evolved — that's the killer feature.")
    from reasoning_followup import load_timeline, add_update, delete_update
    # For existing case, load its timeline; for a new case, use the in-progress id
    editing_case_id = st.session_state.get("author_case_id", "")
    if not editing_case_id:
        # Try to get from form state — fall back to a temp id
        editing_case_id = "new_case_draft"
    timeline = load_timeline(editing_case_id) if editing_case_id != "new_case_draft" else []
    if timeline:
        st.markdown(f"**{len(timeline)} existing update(s) for this case:**")
        for u in timeline:
            with st.container(border=True):
                st.markdown(f"**Day {u.get('at_days','?')} — {u.get('headline','')}**")
                st.caption(f"Type: {u.get('type','?')} • {u.get('prompt','')}")
                if st.button(f"🗑 Remove this update",
                            key=f"del_update_{u['id']}"):
                    delete_update(editing_case_id, u["id"])
                    st.rerun()
    # Add new update form
    st.markdown("**Add a new follow-up update:**")
    fu1, fu2, fu3 = st.columns(3)
    with fu1:
        fu_at = st.number_input("At day (after initial case)", min_value=1,
                                value=7, key="fu_at_days")
    with fu2:
        fu_type = st.selectbox("Type",
                               [t for t, _, _ in UPDATE_TYPES],
                               format_func=lambda t: UPDATE_TYPE_LABELS.get(t, t),
                               key="fu_type")
    with fu3:
        fu_headline = st.text_input("Headline", key="fu_headline",
                                    placeholder="e.g. Troponin returns elevated")
    fu_data_raw = st.text_area("New clinical data (JSON, optional)",
                              key="fu_data",
                              placeholder='{"troponin": 4.2, "units": "ng/mL"}',
                              height=70)
    fu_prompt = st.text_input("Prompt to the resident", key="fu_prompt",
                              placeholder="How does this change your plan?")
    fu_teach = st.text_area("Teaching note (shown after they answer)",
                           key="fu_teach", height=70)
    if st.button("➕ Add this update", key="fu_add_btn"):
        if not fu_headline or not fu_prompt:
            st.error("Headline and prompt are required.")
        else:
            data = None
            if fu_data_raw.strip():
                try:
                    data = json.loads(fu_data_raw)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
                    return
            if editing_case_id == "new_case_draft":
                st.warning("Save the case first, then come back here to add "
                          "follow-up updates. (Updates are keyed to the "
                          "saved case id.)")
            else:
                add_update(editing_case_id, {
                    "at_days": fu_at,
                    "type": fu_type,
                    "headline": fu_headline,
                    "data": data,
                    "prompt": fu_prompt,
                    "teaching_note": fu_teach,
                })
                st.success(f"Update added to case '{editing_case_id}'.")
                # Clear the form
                for k in ["fu_at_days", "fu_type", "fu_headline", "fu_data",
                          "fu_prompt", "fu_teach", "fu_add_btn"]:
                    st.session_state.pop(k, None)
                st.rerun()


def _finish_case(sid: str, case: dict, tree: dict, sim):
    """Score the student tree, save to session, show X-ray.

    If we're in re-reasoning mode, we also save the response to the
    follow-up record and compute the delta against the original tree.
    """
    result = score_tree(tree, case["expert_tree"],
                        case.get("critical_node_labels"))
    # In re-reasoning mode there's no vitals sim, so no outcome to report
    if sim is not None:
        outcome = sim.outcome()
    else:
        outcome = "re-reasoned"
    st.session_state["rc_xray_result"] = {
        "score":   result,
        "outcome": outcome,
        "case_id": case["id"],
        "tree":    tree,
    }

    # If this is a re-reasoning response, save it to the follow-up record
    is_rereason = st.session_state.get("rc_rereason", False)
    if is_rereason:
        from reasoning_followup import (
            load_responses, save_responses, compute_delta, days_since,
        )
        rec = load_responses(case["id"], sid)
        # Backfill initial if somehow missing
        if not rec.get("initial") and tree:
            rec["initial"] = {
                "tree": tree, "score": result["total"],
                "ts": datetime.datetime.now().isoformat(),
            }
        update = st.session_state.get("rc_rereason_update", {})
        if update and update.get("id"):
            day_seen = int(days_since(rec["initial"].get("ts", "")))
            rec["responses"][update["id"]] = {
                "tree":   tree,
                "score":  result["total"],
                "day":    day_seen,
                "ts":     datetime.datetime.now().isoformat(),
                "notes":  "",
            }
            # Compute and stash the delta for the X-ray
            delta = compute_delta(rec["initial"].get("tree", {}), tree)
            st.session_state["rc_xray_result"]["delta"] = delta
            st.session_state["rc_xray_result"]["rereason_update"] = update
        save_responses(case["id"], sid, rec)
    else:
        # Persist as the resident's initial attempt for follow-ups later
        from reasoning_followup import load_responses, save_responses
        rec = load_responses(case["id"], sid)
        if not rec.get("initial"):
            rec["initial"] = {
                "tree":  tree,
                "score": result["total"],
                "ts":    datetime.datetime.now().isoformat(),
            }
            save_responses(case["id"], sid, rec)
        # Persist to student record (treat reasoning canvas as a new dept)
        try:
            record_session(sid, "reasoning_canvas", result["total"],
                           notes=f"case={case['id']} outcome={outcome}")
        except ValueError:
            # Department not whitelisted — record manually
            data = load_student(sid)
            data.setdefault("scores", {})["reasoning_canvas"] = (
                data["scores"].get("reasoning_canvas", []) + [result["total"]]
            )
            data["total_sessions"] = data.get("total_sessions", 0) + 1
            data.setdefault("best", {})["reasoning_canvas"] = max(
                data["best"].get("reasoning_canvas", 0), result["total"])
            save_student(data)
    st.session_state["rc_finished"] = True
    st.rerun()


def render_xray():
    """Show the X-ray debrief after a finished case."""
    xr = st.session_state.get("rc_xray_result")
    if not xr:
        return
    case = get_case(xr["case_id"])
    score = xr["score"]
    outcome = xr["outcome"]

    st.markdown("## 🩻 Reasoning X-Ray")
    st.caption(f"Case: {case['title']}")

    # Score panel
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total", f"{score['total']}/100")
    with c2:
        st.metric("Critical (50%)", f"{score['critical']}/100")
    with c3:
        st.metric("Order (30%)", f"{score['order']}/100")
    with c4:
        st.metric("Value (20%)", f"{score['value']}/100")

    # Outcome
    outcome_msg = {
        "stable":              "🟢 Patient stable. Your plan worked.",
        "deteriorated":        "🟡 Patient deteriorated. Some critical steps were late.",
        "crash":               "🔴 Patient crashed. Critical steps were missed or very late.",
        "missed_critical_step":"🟠 Patient was saved, but a critical step was missed.",
        "re-reasoned":         "🔁 Re-reasoning — see the evolution panel below.",
    }
    st.info(f"**Outcome:** {outcome_msg.get(outcome, outcome)}")
    if outcome != "re-reasoned":
        st.caption(
            "This outcome is your SIMULATED patient reacting to your timing in this "
            "session — not what happened in real life. See the follow-up below for that."
        )

    # ── EVOLUTION PANEL — shown when re-reasoning ─────────────────────────
    if xr.get("delta"):
        st.markdown("---")
        st.markdown("### 🔁 Reasoning Evolution")
        delta = xr["delta"]
        update = xr.get("rereason_update", {})

        ev1, ev2, ev3, ev4 = st.columns(4)
        with ev1:
            st.metric("Original score", f"{score.get('_initial_score', '—')}/100"
                       if score.get("_initial_score") else
                       f"see history")
        # Show score change
        initial_score = (st.session_state.get("rc_rereason_initial_score") or 0)
        # If we don't have it from session, look it up
        if not initial_score:
            from reasoning_followup import load_responses
            rec = load_responses(xr["case_id"], st.session_state.get("student_id", "guest"))
            if rec.get("initial"):
                initial_score = rec["initial"].get("score", 0)
        score_change = score["total"] - initial_score if initial_score else 0
        with ev2:
            st.metric("Re-reasoned score", f"{score['total']}/100",
                      delta=f"{score_change:+.0f}" if initial_score else None)
        with ev3:
            st.metric("Nodes added", len(delta["added_nodes"]))
        with ev4:
            st.metric("Nodes removed", len(delta["removed_nodes"]))

        if delta.get("rebalanced"):
            st.warning(
                f"🔄 **Major re-reasoning detected.** You changed "
                f"{len(delta['added_nodes']) + len(delta['removed_nodes'])} of "
                f"your reasoning steps. That's a deep re-evaluation, not a tweak — "
                f"this is what good clinical reasoning looks like when new data "
                f"arrives."
            )

        # The new data that triggered this re-reasoning
        st.markdown("#### What triggered this re-reasoning")
        st.warning(f"**{update.get('headline','')}** (day {update.get('at_days','?')})")
        if update.get("teaching_note"):
            st.success(f"💡 **Teaching note:** {update['teaching_note']}")

        # Diff in detail
        st.markdown("#### What you changed")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**🟢 Added to your plan**")
            if delta["added_nodes"]:
                for n in delta["added_nodes"]:
                    st.markdown(f"- {n}")
            else:
                st.caption("(nothing added)")
        with d2:
            st.markdown("**🔴 Removed from your plan**")
            if delta["removed_nodes"]:
                for n in delta["removed_nodes"]:
                    st.markdown(f"- {n}")
            else:
                st.caption("(nothing removed)")

        if delta["kept_nodes"]:
            st.markdown("**🟡 Kept from your original plan**")
            st.caption(", ".join(sorted(delta["kept_nodes"])))

        # Cohort comparison
        from reasoning_followup import cohort_evolution_stats
        stats = cohort_evolution_stats(xr["case_id"])
        if stats["n_residents"] >= 1:
            st.markdown("---")
            st.markdown("#### 📊 How your cohort reasoned at this update")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.metric("Residents who re-reasoned",
                          f"{stats['n_residents']}")
            with cc2:
                st.metric("Cohort avg score change",
                          f"{stats['avg_score_change']:+.0f}")
            with cc3:
                st.metric("Rebalanced their tree",
                          f"{int(stats['rebalanced_rate']*100)}%")
            if stats["most_added_nodes"]:
                st.markdown("**What other residents added at this update:**")
                for label, count in stats["most_added_nodes"][:5]:
                    st.markdown(f"- {label} *(added by {count} "
                               f"resident{'s' if count != 1 else ''})*")
            if stats["most_dropped_nodes"]:
                st.markdown("**What other residents dropped at this update:**")
                for label, count in stats["most_dropped_nodes"][:5]:
                    st.markdown(f"- {label} *(dropped by {count} "
                               f"resident{'s' if count != 1 else ''})*")
        st.markdown("---")

    # Real-case follow-up — what actually happened to the real patient,
    # independent of how this student's simulated session went.
    follow_up = case.get("follow_up", {})
    case_source = case.get("case_source", {})
    if follow_up.get("actual_path"):
        st.markdown("---")
        st.markdown("### 📋 What Actually Happened (Real Case Follow-Up)")
        if case_source.get("type") == "real":
            st.caption(f"Source: {case_source.get('origin', 'De-identified hospital case')}")
        else:
            st.caption("⚠️ This case is a composite teaching vignette — the follow-up below is illustrative, not a real patient.")
        st.markdown(f"**Hospital course:** {follow_up['actual_path']}")
        if follow_up.get("complications"):
            st.markdown(f"**Complications:** {follow_up['complications']}")
        if follow_up.get("discharge_outcome"):
            st.markdown(f"**Discharge outcome:** {follow_up['discharge_outcome']}")
        if follow_up.get("long_term"):
            st.markdown(f"**Long-term follow-up:** {follow_up['long_term']}")
        if follow_up.get("teaching_pearl"):
            st.success(f"💡 **Teaching pearl:** {follow_up['teaching_pearl']}")
    else:
        st.markdown("---")
        st.warning(
            "📋 No real-world follow-up has been added to this case yet. "
            "See `new_real_case_template()` in reasoning_canvas.py to add one."
        )

    st.markdown("---")

    # Full original case reveal — the leader's complete authored path,
    # unlocked now that the student has finished. This is the actual case
    # "opening up" for the student to check themselves against, separate
    # from the automated diff below.
    expert_nodes = case.get("expert_tree", {}).get("nodes", [])
    if expert_nodes:
        with st.expander("📖 Reveal the full original case (now unlocked)", expanded=True):
            citation = case.get("case_source", {}).get("citation")
            if citation and case.get("case_source", {}).get("type") == "published":
                cite_str = f"{citation.get('authors', '')}. {citation.get('title', '')}. " \
                          f"{citation.get('journal', '')}, {citation.get('year', '')}."
                if citation.get("doi"):
                    cite_str += f" DOI: {citation['doi']}."
                st.caption(f"📄 Source: {cite_str}")
                if citation.get("url"):
                    st.caption(f"🔗 {citation['url']}")
            st.caption("This is the complete case as written by the leader — "
                      "read it and compare it to your own tree, step by step, "
                      "not just the automated score below.")
            for n in expert_nodes:
                icon = NODE_TYPE_ICONS.get(n["type"], "•")
                st.markdown(f"**{icon} {n['label']}**")
                if n.get("rationale"):
                    st.caption(n["rationale"])

    st.markdown("---")

    # Side-by-side trees
    st.markdown("### Your tree vs. expert tree")
    cl, cr = st.columns(2)
    with cl:
        st.markdown("**🧑‍⚕️ Your tree**")
        st.markdown(render_tree_svg(xr["tree"]), unsafe_allow_html=True)
    with cr:
        st.markdown("**👨‍⚕️ Expert tree**")
        st.markdown(render_tree_svg(case["expert_tree"]),
                    unsafe_allow_html=True)

    # Diff
    st.markdown("### Diff")
    diff = score["diff"]
    match_scores = diff.get("match_scores", {})
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown("#### ✓ Matched (good)")
        for s, e in score["matched"]:
            conf = match_scores.get(e["id"], 1.0)
            conf_note = "" if conf >= 0.99 else f'<br><small style="color:#888">~{int(conf*100)}% match — different wording</small>'
            st.markdown(
                f'<div class="rc-xray-matched">'
                f'<b>{html.escape(s["label"])}</b><br>'
                f'<small>↔ {html.escape(e["label"])}</small>{conf_note}</div>',
                unsafe_allow_html=True)
    with cc2:
        st.markdown("#### ⚠ Missed (expert-only)")
        for n in score["missed"]:
            st.markdown(
                f'<div class="rc-xray-missed">'
                f'<b>{html.escape(n["label"])}</b><br>'
                f'<small>{html.escape(n.get("rationale", ""))}</small></div>',
                unsafe_allow_html=True)
    with cc3:
        st.markdown("#### + Extra (you-only)")
        for n in score["extra"]:
            st.markdown(
                f'<div class="rc-xray-extra">'
                f'<b>{html.escape(n["label"])}</b><br>'
                f'<small>{html.escape(n.get("rationale", ""))}</small></div>',
                unsafe_allow_html=True)

    # Order issues
    if score["order_issues"]:
        st.markdown("### ⏱ Order issues")
        for a, b in score["order_issues"]:
            st.warning(f"'{a}' should come before '{b}' in the expert pathway.")

    # Actions
    st.markdown("---")
    ca1, ca2, ca3 = st.columns(3)
    with ca1:
        if st.button("🔁 Try again (same case)", use_container_width=True):
            for k in ["rc_tree", "rc_sim", "rc_t0", "rc_selected_node",
                      "rc_finished", "rc_xray_result"]:
                st.session_state.pop(k, None)
            st.rerun()
    with ca2:
        if st.button("🧠 New case", use_container_width=True):
            for k in ["rc_case_id", "rc_tree", "rc_sim", "rc_t0",
                      "rc_selected_node", "rc_finished", "rc_xray_result"]:
                st.session_state.pop(k, None)
            st.rerun()
    with ca3:
        # Export student tree as JSON
        export = {
            "case_id":  xr["case_id"],
            "case_title": case["title"],
            "score":    score["total"],
            "outcome":  outcome,
            "tree":     xr["tree"],
        }
        st.download_button(
            "💾 Export my tree (JSON)",
            data=json.dumps(export, indent=2),
            file_name=f"reasoning_canvas_{xr['case_id']}.json",
            mime="application/json",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Top-level routing: case-select / active case / x-ray
    if "rc_finished" in st.session_state and st.session_state["rc_finished"]:
        render_xray()
        if st.button("← Back to case select"):
            for k in ["rc_case_id", "rc_finished", "rc_xray_result"]:
                st.session_state.pop(k, None)
            st.rerun()
    else:
        sid = st.session_state.get("student_id", "guest")
        render(sid)


if __name__ == "__main__":
    # Allow running as `streamlit run reasoning_canvas_ui.py`
    main()
