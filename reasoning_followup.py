"""Multi-month case evolution — the "real cases follow up" engine.

This module turns the static `follow_up` field (a single teaching summary
shown after the case ends) into a *timeline* of clinical updates that
appear days/weeks/months after the resident's initial reasoning.

The pedagogical point: a resident's reasoning about a patient should
*evolve* as new data arrives. A resident who drew a perfect tree in
October and then ignores the December echo result hasn't really learned
the case — they've memorized the initial presentation.

This is the part of the system that actually closes the "reasoning-to-
bedside" gap. No other Q-bank or virtual patient simulator does this.

Data model
==========
A case has a `follow_up_timeline` field: a list of updates that fire
at specific days after the resident's initial attempt. Each update is
a piece of new clinical data (lab result, complication, follow-up
visit, new imaging) plus a prompt asking the resident to re-reason.

The resident's session for a case is stored with:
- initial_tree: their first attempt
- updates: list of (update_id, response_tree, response_score, day_seen)

We score the *delta* between the original tree and the response tree
to measure how the resident's reasoning evolved.

Engagement loop
===============
1. Resident does case on day 0 → scored tree stored
2. Case author (or auto-rule) pushes an update at day 7, 30, 90, ...
3. Resident gets a notification: "Mr. Khalil's case has new data"
4. Resident opens the case → sees their original tree + the new data
5. Resident draws an updated tree → scored against the same expert tree
6. Delta is computed: what did they add? what did they drop? what changed?
7. Cohort effect: after 5+ residents re-reason, the cohort view shows
   the average evolution pattern
"""
from __future__ import annotations
import json
import os
import datetime
import time
from typing import Dict, List, Optional, Tuple

# The follow-ups are stored as a separate JSON file per case, kept in
# cases/<case_id>_followups.json. This is separate from the case file
# so case authors can edit the case without touching the timeline, and
# so we can ship cases that have NO follow-ups yet without the field
# bloating every case.

_FOLLOWUPS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "case_followups")
os.makedirs(_FOLLOWUPS_DIR, exist_ok=True)

# Resident responses (their updated trees at each update) are stored
# in case_responses/<case_id>/<resident_id>.json — keyed by case so
# the chief can browse "everyone's responses for case X" easily.
_RESPONSES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "case_responses")
os.makedirs(_RESPONSES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Update types
# ─────────────────────────────────────────────────────────────────────────────
UPDATE_TYPES = [
    ("lab_result",       "🧪 Lab result returns",       "Lab or bloodwork that changes the picture"),
    ("imaging",          "🩻 Imaging result",            "CXR, CT, echo, US — anything with a picture"),
    ("complication",     "⚠️ Complication develops",     "Patient deteriorates or develops new finding"),
    ("follow_up_visit",  "🏥 Follow-up visit",           "Patient comes back, new H&P"),
    ("new_data",         "📋 New clinical data",         "Anything else (pathology, culture, consult)"),
    ("discharge",        "🚪 Discharge summary",         "End of acute course, summary of stay"),
    ("long_term",        "📅 Long-term follow-up",       "Weeks/months later — what happened"),
    ("course_correction","🔄 Course correction",         "Original plan was wrong, here's what we did instead"),
]

UPDATE_TYPE_LABELS = {t: l for t, l, _ in UPDATE_TYPES}


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up timeline (case author side)
# ─────────────────────────────────────────────────────────────────────────────
def _followups_path(case_id: str) -> str:
    return os.path.join(_FOLLOWUPS_DIR, f"{_safe_id(case_id)}.json")


def _safe_id(s: str) -> str:
    """Filesystem-safe case id (matches _slugify in reasoning_canvas.py)."""
    import re
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_") or "case"


def load_timeline(case_id: str) -> List[dict]:
    """Return the list of follow-up updates for a case. Empty list if none."""
    p = _followups_path(case_id)
    if not os.path.exists(p):
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_timeline(case_id: str, updates: List[dict]) -> None:
    """Persist the follow-up timeline for a case. Atomic write."""
    p = _followups_path(case_id)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(updates, f, indent=2)
    os.replace(tmp, p)


def add_update(case_id: str, update: dict) -> dict:
    """Add a single update to a case's timeline. Returns it with id filled in."""
    updates = load_timeline(case_id)
    if "id" not in update:
        update["id"] = f"u_{len(updates) + 1:03d}"
    if "created_at" not in update:
        update["created_at"] = datetime.datetime.now().isoformat()
    updates.append(update)
    save_timeline(case_id, updates)
    return update


def delete_update(case_id: str, update_id: str) -> None:
    updates = [u for u in load_timeline(case_id) if u.get("id") != update_id]
    save_timeline(case_id, updates)


# ─────────────────────────────────────────────────────────────────────────────
# Resident side: pending updates + responses
# ─────────────────────────────────────────────────────────────────────────────
def pending_updates(case_id: str, days_since_first_attempt: float) -> List[dict]:
    """Return the updates that should have been revealed by now.

    We don't actually need the resident's first attempt time for this
    math — the caller computes `days_since_first_attempt` from the
    resident's stored session data.
    """
    out = []
    for u in load_timeline(case_id):
        if u.get("at_days", 0) <= days_since_first_attempt:
            out.append(u)
    return out


def _response_path(case_id: str, resident_id: str) -> str:
    d = os.path.join(_RESPONSES_DIR, _safe_id(case_id))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{_safe_id(resident_id)}.json")


def load_responses(case_id: str, resident_id: str) -> dict:
    """Return the resident's response record for a case.

    Record shape:
        {
            "case_id": "...",
            "resident_id": "...",
            "initial": {tree, score, ts},  # first attempt
            "responses": {                 # keyed by update_id
                "u_001": {tree, score, day, ts, notes},
                "u_002": {...},
            }
        }
    """
    p = _response_path(case_id, resident_id)
    if not os.path.exists(p):
        return {"case_id": case_id, "resident_id": resident_id,
                "initial": None, "responses": {}}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"case_id": case_id, "resident_id": resident_id,
                "initial": None, "responses": {}}


def save_responses(case_id: str, resident_id: str, record: dict) -> None:
    p = _response_path(case_id, resident_id)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, p)


def all_responses_for_case(case_id: str) -> List[dict]:
    """For the cohort view: every resident's response record for this case."""
    d = os.path.join(_RESPONSES_DIR, _safe_id(case_id))
    if not os.path.isdir(d):
        return []
    out = []
    for fn in os.listdir(d):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(d, fn)) as f:
                    out.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning delta — the part that measures how thinking EVOLVED
# ─────────────────────────────────────────────────────────────────────────────
def compute_delta(original_tree: dict, updated_tree: dict) -> dict:
    """Compare two trees and report what changed.

    Returns a dict suitable for the X-ray "Evolution" panel:
        added_nodes:    list of labels the resident added in the update
        removed_nodes:  list of labels the resident dropped
        kept_nodes:     list of labels that were in both
        added_edges:    list of new connections
        removed_edges:  list of dropped connections
        original_complexity: node count, edge count
        updated_complexity:  node count, edge count
        complexity_delta:    net change
        rebalanced:     True if resident dramatically restructured (>50% nodes different)
    """
    def label_set(t):
        return {"".join(c.lower() for c in n["label"]
                       if c.isalnum() or c.isspace()).strip()
                for n in t.get("nodes", [])}

    def edge_set(t):
        return {(e["from"], e["to"]) for e in t.get("edges", [])}

    o_lab = label_set(original_tree)
    u_lab = label_set(updated_tree)
    added = u_lab - o_lab
    removed = o_lab - u_lab
    kept = o_lab & u_lab

    o_edges = edge_set(original_tree)
    u_edges = edge_set(updated_tree)
    added_edges = u_edges - o_edges
    removed_edges = o_edges - u_edges

    orig_n, orig_e = len(original_tree.get("nodes", [])), len(original_tree.get("edges", []))
    new_n,  new_e  = len(updated_tree.get("nodes", [])),  len(updated_tree.get("edges", []))

    rebalanced = (len(added) + len(removed)) > 0.5 * max(len(o_lab), len(u_lab), 1)

    return {
        "added_nodes":   sorted(added),
        "removed_nodes": sorted(removed),
        "kept_nodes":    sorted(kept),
        "added_edges_count":   len(added_edges),
        "removed_edges_count": len(removed_edges),
        "original_complexity": {"nodes": orig_n, "edges": orig_e},
        "updated_complexity":  {"nodes": new_n,  "edges": new_e},
        "complexity_delta":    {"nodes": new_n - orig_n, "edges": new_e - orig_e},
        "rebalanced":    rebalanced,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cohort evolution statistics
# ─────────────────────────────────────────────────────────────────────────────
def cohort_evolution_stats(case_id: str) -> dict:
    """Aggregate statistics across all residents who did the case and
    re-reasoned with at least one update.

    Returns:
        n_residents:       how many residents re-reasoned
        n_updates:         how many updates exist for the case
        avg_score_change:  mean of (updated_score - original_score) across residents
        rebalanced_rate:   fraction of residents who rebalanced their tree
        most_added_nodes:  Counter of which nodes residents added most often
        most_dropped_nodes: Counter of which nodes residents dropped most often
        per_update_stats:  {update_id: {n_responded, avg_score_for_this_update}}
    """
    from collections import Counter
    all_recs = all_responses_for_case(case_id)
    if not all_recs:
        return {
            "n_residents": 0, "n_updates": len(load_timeline(case_id)),
            "avg_score_change": 0.0, "rebalanced_rate": 0.0,
            "most_added_nodes": [], "most_dropped_nodes": [],
            "per_update_stats": {},
        }

    score_changes = []
    rebalanced_count = 0
    added_counter = Counter()
    removed_counter = Counter()
    per_update: Dict[str, List[float]] = {}

    for rec in all_recs:
        if not rec.get("initial"):
            continue
        original_score = rec["initial"].get("score", 0)
        for uid, resp in rec.get("responses", {}).items():
            score_changes.append(resp.get("score", 0) - original_score)
            per_update.setdefault(uid, []).append(resp.get("score", 0))
            # We don't store the delta in the response — reconstruct on the fly
            delta = compute_delta(rec["initial"].get("tree", {}),
                                  resp.get("tree", {}))
            if delta["rebalanced"]:
                rebalanced_count += 1
            for a in delta["added_nodes"]:
                added_counter[a] += 1
            for r in delta["removed_nodes"]:
                removed_counter[r] += 1

    n = len(all_recs)
    return {
        "n_residents":     n,
        "n_updates":       len(load_timeline(case_id)),
        "avg_score_change": round(sum(score_changes) / max(1, len(score_changes)), 1),
        "rebalanced_rate":  round(rebalanced_count / max(1, n), 2),
        "most_added_nodes":   added_counter.most_common(10),
        "most_dropped_nodes": removed_counter.most_common(10),
        "per_update_stats": {
            uid: {"n_responded": len(scores),
                  "avg_score": round(sum(scores) / max(1, len(scores)), 1)}
            for uid, scores in per_update.items()
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: days since a timestamp
# ─────────────────────────────────────────────────────────────────────────────
def days_since(ts_iso: str) -> float:
    """Return how many days ago an ISO timestamp was, as a float."""
    try:
        then = datetime.datetime.fromisoformat(ts_iso)
        now = datetime.datetime.now()
        return (now - then).total_seconds() / 86400.0
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing multi-month case evolution engine...")

    # Use a temp case id so we don't pollute
    TEST_CASE = "_test_evolution"
    TEST_RES = "test_resident"

    # Clean any previous test data
    for fn in [os.path.join(_FOLLOWUPS_DIR, f"{_safe_id(TEST_CASE)}.json"),
               _response_path(TEST_CASE, TEST_RES)]:
        if os.path.exists(fn):
            os.remove(fn)

    # Add a timeline
    updates = []
    updates.append(add_update(TEST_CASE, {
        "at_days": 7,
        "type": "lab_result",
        "headline": "Troponin returns elevated",
        "data": {"troponin": 4.2, "units": "ng/mL"},
        "prompt": "Does this change your management?",
        "teaching_note": "An elevated troponin at 6h confirms myocardial injury.",
    }))
    updates.append(add_update(TEST_CASE, {
        "at_days": 30,
        "type": "follow_up_visit",
        "headline": "Patient returns with dyspnea on exertion",
        "data": {"new_symptom": "DOE", "duration_days": 5},
        "prompt": "What's your new differential?",
        "teaching_note": "DOE 30 days post-MI raises concern for heart failure.",
    }))
    print(f"  Added {len(updates)} updates to timeline")

    # Test pending updates
    pending_7 = pending_updates(TEST_CASE, days_since_first_attempt=7)
    pending_3 = pending_updates(TEST_CASE, days_since_first_attempt=3)
    assert len(pending_7) == 1, f"Day 7 should show 1 update, got {len(pending_7)}"
    assert len(pending_3) == 0, f"Day 3 should show 0 updates, got {len(pending_3)}"
    print(f"  Day 7 sees {len(pending_7)} updates, day 3 sees {len(pending_3)} ✓")

    # Test delta computation
    original = {
        "nodes": [
            {"id": "n_1", "type": "lab",        "label": "ECG"},
            {"id": "n_2", "type": "treatment",  "label": "Aspirin"},
            {"id": "n_3", "type": "disposition","label": "Discharge"},
        ],
        "edges": [{"from": "n_1", "to": "n_2"}],
    }
    updated = {
        "nodes": [
            {"id": "n_1", "type": "lab",        "label": "ECG"},
            {"id": "n_2", "type": "treatment",  "label": "Aspirin"},
            {"id": "n_3", "type": "imaging",    "label": "Echo"},
            {"id": "n_4", "type": "treatment",  "label": "Diuretic"},
            {"id": "n_5", "type": "disposition","label": "Cardiology follow-up"},
        ],
        "edges": [
            {"from": "n_1", "to": "n_2"},
            {"from": "n_3", "to": "n_4"},
        ],
    }
    delta = compute_delta(original, updated)
    print(f"  Delta: added={len(delta['added_nodes'])}, "
          f"removed={len(delta['removed_nodes'])}, "
          f"kept={len(delta['kept_nodes'])}")
    assert "echo" in str(delta["added_nodes"]).lower()
    assert "discharge" in str(delta["removed_nodes"]).lower()
    assert "ecg" in str(delta["kept_nodes"]).lower()
    assert delta["rebalanced"] is True
    print("  Delta computation correct ✓")

    # Test response storage
    rec = load_responses(TEST_CASE, TEST_RES)
    rec["initial"] = {
        "tree": original, "score": 80.0,
        "ts": datetime.datetime.now().isoformat(),
    }
    rec["responses"][updates[0]["id"]] = {
        "tree": updated, "score": 88.0, "day": 7,
        "ts": datetime.datetime.now().isoformat(), "notes": "I added echo because...",
    }
    save_responses(TEST_CASE, TEST_RES, rec)
    rec2 = load_responses(TEST_CASE, TEST_RES)
    assert rec2["initial"]["score"] == 80.0
    assert rec2["responses"][updates[0]["id"]]["score"] == 88.0
    print("  Response storage/retrieval works ✓")

    # Test cohort stats
    stats = cohort_evolution_stats(TEST_CASE)
    print(f"  Cohort stats: {stats['n_residents']} residents, "
          f"avg score change = {stats['avg_score_change']}, "
          f"rebalanced rate = {stats['rebalanced_rate']}")
    assert stats["n_residents"] == 1
    assert stats["avg_score_change"] == 8.0  # 88 - 80
    print("  Cohort stats correct ✓")

    # Cleanup
    for fn in [os.path.join(_FOLLOWUPS_DIR, f"{_safe_id(TEST_CASE)}.json"),
               _response_path(TEST_CASE, TEST_RES)]:
        if os.path.exists(fn):
            os.remove(fn)

    print("\n✓ All follow-up evolution tests passed.")
