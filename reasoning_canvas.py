"""Reasoning Canvas — clinical decision-tree module (core engine).

Design philosophy
=================
This is NOT a Q-bank and NOT a flowchart viewer. The student BUILDS a clinical
decision tree as a visible artifact, the patient "reacts" to it, and the system
shows the student exactly where their reasoning diverged from the expert's.

The student is a Lebanese medical resident (or intern) learning internal
medicine reasoning. The cases are real-style clinical vignettes. The tree
is the artifact of reasoning — the X-ray is the feedback.

Data model
==========
A "tree" is a directed graph of nodes connected by edges.

    Node = {
        "id":        "n_001",
        "type":      "history" | "exam" | "lab" | "imaging" | "decision" |
                     "treatment" | "diagnosis" | "disposition",
        "label":     "12-lead ECG",          # short title
        "rationale": "Rule out STEMI first",  # 1-line why
        "result":    "ST elevation II,III,aVF",  # optional: simulated result
        "t_added":   0,                       # seconds from case start
        "t_resolved": 90,                     # seconds until result returned
    }

    Edge = {"from": "n_001", "to": "n_003"}

The expert tree is stored the same way. The diff is computed structurally:
- A node in student but not expert  -> "low-value / over-ordering"
- A node in expert but not student  -> "missed"
- Same node, different position      -> "wrong order"
- Same node, same position          -> "matched"

Scoring
=======
Three orthogonal components, each 0-100:

1. CRITICAL  (50%) — does the student tree include all the "must-have" steps
                     (e.g. aspirin, cath activation for STEMI)? Miss a critical
                     step -> big penalty.

2. ORDER     (30%) — did the student put steps in the right order? Penalize
                     e.g. cath after echo (echo can wait), or discharge before
                     labs are back.

3. VALUE     (20%) — penalty for shotgun ordering. Each expert-not-in-student
                     node counts against the student. Encourages hypothesis-
                     driven testing.

Final score = 0.50*critical + 0.30*order + 0.20*value

This module is pure Python, no Streamlit dependency, so it can be unit-tested.
The Streamlit UI lives in reasoning_canvas_ui.py.
"""
from __future__ import annotations
import math
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Set


# ─────────────────────────────────────────────────────────────────────────────
# Leader-authored case storage (JSON files, no code editing required)
# ─────────────────────────────────────────────────────────────────────────────
# Cases built in Python (CASE_LIBRARY, below) still work as before. Cases a
# leader/attending authors through the "Author a case" screen in the UI are
# saved here instead — one JSON file per case, kept server-side. Students
# never see this folder or its contents directly; only get_case() reads it,
# and only after a case is picked.
CASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases")


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "case"


def list_json_case_ids() -> List[str]:
    if not os.path.isdir(CASES_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(CASES_DIR) if f.endswith(".json"))


def load_json_case_summary(case_id: str) -> dict:
    """Lightweight load for the case picker (title/specialty/etc only)."""
    path = os.path.join(CASES_DIR, f"{case_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_case(case: dict) -> str:
    """Persist a leader-authored case to disk. Returns the case_id used.

    If `case['id']` is missing or blank, one is generated from the title.
    Overwrites an existing file with the same id (so a leader can edit and
    re-save a case they authored earlier).
    """
    os.makedirs(CASES_DIR, exist_ok=True)
    case_id = case.get("id") or _slugify(case.get("title", "case"))
    case_id = _slugify(case_id)
    case["id"] = case_id
    path = os.path.join(CASES_DIR, f"{case_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(case, f, indent=2, ensure_ascii=False)
    return case_id


def delete_json_case(case_id: str) -> None:
    path = os.path.join(CASES_DIR, f"{case_id}.json")
    if os.path.exists(path):
        os.remove(path)


# ─────────────────────────────────────────────────────────────────────────────
# Node types and palette
# ─────────────────────────────────────────────────────────────────────────────
NODE_TYPES = [
    ("history",    "❓", "History",    "Ask the patient a question"),
    ("exam",       "🩺", "Exam",       "Perform a physical exam finding"),
    ("lab",        "🧪", "Lab",        "Order a lab test"),
    ("imaging",    "🩻", "Imaging",    "Order imaging (CXR, CT, echo, US)"),
    ("decision",   "🔀", "Decision",   "Branch point: if X -> Y, else Z"),
    ("treatment",  "💊", "Treatment",  "Give a medication or procedure"),
    ("diagnosis",  "📍", "Diagnosis",  "Commit to a working diagnosis"),
    ("disposition","🚪", "Disposition","Admit, discharge, transfer, observe"),
]

NODE_TYPE_ICONS = {t: i for t, i, _, _ in NODE_TYPES}
NODE_TYPE_LABELS = {t: l for t, _, l, _ in NODE_TYPES}

# Time budget per case (seconds). Vitals evolve within this window.
DEFAULT_TIME_BUDGET = 8 * 60  # 8 minutes for an IM chest pain workup


# ─────────────────────────────────────────────────────────────────────────────
# Graph model
# ─────────────────────────────────────────────────────────────────────────────
def empty_tree() -> dict:
    """Return a fresh empty tree structure."""
    return {"nodes": [], "edges": [], "started_at": None}


def add_node(tree: dict, node_type: str, label: str, rationale: str = "",
             t_added: float = 0.0) -> dict:
    """Add a node to the tree. Returns the new node."""
    nid = f"n_{len(tree['nodes']) + 1:03d}"
    node = {
        "id":        nid,
        "type":      node_type,
        "label":     label,
        "rationale": rationale,
        "result":    None,
        "t_added":   t_added,
        "t_resolved": None,
    }
    tree["nodes"].append(node)
    return node


def add_edge(tree: dict, from_id: str, to_id: str) -> dict:
    """Add a directed edge. Rejects duplicates and self-loops."""
    if from_id == to_id:
        raise ValueError("Self-loops are not allowed")
    for e in tree["edges"]:
        if e["from"] == from_id and e["to"] == to_id:
            return e  # already exists, idempotent
    edge = {"from": from_id, "to": to_id}
    tree["edges"].append(edge)
    return edge


def remove_node(tree: dict, node_id: str) -> None:
    tree["nodes"] = [n for n in tree["nodes"] if n["id"] != node_id]
    tree["edges"] = [e for e in tree["edges"]
                     if e["from"] != node_id and e["to"] != node_id]


def topological_order(tree: dict) -> List[str]:
    """Return node IDs in topological order. Raises ValueError on cycles."""
    incoming = {n["id"]: 0 for n in tree["nodes"]}
    for e in tree["edges"]:
        incoming[e["to"]] = incoming.get(e["to"], 0) + 1
    queue = [nid for nid, c in incoming.items() if c == 0]
    order = []
    while queue:
        queue.sort()  # deterministic for equal-rank nodes
        nid = queue.pop(0)
        order.append(nid)
        for e in tree["edges"]:
            if e["from"] == nid:
                incoming[e["to"]] -= 1
                if incoming[e["to"]] == 0:
                    queue.append(e["to"])
    if len(order) != len(tree["nodes"]):
        raise ValueError("Tree contains a cycle — check your edges")
    return order


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
import difflib

_STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "in", "on", "with", "and", "or",
    "within", "at", "by", "into", "during", "from", "is", "are", "was",
    "were", "min", "mins", "minutes", "sec", "secs", "seconds",
    # common clinical order-verbs — carry little distinguishing meaning on
    # their own, and otherwise dilute the Dice score (e.g. "Give ASA" vs
    # "Aspirin 325 mg chewed" should match on the drug, not get penalized
    # because "give" isn't echoed on the expert side)
    "give", "start", "administer", "order", "check", "obtain", "get",
    "perform", "assess", "do", "send", "draw", "place", "apply", "review",
    "initiate", "begin", "run", "repeat", "consider", "arrange",
}

# ─────────────────────────────────────────────────────────────────────────────
# Built-in medical synonym/abbreviation dictionary
# ─────────────────────────────────────────────────────────────────────────────
# Each group is a set of interchangeable ways to say the same clinical thing.
# Every word/phrase in a group gets mapped to the SAME canonical phrase before
# matching, so "ECG" and "electrocardiogram" become identical tokens instead
# of scoring 0 for sharing no letters. This is intentionally a starting set
# covering common terms, not exhaustive — add more groups as real usage
# surfaces gaps (case authors can also add case-specific synonyms per step,
# see `synonyms` on expert nodes, for brand names etc. this dictionary can't
# cover generically).
_SYNONYM_GROUPS = [
    {"ecg", "ekg", "electrocardiogram"},
    {"cxr", "chest xray", "chest x ray", "chest radiograph"},
    {"ct", "cat scan", "computed tomography"},
    {"mri", "magnetic resonance imaging"},
    {"us", "ultrasound", "sonography", "sono"},
    {"cbc", "complete blood count"},
    {"bmp", "basic metabolic panel"},
    {"cmp", "comprehensive metabolic panel"},
    {"abg", "arterial blood gas"},
    {"vbg", "venous blood gas"},
    {"iv", "intravenous"},
    {"im", "intramuscular"},
    {"po", "oral", "by mouth"},
    {"npo", "nothing by mouth", "nil by mouth"},
    {"hx", "history"},
    {"pe", "physical exam", "physical examination"},
    {"cp", "chest pain"},
    {"sob", "shortness of breath", "dyspnea", "dyspnoea"},
    {"hr", "heart rate", "pulse"},
    {"bp", "blood pressure"},
    {"rr", "respiratory rate"},
    {"temp", "temperature"},
    {"o2", "oxygen"},
    {"sat", "saturation", "spo2", "sats"},
    {"trop", "troponin", "trops"},
    {"bnp", "brain natriuretic peptide"},
    {"cr", "creatinine"},
    {"bun", "blood urea nitrogen"},
    {"lft", "lfts", "liver function test", "liver function tests", "liver enzymes"},
    {"inr", "coags", "coagulation studies", "coagulation panel"},
    {"ua", "urinalysis"},
    {"cath", "catheterization", "catheterisation"},
    {"pci", "percutaneous coronary intervention", "angioplasty"},
    {"cabg", "coronary artery bypass graft", "bypass surgery"},
    {"asa", "aspirin"},
    {"ntg", "nitroglycerin", "nitro", "gtn", "glyceryl trinitrate"},
    {"abx", "antibiotics", "antibiotic"},
    {"ns", "normal saline"},
    {"lr", "lactated ringers", "ringers lactate"},
    {"epi", "epinephrine", "adrenaline"},
    {"cpr", "cardiopulmonary resuscitation"},
    {"stemi", "st elevation myocardial infarction"},
    {"nstemi", "non st elevation myocardial infarction"},
    {"mi", "myocardial infarction", "heart attack"},
    {"chf", "congestive heart failure", "heart failure"},
    {"dm", "diabetes mellitus", "diabetes"},
    {"htn", "hypertension"},
    {"copd", "chronic obstructive pulmonary disease"},
    {"dvt", "deep vein thrombosis"},
    {"pe embolism", "pulmonary embolism"},
    {"uti", "urinary tract infection"},
    {"gi", "gastrointestinal"},
    {"lmp", "last menstrual period"},
    {"fhx", "family history"},
    {"shx", "social history"},
    {"meds", "medications"},
    {"dc", "discharge"},
]


def _build_synonym_map() -> Dict[str, str]:
    """word -> canonical phrase, built from _SYNONYM_GROUPS. The first item
    in each group is used as the canonical form (order doesn't matter for
    matching correctness, it's just which spelling 'wins')."""
    mapping: Dict[str, str] = {}
    for group in _SYNONYM_GROUPS:
        canonical = sorted(group, key=len)[-1]  # prefer the fuller phrase
        for variant in group:
            mapping[variant] = canonical
    return mapping


_SYNONYM_MAP = _build_synonym_map()


def _expand_synonyms(s: str) -> str:
    """Replace any known abbreviation/synonym phrase in `s` with its
    canonical form, longest phrases first so multi-word variants (e.g.
    'chest x ray') aren't shadowed by a shorter one first."""
    for variant in sorted(_SYNONYM_MAP, key=len, reverse=True):
        canonical = _SYNONYM_MAP[variant]
        if variant == canonical:
            continue
        pattern = r"\b" + re.escape(variant) + r"\b"
        s = re.sub(pattern, canonical, s)
    return s


def _normalize_label(label: str) -> str:
    s = "".join(c.lower() for c in label if c.isalnum() or c.isspace())
    s = " ".join(s.split())
    s = _expand_synonyms(s)
    return s


def _tokens(label: str) -> Set[str]:
    return {w for w in _normalize_label(label).split() if w not in _STOPWORDS}


def _label_similarity(a: str, b: str) -> float:
    """0..1 similarity between two labels. A resident might write 'ECG' for
    the expert's '12-lead ECG within 10 min' — a plain string diff scores
    that low, so we also check keyword overlap. We use the Dice coefficient
    (2*overlap / total tokens) rather than overlap/min-length: min-length
    over-credits a short label just because all of its (few) tokens appear
    somewhere inside a much longer, different label — e.g. 'ECG' would
    otherwise match 'STEMI confirmed on ECG -> activate cath lab' at 100%,
    which is wrong; that's a distinct, much bigger action that merely
    mentions ECG in passing. Dice naturally favors matches where the two
    labels are close in size AND share tokens — the actual signal we want.
    This is still not true clinical-synonym matching (e.g. 'ECG' vs
    'electrocardiogram' as totally different strings won't match).
    """
    a_norm, b_norm = _normalize_label(a), _normalize_label(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    a_tok, b_tok = _tokens(a), _tokens(b)
    if not a_tok or not b_tok:
        # Degenerate case (label is all stopwords/punctuation/numbers) —
        # token overlap can't tell us anything, fall back to raw string
        # similarity as a last resort.
        return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    overlap = len(a_tok & b_tok)
    if not overlap:
        return 0.0
    return (2.0 * overlap) / (len(a_tok) + len(b_tok))


MATCH_THRESHOLD = 0.4  # calibrated for the Dice-based keyword score above


def _best_similarity_with_synonyms(student_label: str, expert_node: dict) -> float:
    """Best similarity between a student's label and an expert node,
    checking the node's own label AND any case-specific synonyms the
    author attached to it (e.g. 'Ticagrelor' / 'Brilinta' / 'Clopidogrel'
    all pointing at the same 'P2Y12 inhibitor' step)."""
    best = _label_similarity(student_label, expert_node["label"])
    for syn in expert_node.get("synonyms", []):
        best = max(best, _label_similarity(student_label, syn))
    return best


def _match_nodes(student_tree: dict, expert_tree: dict):
    """Greedy best-first matching between student and expert nodes by label
    similarity (not exact string equality). Returns (matched, missed, extra):
      matched — list of (student_node, expert_node, score) sorted by score desc
      missed  — expert nodes with no student match above threshold
      extra   — student nodes with no expert match above threshold
    """
    candidates = []
    for sn in student_tree["nodes"]:
        for en in expert_tree["nodes"]:
            score = _best_similarity_with_synonyms(sn["label"], en)
            if score >= MATCH_THRESHOLD:
                candidates.append((score, sn, en))
    candidates.sort(key=lambda c: c[0], reverse=True)

    used_s, used_e = set(), set()
    matched = []
    for score, sn, en in candidates:
        if sn["id"] in used_s or en["id"] in used_e:
            continue
        matched.append((sn, en, score))
        used_s.add(sn["id"])
        used_e.add(en["id"])

    missed = [en for en in expert_tree["nodes"] if en["id"] not in used_e]
    extra = [sn for sn in student_tree["nodes"] if sn["id"] not in used_s]
    return matched, missed, extra


def score_tree(student_tree: dict, expert_tree: dict,
               critical_node_labels: Optional[List[str]] = None) -> dict:
    """Score a student tree against the expert tree.

    Returns a dict with:
        total      : 0-100 final score
        critical   : 0-100 (50% weight) — critical-step inclusion
        order      : 0-100 (30% weight) — order correctness
        value      : 0-100 (20% weight) — value care
        diff       : detailed diff for the X-ray
        matched    : list of (student_node, expert_node) pairs
        missed     : expert nodes not in student tree
        extra      : student nodes not in expert tree
        order_issues: list of (a, b) where a should come before b but doesn't

    Matching is fuzzy on labels (keyword overlap + string similarity, see
    `_label_similarity`), so "ECG" reasonably matches "12-lead ECG within
    10 min" without needing an exact wording match.
    """
    matched_triples, missed_nodes, extra_nodes = _match_nodes(student_tree, expert_tree)
    matched = [(sn, en) for sn, en, _score in matched_triples]
    matched_expert_ids = {en["id"] for _sn, en, _score in matched_triples}
    match_scores = {en["id"]: score for _sn, en, score in matched_triples}

    # ── CRITICAL score (0..100) ─────────────────────────────────────────────
    # Tied to the SAME exclusive matching as everything else above, so one
    # vague student entry (e.g. "ECG") can't independently satisfy multiple
    # different critical requirements just because it's loosely related to
    # more than one of them — it can only ever be matched to the one expert
    # node it was actually assigned to.
    if critical_node_labels:
        crit_expert_ids = {
            en["id"] for en in expert_tree["nodes"]
            for crit_label in critical_node_labels
            if _normalize_label(en["label"]) == _normalize_label(crit_label)
        }
        hit = len(crit_expert_ids & matched_expert_ids)
        critical = round(100.0 * hit / max(1, len(critical_node_labels)), 1)
    else:
        critical = round(100.0 * len(matched) /
                         max(1, len(expert_tree["nodes"])), 1)

    # ── ORDER score (0..100) ────────────────────────────────────────────────
    # For each pair (a, b) in the expert tree such that a comes before b in
    # the expert's topological order, check that the same is true in the
    # student's tree. We only check pairs where both nodes are in the student
    # tree (so "missed" doesn't double-penalize).
    order_issues = []
    try:
        s_topo = topological_order(student_tree)
        s_pos = {nid: i for i, nid in enumerate(s_topo)}
        e_topo_ids = topological_order(expert_tree)
    except ValueError as e:
        # Student has a cycle — penalize heavily
        return {
            "total": 0.0, "critical": 0.0, "order": 0.0, "value": 0.0,
            "diff": {"error": str(e)},
            "matched": [], "missed": list(expert_tree["nodes"]),
            "extra": list(student_tree["nodes"]),
            "order_issues": [],
        }

    # Build the order check using matched STUDENT node pairs.
    expert_to_student = {en["id"]: sn for sn, en in matched}
    pairs_to_check = []
    for i, a in enumerate(e_topo_ids):
        for b in e_topo_ids[i + 1:]:
            if a in expert_to_student and b in expert_to_student:
                sa = expert_to_student[a]["id"]
                sb = expert_to_student[b]["id"]
                if (sa, sb) not in pairs_to_check:
                    pairs_to_check.append((sa, sb))

    correct_pairs = 0
    for sa, sb in pairs_to_check:
        if s_pos[sa] < s_pos[sb]:
            correct_pairs += 1
        else:
            sn_a = next(sn for sn in student_tree["nodes"] if sn["id"] == sa)
            sn_b = next(sn for sn in student_tree["nodes"] if sn["id"] == sb)
            order_issues.append((sn_a["label"], sn_b["label"]))

    order = round(100.0 * correct_pairs / max(1, len(pairs_to_check)), 1)

    # ── VALUE score (0..100) ───────────────────────────────────────────────
    # Penalize extra nodes (over-ordering). Perfect = 100, each extra costs
    # ~10 points. Floor at 0.
    n_extras = len(extra_nodes)
    value = max(0.0, 100.0 - 10.0 * n_extras)

    # ── Total ───────────────────────────────────────────────────────────────
    total = round(0.50 * critical + 0.30 * order + 0.20 * value, 1)

    diff = {
        "matched_count":  len(matched),
        "missed_count":   len(missed_nodes),
        "extra_count":    len(extra_nodes),
        "missed_labels":  sorted(n["label"] for n in missed_nodes),
        "extra_labels":   sorted(n["label"] for n in extra_nodes),
        "order_issues":   [list(p) for p in order_issues],
        "match_scores":   match_scores,
    }

    return {
        "total":      total,
        "critical":   critical,
        "order":      order,
        "value":      value,
        "diff":       diff,
        "matched":    matched,
        "missed":     missed_nodes,
        "extra":      extra_nodes,
        "order_issues": order_issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Vitals simulation (simple rule-based — placeholder for v2)
# ─────────────────────────────────────────────────────────────────────────────
class VitalsSim:
    """A simple time-evolving vitals model. Pure-Python, deterministic.

    The MVP doesn't aim to be physiologically accurate — it just gives the
    student the *feeling* of time pressure. If they delay a critical step
    past a threshold, vitals deteriorate. If they do it in time, vitals
    stabilize.
    """
    def __init__(self, initial: dict, critical_steps: List[dict]):
        """
        initial:         {hr, sbp, dbp, spo2, rr, temp}
        critical_steps:  [{label, done_at, target_by}, ...]
                         done_at = None until completed, target_by = seconds
                         within which this step must be done or vitals crash.
        """
        self.initial = dict(initial)
        self.current = dict(initial)
        self.critical_steps = critical_steps
        self.t = 0

    def tick(self, dt: float = 1.0):
        """Advance the simulation by dt seconds. Apply deterioration rules."""
        self.t += dt
        # Each un-done critical step past its target_by adds deterioration
        for step in self.critical_steps:
            if step["done_at"] is None and self.t > step["target_by"]:
                # Add a small penalty per second over
                over = self.t - step["target_by"]
                self.current["hr"]  = min(150, self.current["hr"]  + 0.5 * over * dt)
                self.current["sbp"] = max(40,  self.current["sbp"] - 0.4 * over * dt)
                self.current["spo2"]= max(70,  self.current["spo2"]- 0.1 * over * dt)

    def mark_done(self, label: str):
        for step in self.critical_steps:
            if step["label"] == label and step["done_at"] is None:
                step["done_at"] = self.t
                # On-time bonus: stabilize vitals toward initial
                self.current["hr"]  = (self.current["hr"]  + self.initial["hr"])  / 2
                self.current["sbp"] = (self.current["sbp"] + self.initial["sbp"]) / 2
                self.current["spo2"]= (self.current["spo2"]+ self.initial["spo2"])/ 2
                return True
        return False

    def snapshot(self) -> dict:
        return {"t": round(self.t, 1), **self.current}

    def outcome(self) -> str:
        """Did the patient survive stable, deteriorate, or crash?"""
        if self.current["sbp"] < 70 or self.current["spo2"] < 85:
            return "crash"
        if self.current["sbp"] < 90 or self.current["hr"] > 130:
            return "deteriorated"
        for step in self.critical_steps:
            if step["done_at"] is None:
                return "missed_critical_step"
        return "stable"


# ─────────────────────────────────────────────────────────────────────────────
# Case library
# ─────────────────────────────────────────────────────────────────────────────
def _build_expert_node(nid, ntype, label, rationale, t_added=0, result=None, synonyms=None):
    return {
        "id": nid, "type": ntype, "label": label,
        "rationale": rationale, "result": result,
        "t_added": t_added, "t_resolved": None,
        # Case-specific acceptable alternate phrasings this node's label
        # should ALSO match (e.g. brand vs generic drug names) — on top of,
        # not instead of, the built-in _SYNONYM_GROUPS dictionary above.
        "synonyms": synonyms or [],
    }


def case_stemi_58m() -> dict:
    """A 58-year-old male with inferior STEMI. Classic teaching case.

    Designed to be done in 5-8 minutes. The expert tree reflects ACC/AHA
    guideline-concordant care: ECG within 10 min, aspirin + P2Y12 within
    30 min, cath activation within 90 min, right-sided leads if indicated.

    Patient presentation: substernal chest pain at rest, diaphoretic, with
    inferior ST elevation on ECG. Realistic for a Lebanese university
    hospital ED.
    """
    expert_nodes = [
        _build_expert_node("n_001", "history",
            "Focused chest pain history",
            "Onset, character, radiation, associated symptoms — anchors the differential",
            t_added=15),
        _build_expert_node("n_002", "exam",
            "Vitals + cardiovascular exam",
            "Confirm hemodynamic stability, look for signs of CHF, murmurs",
            t_added=30),
        _build_expert_node("n_003", "imaging",
            "12-lead ECG within 10 min",
            "STEMI is an ECG diagnosis — every minute of delay = more myocardium",
            t_added=45, result="ST elevation II, III, aVF; reciprocal in I, aVL"),
        _build_expert_node("n_004", "lab",
            "Troponin + CBC + BMP + coags + type & screen",
            "Baseline before cath, rule out anemia/renal issues",
            t_added=60),
        _build_expert_node("n_005", "decision",
            "STEMI confirmed on ECG -> activate cath lab",
            "Door-to-balloon < 90 min is the standard",
            t_added=75),
        _build_expert_node("n_006", "treatment",
            "Aspirin 325 mg chewed",
            "Irreversible COX inhibition, mortality benefit",
            t_added=80),
        _build_expert_node("n_007", "treatment",
            "P2Y12 inhibitor loading (Ticagrelor 180mg or Clopidogrel 600mg)",
            "Dual antiplatelet before cath",
            t_added=100),
        _build_expert_node("n_008", "treatment",
            "Heparin per cath lab protocol",
            "Anticoagulation during PCI",
            t_added=120),
        _build_expert_node("n_009", "treatment",
            "Statin (Atorvastatin 80mg)",
            "Plaque stabilization, guideline-directed",
            t_added=130),
        _build_expert_node("n_010", "decision",
            "Right-sided ECG leads (V4R)",
            "Inferior STEMI: rule out RV involvement before giving nitrates",
            t_added=150, result="No RV ST elevation"),
        _build_expert_node("n_011", "disposition",
            "Transfer to cath lab",
            "Definitive treatment is PCI",
            t_added=180),
        _build_expert_node("n_012", "diagnosis",
            "Inferior STEMI, likely RCA occlusion",
            "Working diagnosis for the cath team",
            t_added=200),
    ]
    expert_edges = [
        {"from": "n_001", "to": "n_002"},
        {"from": "n_002", "to": "n_003"},
        {"from": "n_003", "to": "n_004"},
        {"from": "n_003", "to": "n_005"},
        {"from": "n_005", "to": "n_006"},
        {"from": "n_005", "to": "n_007"},
        {"from": "n_005", "to": "n_008"},
        {"from": "n_005", "to": "n_009"},
        {"from": "n_005", "to": "n_010"},
        {"from": "n_006", "to": "n_011"},
        {"from": "n_007", "to": "n_011"},
        {"from": "n_008", "to": "n_011"},
        {"from": "n_009", "to": "n_011"},
        {"from": "n_010", "to": "n_011"},
        {"from": "n_011", "to": "n_012"},
    ]

    return {
        "id":          "stemi_58m",
        "title":       "Mr. Khalil, 58M — Substernal chest pain × 90 min",
        "specialty":   "Internal Medicine / Cardiology",
        "difficulty":  "Intern",
        "estimated_min": 6,
        "vignette": (
            "Mr. Youssef Khalil, 58M, presents to the ED with substernal chest pain "
            "for the past 90 minutes. He describes it as a heavy pressure radiating "
            "to his jaw and left arm, associated with diaphoresis and nausea. He was "
            "at rest when it started. He has a history of hypertension, type 2 "
            "diabetes (on metformin), and is a 30-pack-year smoker. Father died of "
            "MI at 62.\n\n"
            "Medications: amlodipine 5mg, metformin 1g BID. No known drug allergies.\n\n"
            "Vitals on arrival: BP 92/60, HR 108, RR 22, SpO2 94% on room air, "
            "T 36.8°C. He is diaphoretic and anxious. Lungs clear, no JVD, no "
            "peripheral edema. Heart sounds regular, no murmurs. Abdomen soft."
        ),
        "questions_for_student": [
            "What is your immediate differential?",
            "What is the single most important test to do FIRST, and why?",
            "If the ECG shows ST elevation in II, III, aVF, what is your diagnosis "
            "and what activates the next phase of care?",
            "List the medications you would give in the first 30 minutes.",
            "What is the disposition?",
        ],
        "critical_steps": [
            {"label": "12-lead ECG within 10 min", "target_by": 120,
             "done_at": None},
            {"label": "Aspirin 325 mg chewed", "target_by": 240, "done_at": None},
            {"label": "STEMI confirmed on ECG -> activate cath lab",
             "target_by": 360, "done_at": None},
            {"label": "Transfer to cath lab", "target_by": 600, "done_at": None},
        ],
        "critical_node_labels": [
            "12-lead ECG within 10 min",
            "Aspirin 325 mg chewed",
            "STEMI confirmed on ECG -> activate cath lab",
            "Transfer to cath lab",
        ],
        "vitals_initial": {"hr": 108, "sbp": 92, "dbp": 60, "spo2": 94, "rr": 22},
        "expert_nodes": expert_nodes,
        "expert_edges": expert_edges,
        "case_source": {
            "type": "composite",
            "origin": "Authored teaching vignette, guideline-concordant (ACC/AHA STEMI pathway) — not sourced from a specific real chart",
            "deidentified": None,
            "consent_note": "N/A — synthetic case, no real patient involved",
        },
        # This is what a REAL case's follow_up looks like once you replace
        # this vignette with a de-identified chart. Left in as a worked
        # example — swap every field below with what actually happened to
        # the real patient once you have ethics/consent clearance.
        "follow_up": {
            "actual_path": (
                "EXAMPLE ONLY (not a real patient): ECG at 6 min confirmed inferior "
                "STEMI. Aspirin + ticagrelor given at 9 min. Cath lab activated at "
                "11 min, door-to-balloon 58 min. Angiography showed 100% proximal "
                "RCA occlusion, single DES placed with TIMI-3 flow restored."
            ),
            "complications": "EXAMPLE ONLY: Transient Mobitz I AV block during reperfusion, resolved without pacing.",
            "discharge_outcome": "EXAMPLE ONLY: Discharged day 3, EF 48% on discharge echo, ambulatory and asymptomatic.",
            "long_term": "EXAMPLE ONLY: 3-month follow-up — cardiac rehab completed, EF improved to 52%, back to work.",
            "teaching_pearl": (
                "EXAMPLE ONLY: Door-to-balloon time is the single biggest lever residents "
                "control directly — recognizing STEMI on the FIRST ECG and activating the "
                "cath lab before troponin even returns is what saved this patient's myocardium."
            ),
        },
    }


def case_placeholder_sepsis() -> dict:
    """Stub: 65F with fever, hypotension, AMS — sepsis bundle."""
    return {
        "id":          "sepsis_65f",
        "title":       "Mrs. Habib, 65F — Fever and confusion × 1 day",
        "specialty":   "Internal Medicine / Critical Care",
        "difficulty":  "Intern",
        "estimated_min": 7,
        "vignette": (
            "[Stub case — to be written from a real Lebanese hospital case. "
            "Brief: 65F with 1-day fever, altered mental status, BP 88/54, "
            "HR 118, suspected urinary source. Tests Surviving Sepsis bundle.]"
        ),
        "questions_for_student": [
            "What is your differential for AMS in this patient?",
            "What is the sepsis bundle? What is the time target?",
            "What is your source of infection workup?",
        ],
        "critical_steps": [
            {"label": "Lactate", "target_by": 120, "done_at": None},
            {"label": "Blood cultures before antibiotics", "target_by": 180,
             "done_at": None},
            {"label": "Broad-spectrum antibiotics", "target_by": 180,
             "done_at": None},
            {"label": "30 mL/kg crystalloid", "target_by": 180, "done_at": None},
        ],
        "critical_node_labels": [
            "Lactate", "Blood cultures before antibiotics",
            "Broad-spectrum antibiotics", "30 mL/kg crystalloid",
        ],
        "vitals_initial": {"hr": 118, "sbp": 88, "dbp": 54, "spo2": 93, "rr": 24},
        "expert_nodes": [], "expert_edges": [],
    }


def case_placeholder_abdominal_pain() -> dict:
    """Stub: 22F with RLQ pain — appendicitis vs. ovarian."""
    return {
        "id":          "abdopain_22f",
        "title":       "Sarah, 22F — Right lower quadrant pain × 18 hours",
        "specialty":   "Internal Medicine / General Surgery",
        "difficulty":  "Intern",
        "estimated_min": 6,
        "vignette": (
            "[Stub case — to be written. Brief: 22F with RLQ pain, anorexia, "
            "low-grade fever. Tests the workup of acute abdomen, the role of "
            "imaging, and the decision to involve surgery.]"
        ),
        "questions_for_student": [
            "What is your differential?",
            "What imaging would you order, and in what order?",
            "When do you involve surgery?",
        ],
        "critical_steps": [],
        "critical_node_labels": [],
        "vitals_initial": {"hr": 96, "sbp": 118, "dbp": 72, "spo2": 99, "rr": 16},
        "expert_nodes": [], "expert_edges": [],
    }


CASE_LIBRARY = {
    "stemi_58m":     case_stemi_58m,
    "sepsis_65f":    case_placeholder_sepsis,
    "abdopain_22f":  case_placeholder_abdominal_pain,
}


def list_all_case_ids() -> List[str]:
    """All case ids, from both the Python library and leader-authored JSON."""
    return list(CASE_LIBRARY.keys()) + list_json_case_ids()


def get_case(case_id: str) -> dict:
    if case_id in CASE_LIBRARY:
        case = CASE_LIBRARY[case_id]()
    elif case_id in list_json_case_ids():
        case = load_json_case_summary(case_id)
    else:
        raise ValueError(
            f"Unknown case '{case_id}'. Available: {', '.join(list_all_case_ids())}"
        )
    # Convert expert_nodes/edges list into a tree dict
    case["expert_tree"] = {
        "nodes": case["expert_nodes"],
        "edges": case["expert_edges"],
    }
    # Backward-compatible defaults for cases authored before v16
    # (case_source / follow_up). Any case function that doesn't set these
    # falls back to an explicit "not yet real" state rather than silently
    # pretending it has provenance/follow-up it doesn't have.
    case.setdefault("case_source", {
        "type": "composite",          # "real" | "published" | "composite" | "stub"
        "origin": "Authored teaching vignette (not sourced from a real chart)",
        "deidentified": None,
        "consent_note": "N/A — synthetic case, no real patient involved",
        "citation": None,             # only used when type == "published"
    })
    case.setdefault("follow_up", {
        "actual_path": None,
        "complications": None,
        "discharge_outcome": None,
        "long_term": None,
        "teaching_pearl": None,
    })
    return case


def new_published_case_template() -> dict:
    """Blank template for authoring a case from a PUBLISHED case report
    (PubMed / PMC / BMJ Case Reports / etc).

    This is a different provenance than `new_real_case_template()`: a
    published case report is already de-identified and consent-cleared by
    the journal before publication, so you don't need your own hospital's
    IRB sign-off to teach from it. What you DO need:

      1. Paraphrase the vignette/course in your own words. Do not copy the
         article's text verbatim — even a few sentences lifted directly is
         a copyright problem. Facts, findings, and the clinical timeline are
         fine to reuse; the original authors' exact sentences are not.
      2. Always keep the citation (PMID/DOI, journal, year) attached so
         students see where the real case came from — that's both good
         academic practice and your paper trail.
      3. If a figure/image from the article would help, don't embed it
         directly unless the journal is open-access with a license that
         permits reuse (many BMJ Case Reports are CC BY-NC) — check the
         specific article's license first.
    """
    return {
        "id":          "REPLACE_ME",
        "title":       "REPLACE_ME — age/sex, chief complaint",
        "specialty":   "REPLACE_ME",
        "difficulty":  "Intern | Resident | Senior",
        "estimated_min": 6,
        "vignette":    "REPLACE_ME — paraphrased presentation, HPI, PMH, meds, vitals, exam (in your own words, not copied from the article).",
        "questions_for_student": [
            "REPLACE_ME — 3-5 guiding questions, not graded",
        ],
        "critical_steps": [],
        "critical_node_labels": [],
        "vitals_initial": {"hr": 0, "sbp": 0, "dbp": 0, "spo2": 0, "rr": 0},
        "expert_nodes": [],
        "expert_edges": [],
        "case_source": {
            "type": "published",
            "origin": "REPLACE_ME — e.g. 'BMJ Case Reports' or 'PubMed / PMC'",
            "deidentified": True,
            "consent_note": "Published case report — de-identification and consent handled by the journal prior to publication",
            "citation": {
                "authors": "REPLACE_ME",
                "title": "REPLACE_ME — the article's own title",
                "journal": "REPLACE_ME",
                "year": "REPLACE_ME",
                "pmid": "REPLACE_ME",
                "doi": "REPLACE_ME",
                "url": "REPLACE_ME",
            },
        },
        "follow_up": {
            "actual_path": "REPLACE_ME — paraphrased from the article's case presentation/course section",
            "complications": "REPLACE_ME — or 'None reported'",
            "discharge_outcome": "REPLACE_ME — paraphrased from the article's outcome section",
            "long_term": "REPLACE_ME — paraphrased from the article's follow-up section, or 'Not reported'",
            "teaching_pearl": "REPLACE_ME — usually taken almost directly from the article's 'Learning points' section, still paraphrased",
        },
    }


def new_real_case_template() -> dict:
    """Blank template for authoring a case from a real, de-identified hospital
    chart. Copy this into a new `case_<name>()` function in CASE_LIBRARY.

    IMPORTANT — before using any real hospital case in this system:
      1. The case MUST be de-identified (no name, MRN, exact dates, or any
         of the 18 HIPAA-style identifiers — even for a Lebanese hospital,
         treat this as the minimum bar). Shift dates, rename the patient,
         round exact ages if the case is rare enough to be identifying.
      2. Check your hospital / LU ethics committee's policy on using patient
         cases for teaching software — many require IRB or ethics-committee
         sign-off even for de-identified teaching cases, especially if the
         tool is used across institutions or made publicly available.
      3. Keep a private, non-shared log of which real chart each case_id maps
         to (for audit purposes) — but NEVER put that mapping in this file
         or anywhere the app/student can see it.
    """
    return {
        "id":          "REPLACE_ME",
        "title":       "REPLACE_ME — age/sex, chief complaint",
        "specialty":   "REPLACE_ME",
        "difficulty":  "Intern | Resident | Senior",
        "estimated_min": 6,
        "vignette":    "REPLACE_ME — de-identified presentation, HPI, PMH, meds, vitals, exam.",
        "questions_for_student": [
            "REPLACE_ME — 3-5 guiding questions, not graded",
        ],
        "critical_steps": [
            # {"label": "...", "target_by": <seconds>, "done_at": None},
        ],
        "critical_node_labels": [
            # must match critical_steps labels, used for scoring
        ],
        "vitals_initial": {"hr": 0, "sbp": 0, "dbp": 0, "spo2": 0, "rr": 0},
        "expert_nodes": [
            # use _build_expert_node(...) for each step of the ideal workup
        ],
        "expert_edges": [
            # {"from": "n_001", "to": "n_002"}, ...
        ],
        "case_source": {
            "type": "real",
            "origin": "REPLACE_ME — e.g. 'LU-affiliated hospital, Internal Medicine, 2025'",
            "deidentified": True,
            "consent_note": "REPLACE_ME — ethics/IRB approval reference or waiver basis",
        },
        "follow_up": {
            "actual_path": "REPLACE_ME — what actually happened clinically, in sequence",
            "complications": "REPLACE_ME — or 'None' if the course was uncomplicated",
            "discharge_outcome": "REPLACE_ME — condition and disposition at discharge",
            "long_term": "REPLACE_ME — follow-up at 1/3/6 months if available, or 'Not available'",
            "teaching_pearl": "REPLACE_ME — the attending's key real-world takeaway from this exact case",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests (run with `python reasoning_canvas.py`)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing empty tree...")
    t = empty_tree()
    n1 = add_node(t, "history", "Chest pain history")
    n2 = add_node(t, "exam",    "CV exam")
    add_edge(t, n1["id"], n2["id"])
    print(f"  nodes={len(t['nodes'])} edges={len(t['edges'])}")

    print("\nTesting STEMI case scoring...")
    case = get_case("stemi_58m")
    # Build a "perfect" student tree = the expert tree
    student_perfect = {
        "nodes": [dict(n) for n in case["expert_tree"]["nodes"]],
        "edges": [dict(e) for e in case["expert_tree"]["edges"]],
    }
    result = score_tree(student_perfect, case["expert_tree"],
                        case.get("critical_node_labels"))
    print(f"  Perfect tree: total={result['total']} "
          f"(critical={result['critical']}, order={result['order']}, "
          f"value={result['value']})")
    assert result["total"] >= 95, f"Perfect tree should score ~100, got {result['total']}"

    # Build a "terrible" student tree: ECG is the only good thing, plus
    # shotgun tests and a wrong disposition. Critical steps all missed.
    student_bad = empty_tree()
    add_node(student_bad, "imaging", "ECG", "Check the heart", t_added=60)
    add_node(student_bad, "lab",     "CBC", "Routine labs", t_added=120)
    add_node(student_bad, "imaging", "CT chest", "Just in case", t_added=300)
    add_node(student_bad, "lab",     "CRP", "Inflammation", t_added=400)
    add_node(student_bad, "lab",     "ESR", "Also inflammation", t_added=500)
    add_node(student_bad, "disposition", "Discharge home with PPI",
             "Maybe GERD", t_added=600)
    result = score_tree(student_bad, case["expert_tree"],
                        case.get("critical_node_labels"))
    print(f"  Bad tree:     total={result['total']} "
          f"(critical={result['critical']}, order={result['order']}, "
          f"value={result['value']})")
    print(f"    missed: {result['diff']['missed_labels']}")
    print(f"    extras: {result['diff']['extra_labels']}")
    assert result["total"] < 60, (
        f"Bad tree should score low (only 1/4 critical steps loosely covered, "
        f"lots of extras), got {result['total']}"
    )
    # 'ECG' correctly earns credit for the '12-lead ECG within 10 min' critical
    # step (that's the whole point of fuzzy matching) but must NOT also get
    # credit for 'STEMI confirmed on ECG -> activate cath lab' just because
    # that label happens to mention ECG too — those are different actions.
    assert result["critical"] == 25.0, (
        f"Bad tree should have critical=25 (1 of 4 critical steps: ECG only, "
        f"aspirin/cath-activation/transfer all genuinely missing), "
        f"got {result['critical']}"
    )

    # Build a "right idea, wrong order" tree — student has ECG, aspirin,
    # and cath activation, but in the wrong sequence (e.g. cath before ECG).
    student_wrong_order = empty_tree()
    n1 = add_node(student_wrong_order, "decision",
                  "STEMI confirmed on ECG -> activate cath lab")
    n2 = add_node(student_wrong_order, "imaging",
                  "12-lead ECG within 10 min")
    n3 = add_node(student_wrong_order, "treatment", "Aspirin 325 mg chewed")
    add_edge(student_wrong_order, n1["id"], n2["id"])
    add_edge(student_wrong_order, n2["id"], n3["id"])
    result = score_tree(student_wrong_order, case["expert_tree"],
                        case.get("critical_node_labels"))
    print(f"  Wrong-order:  total={result['total']} "
          f"(critical={result['critical']}, order={result['order']}, "
          f"value={result['value']})")
    print(f"    order_issues: {result['diff']['order_issues']}")
    assert result["order"] < 100, (
        f"Wrong order should lose points, got {result['order']}"
    )
    assert len(result["order_issues"]) > 0, (
        f"Wrong order should produce order_issues, got "
        f"{result['order_issues']}"
    )

    # Vitals sim
    print("\nTesting VitalsSim...")
    sim = VitalsSim(
        initial={"hr": 108, "sbp": 92, "dbp": 60, "spo2": 94, "rr": 22},
        critical_steps=[
            {"label": "ECG", "target_by": 120, "done_at": None},
            {"label": "Aspirin", "target_by": 240, "done_at": None},
        ],
    )
    for _ in range(200):  # simulate 200s of delay
        sim.tick(1.0)
    print(f"  After 200s of nothing: {sim.snapshot()}  outcome={sim.outcome()}")
    assert sim.outcome() in ("crash", "deteriorated", "missed_critical_step")

    sim.mark_done("ECG")
    sim.mark_done("Aspirin")
    print(f"  After aspirin: {sim.snapshot()}  outcome={sim.outcome()}")

    print("\n✓ All smoke tests passed.")
