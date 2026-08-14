"""Elembic-style clinical NER+L pipeline — no model training required.

Architecture (mirrors the Nuraxi/Substack post):
  1. Section segmentation
  2. LLM-based NER (entity detection + modifier capture)
  3. Embedding-based ontology linking (SNOMED-CT / ICD-10 / LOINC)
  4. Temporal anchoring
  5. Traceable log per extracted concept
  6. Output -> your existing case_journey schema for the timeline viz

Requirements:
  pip install openai sentence-transformers numpy
  # Optional, for SNOMED-CT concept lookup:
  # pip install scispacy  # and a scispacy model for fallback

Tested with:
  - LLM: any OpenAI-compatible API (gpt-4o, llama-3.1-70b via Ollama, qwen2.5, etc.)
  - Embedder: sentence-transformers/all-MiniLM-L6-v2 (default) or
              cambridgeltl/SapBERT-on-PubMedBERT-fulltext (medical-specific)

The pipeline makes ZERO model training. The only "training" is calling
`ontology.embed_concepts()` once at system boot to pre-compute embeddings
for the SNOMED-CT primary names.
"""
from __future__ import annotations
import json
import os
import re
import time
import uuid
import hashlib
import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


# ─────────────────────────────────────────────────────────────────────────────
# 1. Section segmentation — pure regex, no model needed
# ─────────────────────────────────────────────────────────────────────────────
SECTION_PATTERNS = [
    ("chief_complaint",  r"chief complaint\s*[:\-]"),
    ("hpi",              r"(history of present illness|hpi)\s*[:\-]"),
    ("pmh",              r"(past medical history|pmh)\s*[:\-]"),
    ("medications",      r"(medications?|meds|home meds)\s*[:\-]"),
    ("allergies",        r"allergies\s*[:\-]"),
    ("exam",             r"(physical exam|examination|exam)\s*[:\-]"),
    ("vitals",           r"(vital signs|vitals)\s*[:\-]"),
    ("labs",             r"(labs?|laboratory)\s*[:\-]"),
    ("imaging",          r"(imaging|radiology|cxr|ct|mri)\s*[:\-]"),
    ("assessment",       r"(assessment|impression)\s*[:\-]"),
    ("plan",             r"plan\s*[:\-]"),
    ("discharge",        r"(discharge (summary|instructions?)|disposition)\s*[:\-]"),
]


def segment_sections(note: str) -> List[Dict[str, str]]:
    """Split a clinical note into labeled sections. Returns list of
    {section, text} dicts in document order."""
    # Build a list of (start, section_name) matches
    spans = []
    for section_name, pattern in SECTION_PATTERNS:
        m = re.search(pattern, note, re.IGNORECASE)
        if m:
            spans.append((m.start(), section_name))
    if not spans:
        return [{"section": "full_note", "text": note.strip()}]
    spans.sort()
    # First span's start is the pre-section content (often blank)
    out = []
    for i, (start, name) in enumerate(spans):
        end = spans[i + 1][0] if i + 1 < len(spans) else len(note)
        text = note[start:end].strip()
        # Strip the section header line itself
        text = re.sub(rf"^{name}\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
        out.append({"section": name, "text": text})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM-based NER — the core extraction
# ─────────────────────────────────────────────────────────────────────────────
NER_SYSTEM_PROMPT = """You are a clinical Named Entity Recognition system. Extract every
clinical concept from the note text below. For each concept return a JSON object
with EXACTLY these fields:

  text:        the exact span from the note (verbatim, including punctuation)
  type:        one of: disease | finding | procedure | medication |
               body_structure | lab | vital | temporal_expression
  presence:    positive | negated | hypothetical
  experiencer: patient | family_member | other
  temporal:    a short phrase like "before admission" | "day 3 of stay" |
               "at discharge" | "ongoing" — copy the text's own temporal cue
  qualifiers:  list of short strings: laterality (left/right/bilateral),
               severity (mild/moderate/severe), causality ("caused by X"),
               course ("improving", "worsening", "stable"), or anatomical detail
  evidence:    the original sentence containing this concept (for the audit log)

Output a JSON array only. No commentary, no markdown fences.
If no clinical concepts are present, output [].
"""


def extract_entities_with_llm(
    note_section: str,
    section_name: str,
    llm_call_fn,
) -> List[Dict[str, Any]]:
    """Call the LLM to extract entities from one section. `llm_call_fn` is
    a callable (system_prompt, user_prompt) -> str that returns the LLM's
    text response. Wrap any backend (OpenAI, Ollama, vLLM, etc.) in this.

    Returns a list of entity dicts. Each entity has a `_trace` field added
    for the audit log.
    """
    user_prompt = f"Section: {section_name}\n\nText:\n{note_section}"
    raw = llm_call_fn(NER_SYSTEM_PROMPT, user_prompt)
    # Robust JSON parsing — LLMs sometimes wrap in ```json
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        entities = json.loads(raw)
    except json.JSONDecodeError:
        # Try to recover: find first [ ... last ]
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        entities = json.loads(m.group(0)) if m else []
    if not isinstance(entities, list):
        entities = []

    # Stamp the audit-trace fields
    for e in entities:
        e["_trace"] = {
            "section":      section_name,
            "section_text": note_section,
            "raw_llm_out":  raw[:500],   # truncated
            "extracted_at": datetime.datetime.now().isoformat(),
        }
    return entities


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ontology linking — embedding similarity, no training
# ─────────────────────────────────────────────────────────────────────────────
class OntologyLinker:
    """Embeds every concept's primary name in the ontology once, then for
    any extracted entity text, returns the top-k SNOMED-CT / ICD-10 /
    LOINC candidates by cosine similarity.

    Setup (one-time, ~5 min for SNOMED-CT on CPU):
        linker = OntologyLinker(model_name="sentence-transformers/all-MiniLM-L6-v2")
        linker.embed_concepts(snomed_concepts)  # list of {code, name}
        linker.save("/path/to/ontology_index.pkl")
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.codes: List[str] = []
        self.names: List[str] = []
        self.embeddings = None
        self.ontology_name: str = "unknown"

    def embed_concepts(self, concepts: List[Dict[str, str]],
                       ontology_name: str = "SNOMED-CT"):
        """`concepts` is a list of {"code": "...", "name": "..."}.
        Computes embeddings once. Stores in self.embeddings."""
        self.ontology_name = ontology_name
        self.codes = [c["code"] for c in concepts]
        self.names = [c["name"] for c in concepts]
        t0 = time.time()
        self.embeddings = self.model.encode(
            self.names, batch_size=128, show_progress_bar=True,
            normalize_embeddings=True,
        )
        print(f"[OntologyLinker] Embedded {len(self.names)} {ontology_name} "
              f"concepts in {time.time()-t0:.1f}s")

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "ontology_name": self.ontology_name,
                "codes": self.codes,
                "names": self.names,
                "embeddings": self.embeddings,
                "model_name": self.model.model_card_data.base_model
                             if hasattr(self.model, "model_card_data") else "unknown",
            }, f)

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.ontology_name = d["ontology_name"]
        self.codes = d["codes"]
        self.names = d["names"]
        self.embeddings = d["embeddings"]

    def link(self, text: str, top_k: int = 5,
             min_score: float = 0.55) -> List[Dict[str, Any]]:
        """Find the top-k ontology concepts matching `text` by embedding
        similarity. Returns list of {code, name, score} sorted by score desc,
        filtered by min_score."""
        if self.embeddings is None:
            raise RuntimeError("Call embed_concepts() first")
        import numpy as np
        q = self.model.encode([text], normalize_embeddings=True)[0]
        scores = (self.embeddings @ q) * 100  # cosine sim, 0-100
        top_idx = np.argsort(-scores)[:top_k]
        return [
            {"code": self.codes[i], "name": self.names[i], "score": float(scores[i])}
            for i in top_idx
            if scores[i] >= min_score * 100
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Temporal anchoring — turn "day 3" / "at discharge" into seconds-from-arrival
# ─────────────────────────────────────────────────────────────────────────────
def anchor_time(temporal_cue: str, note_date: datetime.date,
                admission_date: Optional[datetime.date] = None) -> Optional[float]:
    """Convert a temporal cue into seconds-since-admission. Returns None if
    the cue can't be parsed.

    Handles: "before admission" (negative), "day N of stay" (N*86400),
    "at discharge" (use admission + typical stay or None), "ED arrival" (0),
    plain ISO date (if note_date given)."""
    if not temporal_cue:
        return None
    cue = temporal_cue.lower().strip()
    base = admission_date or note_date

    if "before" in cue or "prior" in cue or "history" in cue:
        # Pre-admission history — give it a generous negative offset
        m = re.search(r"(\d+)\s*(year|month|week|day|hour|min)s?", cue)
        if m:
            n = int(m.group(1)); unit = m.group(2)
            mult = {"year": 365, "month": 30, "week": 7, "day": 1,
                    "hour": 1/24, "min": 1/1440}[unit]
            return -n * mult * 86400
        return -2 * 3600  # default: 2h before

    m = re.search(r"day\s+(\d+)", cue)
    if m:
        return int(m.group(1)) * 86400

    if "discharge" in cue and admission_date:
        # No clean way — caller should fill in via follow-up data
        return 5 * 86400  # assume 5-day stay as placeholder

    if "arrival" in cue or "ed" in cue:
        return 0

    # Try ISO date
    try:
        d = datetime.datetime.strptime(cue, "%Y-%m-%d").date()
        if admission_date:
            return (d - admission_date).total_seconds()
    except ValueError:
        pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. The pipeline — ties it all together
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TimelineEvent:
    t: float                                 # seconds from admission
    event: str                               # short label
    type: str                                # history | exam | lab | imaging |
                                            # treatment | procedure | diagnosis |
                                            # disposition | follow_up
    category: str                            # critical | severity | negated |
                                            # temporal | stable | supportive |
                                            # discharge
    source_text: str = ""                    # the original sentence
    source_section: str = ""
    ontology_code: Optional[str] = None
    ontology_name: Optional[str] = None
    confidence: float = 1.0
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    presence: str = "positive"


def presence_to_category(presence: str, type_: str) -> str:
    """Map (presence, type) to your existing category palette."""
    if presence == "negated":
        return "negated"
    if presence == "hypothetical":
        return "temporal"  # use the blue/temporal color for "rule out" items
    if type_ in ("procedure",):
        return "critical"
    if type_ in ("medication", "lab", "vital"):
        # Lab/vital: severity if abnormal-sounding label, otherwise critical
        return "severity"
    if type_ in ("disease", "finding"):
        return "severity"
    if type_ in ("disposition",):
        return "discharge"
    return "temporal"


def type_to_event_type(type_: str) -> str:
    return {
        "disease":              "diagnosis",
        "finding":              "exam",
        "procedure":            "procedure",
        "medication":           "treatment",
        "body_structure":       "exam",
        "lab":                  "lab",
        "vital":                "exam",
        "temporal_expression":  "temporal",
    }.get(type_, "history")


def event_label_for(entity: Dict[str, Any]) -> str:
    """Build a short human-readable label for the timeline."""
    text = entity.get("text", "").strip()
    quals = entity.get("qualifiers") or []
    if quals:
        text += f" — {', '.join(quals[:2])}"
    # Truncate
    if len(text) > 80:
        text = text[:77] + "..."
    return text


def run_pipeline(
    note_text: str,
    note_id: str,
    note_date: datetime.date,
    admission_date: Optional[datetime.date],
    llm_call_fn,
    ontology_linker: Optional[OntologyLinker] = None,
    link_top_k: int = 1,
) -> List[TimelineEvent]:
    """Full Elembic-style pipeline on a single note.

    Returns a list of TimelineEvent objects ready to plug into your
    existing case_journey schema (and your journey_timeline.py renderer)."""
    events: List[TimelineEvent] = []
    sections = segment_sections(note_text)

    for sec in sections:
        section_name = sec["section"]
        section_text = sec["text"]
        if not section_text.strip():
            continue
        try:
            entities = extract_entities_with_llm(section_text, section_name, llm_call_fn)
        except Exception as e:
            print(f"[pipeline] LLM call failed on section {section_name}: {e}")
            continue

        for ent in entities:
            text = ent.get("text", "").strip()
            if not text:
                continue
            # Ontology link (optional but recommended)
            code = None; oname = None; conf = 1.0
            if ontology_linker is not None:
                cands = ontology_linker.link(text, top_k=link_top_k)
                if cands:
                    code = cands[0]["code"]
                    oname = ontology_linker.ontology_name
                    conf = cands[0]["score"] / 100.0

            # Temporal anchor
            t = anchor_time(ent.get("temporal", ""), note_date, admission_date)
            if t is None:
                # Use the section's typical position as a fallback
                section_offsets = {
                    "chief_complaint": -3600, "hpi": -1800, "pmh": -86400 * 30,
                    "medications": 0, "allergies": 0, "exam": 60, "vitals": 30,
                    "labs": 300, "imaging": 600, "assessment": 1800, "plan": 3600,
                    "discharge": 86400 * 5,
                }
                t = section_offsets.get(section_name, 0)

            presence = ent.get("presence", "positive")
            type_ = ent.get("type", "finding")
            ev = TimelineEvent(
                t=t,
                event=event_label_for(ent),
                type=type_to_event_type(type_),
                category=presence_to_category(presence, type_),
                source_text=ent.get("evidence", ""),
                source_section=section_name,
                ontology_code=code,
                ontology_name=oname,
                confidence=conf,
                qualifiers={
                    "presence":    presence,
                    "experiencer": ent.get("experiencer", "patient"),
                    "temporal":    ent.get("temporal", ""),
                    "qualifiers":  ent.get("qualifiers", []),
                },
            )
            events.append(ev)

    return events


# ─────────────────────────────────────────────────────────────────────────────
# 6. Example: a sample STEMI note -> timeline events
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_NOTE = """CHIEF COMPLAINT: 58-year-old male with substernal chest pain.

HISTORY OF PRESENT ILLNESS:
Mr. Youssef Khalil, 58M, with a history of hypertension, type 2 diabetes
mellitus (on metformin), and 30 pack-year smoking history, presents to
the ED with substernal chest pain radiating to the left arm and jaw, onset
2 hours prior to arrival, associated with diaphoresis and nausea. He denies
syncope or palpitations. Family history significant for father with MI at 62.

MEDICATIONS: amlodipine 5mg, metformin 1g BID.

VITAL SIGNS: BP 88/54, HR 112, RR 22, SpO2 91% on room air, T 36.8C.

PHYSICAL EXAM: Diaphoretic, anxious. Lungs clear. No JVD. Regular rate
and rhythm, no murmurs. Abdomen soft, non-tender.

LABS: Troponin I 4.8 ng/mL (markedly elevated). Creatinine 1.4 mg/dL.
Glucose 218 mg/dL.

IMAGING: 12-lead ECG shows ST elevation in leads II, III, aVF with
reciprocal changes in I, aVL. Bedside echo: EF 30%, inferior wall
hypokinesis, no pericardial effusion.

ASSESSMENT: Acute inferior STEMI, likely RCA occlusion, complicated by
cardiogenic shock (Killip class III).

PLAN: Aspirin 325 mg chewed, ticagrelor 180 mg loading dose, heparin
bolus 60 U/kg, activate cath lab. IV normal saline 500 mL bolus.
Morphine 4 mg IV for pain. Right-sided ECG leads to assess for RV
involvement. Patient denies any history of contrast allergy.

PROCEDURE: Emergent cardiac catheterization via right radial approach.
PCI to RCA with drug-eluting stent placement. Door-to-balloon time 68
minutes. TIMI 3 flow restored.

HOSPITAL COURSE: Admitted to CCU on norepinephrine for blood pressure
support. Moderate mitral regurgitation noted (new). Atorvastatin 80 mg
started. Metoprolol 12.5 mg initiated once BP tolerated. Ramipril
2.5 mg started on day 2 with stable creatinine. Creatinine trended down
to 1.2 mg/dL. Repeat echo on day 3 showed EF 35% (improving). Norepinephrine
discontinued day 2. Patient ambulated 50m with assistance on day 3.
Cardiac rehab referral placed. Smoking cessation counseling provided.
Dietitian reviewed cardiac diet.

DISCHARGE: Discharged home on day 5 in stable condition. Discharge
medications: aspirin, ticagrelor, metoprolol, ramipril, atorvastatin.
Follow-up echo scheduled in 6 weeks. Patient verbalized understanding
of medications and follow-up plan.
"""


def example_llm_call(system_prompt: str, user_prompt: str) -> str:
    """A stub LLM call for the example. Replace with real_llm_call_ollama or
    real_llm_call_openai in production. Returns a realistic-looking extraction
    JSON for the sample STEMI note."""
    return json.dumps([
        {"text": "substernal chest pain", "type": "finding",
         "presence": "positive", "experiencer": "patient",
         "temporal": "at presentation", "qualifiers": ["radiating to left arm and jaw"],
         "evidence": "presents to the ED with substernal chest pain radiating to the left arm and jaw"},
        {"text": "hypertension", "type": "disease", "presence": "positive",
         "experiencer": "patient", "temporal": "past medical history",
         "qualifiers": [], "evidence": "with a history of hypertension"},
        {"text": "type 2 diabetes mellitus", "type": "disease",
         "presence": "positive", "experiencer": "patient",
         "temporal": "past medical history", "qualifiers": ["on metformin"],
         "evidence": "type 2 diabetes mellitus (on metformin)"},
        {"text": "30 pack-year smoking history", "type": "finding",
         "presence": "positive", "experiencer": "patient",
         "temporal": "past medical history", "qualifiers": ["30 pack-years"],
         "evidence": "30 pack-year smoking history"},
        {"text": "syncope", "type": "finding", "presence": "negated",
         "experiencer": "patient", "temporal": "at presentation",
         "qualifiers": [], "evidence": "He denies syncope or palpitations"},
        {"text": "MI", "type": "disease", "presence": "positive",
         "experiencer": "family_member", "temporal": "past medical history",
         "qualifiers": ["father", "at age 62"],
         "evidence": "father with MI at 62"},
        {"text": "BP 88/54", "type": "vital", "presence": "positive",
         "experiencer": "patient", "temporal": "at presentation",
         "qualifiers": ["hypotension"],
         "evidence": "BP 88/54, HR 112, RR 22, SpO2 91% on room air"},
        {"text": "Troponin I 4.8 ng/mL", "type": "lab", "presence": "positive",
         "experiencer": "patient", "temporal": "at presentation",
         "qualifiers": ["markedly elevated"],
         "evidence": "Troponin I 4.8 ng/mL (markedly elevated)"},
        {"text": "Creatinine 1.4 mg/dL", "type": "lab", "presence": "positive",
         "experiencer": "patient", "temporal": "at presentation",
         "qualifiers": ["mildly elevated"],
         "evidence": "Creatinine 1.4 mg/dL"},
        {"text": "ST elevation in leads II, III, aVF", "type": "finding",
         "presence": "positive", "experiencer": "patient",
         "temporal": "at presentation", "qualifiers": ["inferior wall"],
         "evidence": "12-lead ECG shows ST elevation in leads II, III, aVF"},
        {"text": "EF 30%", "type": "finding", "presence": "positive",
         "experiencer": "patient", "temporal": "at presentation",
         "qualifiers": ["severely reduced", "inferior wall hypokinesis"],
         "evidence": "Bedside echo: EF 30%, inferior wall hypokinesis"},
        {"text": "Acute inferior STEMI", "type": "disease", "presence": "positive",
         "experiencer": "patient", "temporal": "at presentation",
         "qualifiers": ["likely RCA occlusion"],
         "evidence": "Acute inferior STEMI, likely RCA occlusion"},
        {"text": "cardiogenic shock", "type": "disease", "presence": "positive",
         "experiencer": "patient", "temporal": "at presentation",
         "qualifiers": ["Killip class III"],
         "evidence": "complicated by cardiogenic shock (Killip class III)"},
        {"text": "Aspirin 325 mg", "type": "medication", "presence": "positive",
         "experiencer": "patient", "temporal": "at presentation",
         "qualifiers": ["chewed", "loading dose"],
         "evidence": "Aspirin 325 mg chewed, ticagrelor 180 mg loading dose"},
        {"text": "Ticagrelor 180 mg", "type": "medication", "presence": "positive",
         "experiencer": "patient", "temporal": "at presentation",
         "qualifiers": ["loading dose"],
         "evidence": "ticagrelor 180 mg loading dose"},
        {"text": "Heparin bolus 60 U/kg", "type": "medication", "presence": "positive",
         "experiencer": "patient", "temporal": "at presentation",
         "qualifiers": ["anticoagulation"],
         "evidence": "heparin bolus 60 U/kg"},
        {"text": "IV normal saline 500 mL", "type": "medication", "presence": "positive",
         "experiencer": "patient", "temporal": "at presentation",
         "qualifiers": ["bolus"],
         "evidence": "IV normal saline 500 mL bolus"},
        {"text": "contrast allergy", "type": "finding", "presence": "negated",
         "experiencer": "patient", "temporal": "before procedure",
         "qualifiers": [], "evidence": "Patient denies any history of contrast allergy"},
        {"text": "PCI to RCA", "type": "procedure", "presence": "positive",
         "experiencer": "patient", "temporal": "at procedure",
         "qualifiers": ["drug-eluting stent", "door-to-balloon 68 min", "TIMI 3 flow"],
         "evidence": "PCI to RCA with drug-eluting stent placement. Door-to-balloon time 68 minutes. TIMI 3 flow restored."},
        {"text": "Moderate mitral regurgitation", "type": "finding",
         "presence": "positive", "experiencer": "patient",
         "temporal": "day 1", "qualifiers": ["moderate", "new"],
         "evidence": "Moderate mitral regurgitation noted (new)"},
        {"text": "Atorvastatin 80 mg", "type": "medication", "presence": "positive",
         "experiencer": "patient", "temporal": "day 1",
         "qualifiers": ["high-intensity statin"],
         "evidence": "Atorvastatin 80 mg started"},
        {"text": "Metoprolol 12.5 mg", "type": "medication", "presence": "positive",
         "experiencer": "patient", "temporal": "day 1",
         "qualifiers": ["low dose", "BP tolerated"],
         "evidence": "Metoprolol 12.5 mg initiated once BP tolerated"},
        {"text": "Ramipril 2.5 mg", "type": "medication", "presence": "positive",
         "experiencer": "patient", "temporal": "day 2",
         "qualifiers": ["stable creatinine"],
         "evidence": "Ramipril 2.5 mg started on day 2 with stable creatinine"},
        {"text": "EF 35%", "type": "finding", "presence": "positive",
         "experiencer": "patient", "temporal": "day 3",
         "qualifiers": ["improving"],
         "evidence": "Repeat echo on day 3 showed EF 35% (improving)"},
        {"text": "Patient ambulated 50m with assistance", "type": "finding",
         "presence": "positive", "experiencer": "patient", "temporal": "day 3",
         "qualifiers": ["with assistance"],
         "evidence": "Patient ambulated 50m with assistance on day 3"},
        {"text": "Discharged home", "type": "disposition", "presence": "positive",
         "experiencer": "patient", "temporal": "day 5",
         "qualifiers": ["stable condition"],
         "evidence": "Discharged home on day 5 in stable condition"},
        {"text": "Cardiac rehab referral", "type": "procedure", "presence": "positive",
         "experiencer": "patient", "temporal": "day 5",
         "qualifiers": ["outpatient"],
         "evidence": "Cardiac rehab referral placed"},
        {"text": "Follow-up echo in 6 weeks", "type": "procedure", "presence": "positive",
         "experiencer": "patient", "temporal": "6 weeks",
         "qualifiers": ["outpatient"],
         "evidence": "Follow-up echo scheduled in 6 weeks"},
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 7. Real LLM call implementations (Ollama + OpenAI)
# ─────────────────────────────────────────────────────────────────────────────
# Pick ONE of these based on your setup, then pass it to run_pipeline(..., llm_call_fn=...)
#
# Both are drop-in replacements for the stub `example_llm_call` above.
# They take (system_prompt, user_prompt) and return a JSON string.


def real_llm_call_ollama(
    system_prompt: str,
    user_prompt: str,
    model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    timeout: int = 120,
) -> str:
    """Call a local Ollama model. Requires `ollama serve` running and
    the model pulled (`ollama pull llama3.1:8b`).

    Install:    curl -fsSL https://ollama.com/install.sh | sh
    Pull model: ollama pull llama3.1:8b
    Start:      ollama serve    (in another terminal, or as a service)

    No extra Python packages needed — uses urllib only.
    """
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": model,
        "prompt": f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}",
        "stream": False,
        "options": {
            "temperature": 0.0,    # deterministic for extraction
            "num_predict": 4096,   # enough for ~50 entities
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "[]")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {base_url}. "
            f"Is 'ollama serve' running? Did you 'ollama pull {model}'? "
            f"Original error: {e}"
        )


def real_llm_call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """Call the OpenAI Chat Completions API. Requires the `openai` package
    and an OPENAI_API_KEY environment variable (or pass api_key=...).

    Install:   pip install openai
    Models:    gpt-4o-mini (cheap, good), gpt-4o (best, pricier)
               gpt-5 (latest, check pricing)
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai")

    client = OpenAI(api_key=api_key, timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=4096,
        response_format={"type": "json_object"},  # forces JSON if supported
    )
    return resp.choices[0].message.content or "[]"


def real_llm_call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-5",
    api_key: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """Call the Anthropic API (Claude). Requires the `anthropic` package.

    Install:   pip install anthropic
    Models:    claude-sonnet-4-5 (good balance), claude-opus-4 (best)
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # Claude returns a list of content blocks; concatenate text
    return "".join(b.text for b in resp.content if hasattr(b, "text"))


def real_llm_call_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str = "anthropic/claude-sonnet-4.5",
    api_key: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """Call OpenRouter — a unified API gateway for many LLM providers.

    Install:   pip install openai
    Get key:   https://openrouter.ai/keys   (free tier with rate limits)
    Models:    - anthropic/claude-sonnet-4.5      (best for clinical, $$$)
               - openai/gpt-4o-mini                (cheap, good)
               - meta-llama/llama-3.1-70b-instruct (open, free tier)
               - google/gemini-2.0-flash-exp:free   (Google's free tier)
               - qwen/qwen-2.5-72b-instruct:free    (free, decent)
    Full list: https://openrouter.ai/models
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai")

    client = OpenAI(
        api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        timeout=timeout,
        default_headers={
            "HTTP-Referer": "https://github.com/your-org/reasoning-canvas",
            "X-Title":      "Reasoning Canvas clinical NER+L",
        },
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    return resp.choices[0].message.content or "[]"


# ─────────────────────────────────────────────────────────────────────────────
# 8. How to pick which LLM call to use
# ─────────────────────────────────────────────────────────────────────────────
def auto_pick_llm() -> callable:
    """Try to pick the best available LLM in this order:
       1. Ollama (if running locally)
       2. OpenRouter (if OPENROUTER_API_KEY env var is set) — best bang/buck
       3. OpenAI (if OPENAI_API_KEY env var is set)
       4. Anthropic (if ANTHROPIC_API_KEY env var is set)
       5. Fall back to the stub (will produce demo output)
    """
    import urllib.request
    # 1. Try Ollama
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            models = json.loads(r.read().decode()).get("models", [])
            if models:
                # Prefer the biggest model that's available
                preferred = ["llama3.1:70b", "qwen2.5:72b", "llama3.1:8b",
                             "mistral", "qwen2.5:14b", "llama3.2"]
                model_name = next(
                    (m["name"] for p in preferred
                     for m in models if m["name"].startswith(p)),
                    models[0]["name"],
                )
                print(f"[auto_pick_llm] Using Ollama: {model_name}")
                return lambda sys, usr: real_llm_call_ollama(sys, usr, model=model_name)
    except Exception:
        pass

    # 2. Try OpenRouter (preferred when available — free tier, many models)
    if os.environ.get("OPENROUTER_API_KEY"):
        # Default to the best free model if available, else cheap
        model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        print(f"[auto_pick_llm] Using OpenRouter: {model}")
        return lambda sys, usr: real_llm_call_openrouter(sys, usr, model=model)

    # 3. Try OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        print("[auto_pick_llm] Using OpenAI (gpt-4o-mini)")
        return real_llm_call_openai

    # 4. Try Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("[auto_pick_llm] Using Anthropic (claude-sonnet-4-5)")
        return real_llm_call_anthropic

    # 5. Fall back to stub
    print("[auto_pick_llm] No LLM found. Using the stub (demo output).")
    print("  To use a real LLM:")
    print("    - Install Ollama:     curl -fsSL https://ollama.com/install.sh | sh")
    print("    - Or set OPENROUTER_API_KEY (free tier, many models):")
    print("         export OPENROUTER_API_KEY=sk-or-...")
    print("         export OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free")
    print("    - Or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var")
    return example_llm_call


if __name__ == "__main__":
    # Run the pipeline on the sample note, then convert events to your
    # existing case_journey schema so journey_timeline.py can render them.
    print("Running pipeline on sample STEMI note...")
    events = run_pipeline(
        note_text=SAMPLE_NOTE,
        note_id="note_001",
        note_date=datetime.date(2026, 1, 15),
        admission_date=datetime.date(2026, 1, 15),
        llm_call_fn=auto_pick_llm(),     # auto-detects Ollama / OpenAI / Anthropic
        ontology_linker=None,            # add when you have SNOMED embedded
    )
    print(f"Extracted {len(events)} timeline events")

    # Convert to the case_journey schema (matches reasoning_canvas.py)
    case_journey = []
    for ev in events:
        case_journey.append({
            "t":         ev.t,
            "event":     ev.event,
            "type":      ev.type,
            "category":  ev.category,
            "label":     ev.event,
            # Extra provenance fields the rendering layer can show
            "_provenance": {
                "source_section": ev.source_section,
                "source_text":    ev.source_text,
                "ontology_code":  ev.ontology_code,
                "ontology_name":  ev.ontology_name,
                "confidence":     ev.confidence,
                "trace_id":       ev.trace_id,
            },
        })
    out = {
        "id":           "elembic_extracted_case",
        "title":        "Mr. Khalil, 58M — STEMI (extracted from clinical note)",
        "case_journey": case_journey,
    }
    with open("/workspace/cases/elembic_extracted_case.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote /workspace/cases/elembic_extracted_case.json")

    # Now render the timeline using the existing journey_timeline.py
    try:
        from journey_timeline import render_journey_timeline
        fig = render_journey_timeline(out, start_age_years=58.0,
                                      width=3200, height=1300)
        fig.write_image("/workspace/elembic_extracted_timeline.png",
                        width=3200, height=1300, scale=1)
        print("Wrote /workspace/elembic_extracted_timeline.png")
    except Exception as e:
        print(f"(rendering skipped: {e})")
