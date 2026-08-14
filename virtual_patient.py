"""Virtual Patient Simulator — turn a real patient into an LLM character.

Architecture (no model training):
  1. COLLECT — gather free-text notes + structured data for the patient
  2. INDEX — chunk + embed with sentence-transformers, store in FAISS
  3. SIMULATE — at inference: retrieve relevant chunks, inject into prompt,
                 call LLM, get in-character answer

The LLM is NOT retrained per patient. It just receives the patient's actual
data as context and uses its pre-existing medical + language skills to
respond AS the patient.

The "training" you wanted is actually RAG (retrieval-augmented generation).
This is the same approach Elembic/Substack uses — they explicitly state
"we do not need to train the models."
"""
from __future__ import annotations
import json
import re
import os
import datetime
from typing import List, Dict, Any, Optional, Tuple

# Optional imports — these are required at runtime but not at import time
# so the module loads even if deps aren't installed yet.
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Patient record — the source of truth for a real patient
# ─────────────────────────────────────────────────────────────────────────────
PATIENT_RECORD_SCHEMA = {
    "patient_id":         "anonymized_patient_001",   # never the real MRN
    "age_at_admission":   58,
    "sex":                "M",
    "admission_date":     "2026-01-15",
    "discharge_date":     "2026-01-20",
    "primary_diagnosis":  "Acute inferior STEMI",

    # The full free-text record (any combination of these)
    "free_text_notes": [
        "H&P note: 58M with crushing substernal chest pain × 90 min, "
        "started while watching TV, radiates to left arm and jaw, 8/10 severity, "
        "associated with diaphoresis and nausea. PMH: HTN, T2DM. Meds: amlodipine, "
        "metformin. Father MI at 62. Smoker 30 pack-years.",
        "Nursing note: Patient anxious, diaphoretic. Pain 8/10 on arrival. "
        "Aspirin 325 mg given at 04:35 chewed. Pain decreased to 6/10.",
        "Procedure note: Emergent cardiac cath via right radial approach. "
        "PCI to RCA with drug-eluting stent. Door-to-balloon 68 min. TIMI 3 flow.",
        "Discharge summary: Patient discharged home on day 5. Discharged on "
        "aspirin, ticagrelor, metoprolol, ramipril, atorvastatin. Follow-up "
        "echo in 6 weeks. Patient verbalized understanding of medications.",
    ],

    # Original Q&A from the actual encounter (what the patient told the doctor)
    "original_qa": [
        {"q": "When did the pain start?", "a": "About an hour and a half ago, "
         "while I was watching television."},
        {"q": "Does the pain radiate?", "a": "Yes, to my left arm and jaw."},
        {"q": "How severe is the pain?", "a": "8 out of 10. The worst pain I've "
         "ever had in my life."},
        {"q": "Any associated symptoms?", "a": "I'm sweating a lot and feel "
         "nauseous. I almost threw up."},
        {"q": "Have you had this before?", "a": "No, never anything like this."},
        {"q": "Any medical conditions?", "a": "I have high blood pressure and "
         "diabetes. Take pills for both."},
        {"q": "What medications?", "a": "Amlodipine 5 mg once a day, and "
         "metformin 1000 mg twice a day."},
        {"q": "Any drug allergies?", "a": "No, none that I know of."},
        {"q": "Family history?", "a": "My father died of a heart attack at 62."},
        {"q": "Do you smoke?", "a": "About a pack a day for 30 years."},
        {"q": "When can you go home?", "a": "They said I can go home tomorrow. "
         "I have to take these new pills every day."},
    ],

    # Optional: lab values, vitals, exam findings (structured)
    "structured": {
        "vitals": [
            {"t": 0,    "hr": 108, "sbp": 92, "dbp": 60, "spo2": 94, "rr": 22},
            {"t": 3600, "hr": 88,  "sbp": 102, "dbp": 64, "spo2": 97, "rr": 18},
        ],
        "labs": [
            {"test": "Troponin I", "value": "4.8 ng/mL", "at": 60,  "flag": "critical"},
            {"test": "Creatinine", "value": "1.4 mg/dL",  "at": 60,  "flag": "abnormal"},
            {"test": "EF (echo)",  "value": "30%",        "at": 150, "flag": "critical"},
        ],
        "medications_administered": [
            {"drug": "Aspirin",     "dose": "325 mg", "route": "PO chewed", "at": 80},
            {"drug": "Ticagrelor",  "dose": "180 mg", "route": "PO",        "at": 100},
            {"drug": "Heparin",     "dose": "60 U/kg","route": "IV bolus",  "at": 120},
            {"drug": "Atorvastatin","dose": "80 mg",  "route": "PO",        "at": 300},
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chunking — split the free-text record into retrievable chunks
# ─────────────────────────────────────────────────────────────────────────────
def chunk_patient_record(record: Dict[str, Any],
                         chunk_size: int = 500,
                         overlap: int = 50) -> List[Dict[str, Any]]:
    """Split the patient's free-text notes into overlapping chunks. Each
    chunk keeps metadata (note type, date, source) so we can filter later."""
    chunks = []
    for i, note in enumerate(record.get("free_text_notes", [])):
        # Try to extract a section header from the first sentence
        first_sentence = note.split(".")[0]
        section = first_sentence.split(":")[0] if ":" in first_sentence else f"note_{i}"

        # Split into sentences for chunking
        sentences = re.split(r"(?<=[.!?])\s+", note)
        current_chunk = []
        current_len = 0
        for sent in sentences:
            if current_len + len(sent) > chunk_size and current_chunk:
                chunks.append({
                    "text":       " ".join(current_chunk),
                    "source":     "free_text_note",
                    "section":    section,
                    "note_idx":   i,
                    "patient_id": record.get("patient_id"),
                })
                # Keep overlap
                current_chunk = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_len = sum(len(s) for s in current_chunk)
            current_chunk.append(sent)
            current_len += len(sent)
        if current_chunk:
            chunks.append({
                "text":       " ".join(current_chunk),
                "source":     "free_text_note",
                "section":    section,
                "note_idx":   i,
                "patient_id": record.get("patient_id"),
            })

    # Also add original Q&A as chunks (highest signal)
    for qa in record.get("original_qa", []):
        chunks.append({
            "text":       f"Q: {qa['q']}\nA: {qa['a']}",
            "source":     "original_qa",
            "section":    "patient_dialogue",
            "patient_id": record.get("patient_id"),
        })

    # Also add structured data as a chunk
    structured = record.get("structured", {})
    if structured.get("medications_administered"):
        med_lines = [f"- {m['drug']} {m['dose']} {m['route']} at t={m['at']}s"
                     for m in structured["medications_administered"]]
        chunks.append({
            "text":       "Medications administered during admission:\n"
                          + "\n".join(med_lines),
            "source":     "structured_data",
            "section":    "medications",
            "patient_id": record.get("patient_id"),
        })
    if structured.get("labs"):
        lab_lines = [f"- {l['test']}: {l['value']} (at t={l['at']}s, {l['flag']})"
                     for l in structured["labs"]]
        chunks.append({
            "text":       "Lab results:\n" + "\n".join(lab_lines),
            "source":     "structured_data",
            "section":    "labs",
            "patient_id": record.get("patient_id"),
        })

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3. Index — embed + store the chunks (one-time per patient)
# ─────────────────────────────────────────────────────────────────────────────
class PatientMemory:
    """A per-patient retrievable memory. Stores chunked text + embeddings
    so we can answer questions about this specific patient.

    No model training. Just an embedding model + a small vector index.
    """
    def __init__(self, patient_id: str, model_name: str = None):
        if not HAS_EMBEDDINGS:
            raise RuntimeError(
                "pip install sentence-transformers numpy"
            )
        self.patient_id = patient_id
        # Default: small, fast, no medical-specific training
        if model_name is None:
            # Try SapBERT for medical, fall back to MiniLM
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings = None  # numpy array, shape (n_chunks, dim)

    def index_record(self, record: Dict[str, Any]) -> int:
        """Chunk and embed a patient record. Returns the number of chunks
        indexed. ~5 min on CPU, ~30 sec on GPU for a typical 50-page chart."""
        chunks = chunk_patient_record(record)
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=False,
            normalize_embeddings=True,
        )
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3,
                 min_score: float = 0.3) -> List[Tuple[Dict[str, Any], float]]:
        """Find the top-k chunks most relevant to the query. Returns list
        of (chunk, similarity_score) sorted by score desc."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = (self.embeddings @ q_emb) * 100  # cosine similarity, 0-100
        top_idx = np.argsort(-scores)[:top_k]
        return [
            (self.chunks[i], float(scores[i]))
            for i in top_idx
            if scores[i] >= min_score * 100
        ]

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "patient_id":  self.patient_id,
                "chunks":      self.chunks,
                "embeddings":  self.embeddings,
            }, f)

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.patient_id = d["patient_id"]
        self.chunks = d["chunks"]
        self.embeddings = d["embeddings"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. The simulator — turn the LLM into THIS specific patient
# ─────────────────────────────────────────────────────────────────────────────
def build_patient_system_prompt(record: Dict[str, Any],
                                retrieved_chunks: List[Tuple[Dict, float]]) -> str:
    """Build the system prompt that turns the LLM into this specific patient.

    Key components:
      - Persona (age, sex, demeanor)
      - Retrieved relevant context from the chart
      - Instructions for in-character response
    """
    age = record.get("age_at_admission", "?")
    sex = record.get("sex", "?")
    diagnosis = record.get("primary_diagnosis", "?")

    # Build context block from retrieved chunks
    context_lines = []
    for chunk, score in retrieved_chunks:
        src = chunk.get("source", "chart")
        sec = chunk.get("section", "")
        context_lines.append(f"--- ({src} / {sec}, relevance {score:.0f}%) ---")
        context_lines.append(chunk["text"])
        context_lines.append("")
    context_block = "\n".join(context_lines) if context_lines else "(no relevant context found)"

    system = f"""You are a simulated patient in a clinical training scenario. You
are a {age}-year-old {sex} who was admitted with {diagnosis}.

CRITICAL INSTRUCTIONS:
- Stay in character as this patient. Use first person ("I", "my").
- Be realistic. Don't volunteer information the patient wouldn't naturally offer.
  If the resident doesn't ask the right question, they don't get the answer.
- For common HPI questions, answer naturally using the context below.
- For questions about your chart (medications, labs), use ONLY what's in your context.
  If you don't have that information, say "I don't remember" or "I'd have to ask the nurse."
- Never break character. Never reveal you're an AI. Never give medical advice.
- Match the patient's likely demeanor (anxious, in pain, short answers if in distress).

YOUR MEDICAL RECORD (retrieved based on the current question):
{context_block}

If a question is completely unrelated to your case, say "I'm not sure what you mean"
or "I don't think that's relevant to why I'm here."

If a question is about something not in the context above, say "I don't remember
the details" or "The doctor would know that better than me."
"""
    return system


def simulate_patient(
    record: Dict[str, Any],
    memory: PatientMemory,
    question: str,
    llm_call_fn,
    top_k: int = 3,
) -> Dict[str, Any]:
    """The main simulator function. Takes a question from the resident,
    retrieves relevant context from the patient's memory, and calls the
    LLM to generate an in-character answer.

    Returns:
        {
            "question":     the question asked,
            "answer":       the in-character answer,
            "sources":      the chunks used (with scores),
            "context_used": the full prompt context,
        }
    """
    # 1. Retrieve relevant chunks from this patient's memory
    retrieved = memory.retrieve(question, top_k=top_k)
    if not retrieved:
        # Fallback: use the original_qa if available, or all chunks
        if record.get("original_qa"):
            q_lower = question.lower()
            for qa in record["original_qa"]:
                if any(w in q_lower for w in qa["q"].lower().split() if len(w) > 3):
                    return {
                        "question":  question,
                        "answer":    qa["a"],
                        "sources":   [{"text": qa["q"] + " -> " + qa["a"],
                                       "score": 100, "source": "original_qa"}],
                        "context_used": "(matched from original Q&A)",
                    }
        retrieved = [(c, 50.0) for c in memory.chunks[:top_k]]

    # 2. Build the system prompt
    system = build_patient_system_prompt(record, retrieved)

    # 3. Call the LLM
    user = f"The resident asks me: \"{question}\"\n\nMy answer (in first person, as the patient):"
    raw_answer = llm_call_fn(system, user)

    return {
        "question":     question,
        "answer":       raw_answer,
        "sources":      [{"text": c["text"][:200], "score": s,
                          "source": c.get("source", "")}
                         for c, s in retrieved],
        "context_used": "\n".join(c["text"][:300] for c, _ in retrieved),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation — how faithful is the LLM to the real patient?
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_against_real_qa(
    record: Dict[str, Any],
    memory: PatientMemory,
    llm_call_fn,
    llm_judge_fn = None,
) -> Dict[str, Any]:
    """For each question in record['original_qa'], call the simulator and
    compare the LLM's answer to the real patient's documented answer.
    Uses an LLM-as-judge for semantic similarity (or you can pass a
    different judge function)."""
    results = []
    for qa in record.get("original_qa", []):
        sim = simulate_patient(record, memory, qa["q"], llm_call_fn)
        # Default judge: word overlap ratio (cheap, no extra LLM call)
        if llm_judge_fn is None:
            real_words = set(re.findall(r"\w+", qa["a"].lower()))
            sim_words  = set(re.findall(r"\w+", sim["answer"].lower()))
            overlap = len(real_words & sim_words) / max(len(real_words), 1)
        else:
            overlap = llm_judge_fn(qa["a"], sim["answer"])
        results.append({
            "question":      qa["q"],
            "real_answer":   qa["a"],
            "sim_answer":    sim["answer"],
            "overlap":       overlap,
        })
    avg_overlap = sum(r["overlap"] for r in results) / max(len(results), 1)
    return {
        "n_questions":  len(results),
        "avg_overlap":  avg_overlap,
        "results":      results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Demo — turn the sample patient record into a virtual patient
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("VIRTUAL PATIENT SIMULATOR — no training, RAG-based")
    print("=" * 70)
    print()

    if not HAS_EMBEDDINGS:
        print("ERROR: pip install sentence-transformers numpy")
        exit(1)

    # 1. Build the patient memory from the sample record
    print("Step 1-3: Indexing the patient record...")
    record = PATIENT_RECORD_SCHEMA
    memory = PatientMemory(patient_id=record["patient_id"])
    n = memory.index_record(record)
    print(f"  Indexed {n} chunks from this patient's record")
    print()

    # 2. Use the auto-detected LLM
    import sys
    sys.path.insert(0, "/workspace/latest")
    from elembic_pipeline import auto_pick_llm
    llm = auto_pick_llm()
    print()

    # 3. Demo: ask the simulated patient a few questions
    demo_questions = [
        "When did the chest pain start?",
        "Does the pain radiate anywhere?",
        "How severe is the pain?",
        "What medications are you on?",
        "What did the doctors do for you in the hospital?",
        "When can you go home?",
    ]

    print("=" * 70)
    print("SIMULATED CONVERSATION")
    print("=" * 70)
    for q in demo_questions:
        result = simulate_patient(record, memory, q, llm)
        print(f"\nResident: {q}")
        print(f"Patient:  {result['answer'][:300]}")
        if result["sources"]:
            top_src = result["sources"][0]
            print(f"  (retrieved from {top_src['source']}, score {top_src['score']:.0f}%)")

    # 4. Evaluate against the real patient's documented answers
    print()
    print("=" * 70)
    print("EVALUATION — how well does the LLM match the real patient?")
    print("=" * 70)
    eval_result = evaluate_against_real_qa(record, memory, llm)
    print(f"\nTested {eval_result['n_questions']} questions from the real chart")
    print(f"Average word-overlap with real patient answers: {eval_result['avg_overlap']:.0%}")
    print()
    print("Per-question results:")
    for r in eval_result["results"][:5]:
        print(f"\n  Q: {r['question']}")
        print(f"  Real:  {r['real_answer'][:100]}")
        print(f"  LLM:   {r['sim_answer'][:100]}")
        print(f"  Overlap: {r['overlap']:.0%}")
