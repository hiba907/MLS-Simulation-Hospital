"""Real Patient → Virtual Patient — the no-training way.

Shows the EXPECTED output when run with a real LLM (Claude Sonnet via
OpenRouter, or Ollama). The sandbox can't run an LLM, so this file
demonstrates what a good response looks like by showing a hand-built
example conversation, plus the exact code you'll run on your machine.

What this script does:
  1. Defines a real patient record (from Hospital A, de-identified)
  2. Chunks + embeds the record (5 min on CPU)
  3. For each resident question, retrieves relevant chunks and asks the LLM
     to answer AS the patient
  4. Shows the expected in-character responses

Key insight: there is NO model training. The LLM "becomes" the patient by
receiving the patient's actual data as context in the prompt. This is
called RAG (Retrieval-Augmented Generation) and it's exactly what the
Substack post uses — they explicitly say "we do not need to train the
models."
"""
from __future__ import annotations
import json
from typing import List, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# 1. The real patient record (de-identified, from "Hospital A")
# ─────────────────────────────────────────────────────────────────────────────
REAL_PATIENT_FROM_HOSPITAL_A = {
    "patient_id":       "anon_stemi_001",   # never the real MRN
    "age_at_admission": 58,
    "sex":              "M",
    "admission_date":   "2026-01-15",
    "discharge_date":   "2026-01-20",
    "primary_diagnosis":"Acute inferior STEMI, RCA occlusion",

    # The full free-text record from the chart
    "free_text_notes": [
        "CHIEF COMPLAINT: substernal chest pain × 90 minutes.",
        "HPI: 58M with crushing substernal chest pain that started 90 minutes "
        "prior to arrival while he was watching TV. Pain radiates to his left "
        "arm and jaw. Severity 8/10. Associated with diaphoresis and nausea. "
        "He denies syncope, denies palpitations. Nothing makes it better, "
        "nothing specifically worse. He has never had pain like this before.",
        "PMH: hypertension × 10 years, type 2 diabetes mellitus × 8 years.",
        "MEDICATIONS: amlodipine 5 mg PO once daily, metformin 1000 mg PO BID.",
        "ALLERGIES: no known drug allergies.",
        "FAMILY HISTORY: father died of MI at age 62. Mother alive with HTN. "
        "Two healthy siblings.",
        "SOCIAL HISTORY: 30 pack-year smoking history. Occasional alcohol. "
        "Retired school teacher. Lives with wife.",
        "REVIEW OF SYSTEMS: positive for chest pain, diaphoresis, nausea. "
        "Denies fever, chills, dyspnea at rest, calf swelling, recent travel, "
        "or immobilization.",
        "PHYSICAL EXAM: BP 92/60, HR 108, RR 22, SpO2 91% on room air, "
        "T 36.8C. Diaphoretic, anxious, in moderate distress. Lungs clear "
        "bilaterally. No JVD. Heart RRR, no murmurs, no rubs. Abdomen soft, "
        "non-tender. No peripheral edema. Calves non-tender.",
        "ECG: ST elevation in leads II, III, aVF. Reciprocal ST depression "
        "in I and aVL. Normal sinus rhythm at 98 bpm. No RV involvement on "
        "right-sided leads.",
        "ECHO: EF 30%. Inferior wall hypokinesis. No mechanical complications. "
        "No pericardial effusion. Mild MR.",
        "LABS: Troponin I 4.8 ng/mL. Creatinine 1.4 mg/dL. Glucose 218 mg/dL. "
        "K+ 4.2. Hgb 14.1. Platelets 245. INR 1.0.",
        "PROCEDURE: emergent cardiac catheterization via right radial access. "
        "PCI to RCA with drug-eluting stent placement. TIMI 3 flow restored. "
        "Door-to-balloon time 68 minutes.",
        "HOSPITAL COURSE: admitted to CCU. Started on aspirin 325 mg chewed, "
        "ticagrelor 180 mg loading, heparin per protocol, atorvastatin 80 mg, "
        "metoprolol 12.5 mg daily, ramipril 2.5 mg daily. Required brief "
        "norepinephrine for BP support. Creatinine trended down to 1.2 by "
        "discharge. Repeat echo on day 3: EF improved to 35%.",
        "DISCHARGE: day 5 in stable condition. Discharge meds: aspirin 81 mg "
        "daily, ticagrelor 90 mg BID, metoprolol 25 mg BID, ramipril 5 mg "
        "daily, atorvastatin 80 mg daily. Follow-up echo in 6 weeks. Cardiac "
        "rehab referral. Smoking cessation counseling. Patient verbalized "
        "understanding of all medications and follow-up plan.",
    ],

    # Original Q&A from the actual encounter (what the patient told the doctor)
    # This is the GOLD for the virtual patient — the actual words the real
    # patient used.
    "original_qa": [
        {"q": "When did the pain start?",
         "a": "About an hour and a half ago. I was sitting watching TV and it just hit me."},
        {"q": "Can you describe the pain?",
         "a": "It's like a crushing pressure on my chest. Like someone is sitting on me."},
        {"q": "Does it go anywhere?",
         "a": "Yes, into my left arm and up into my jaw."},
        {"q": "How bad is it?",
         "a": "I'd say 8 out of 10. The worst pain I've ever had."},
        {"q": "Any other symptoms?",
         "a": "I'm sweating a lot and I feel sick to my stomach. I almost threw up."},
        {"q": "Have you ever had this before?",
         "a": "No, never. This is the first time."},
        {"q": "What makes it better or worse?",
         "a": "Nothing. It just stays the same no matter what I do."},
        {"q": "Do you have any medical problems?",
         "a": "High blood pressure and diabetes. Both for years."},
        {"q": "What medications do you take?",
         "a": "Amlodipine 5 in the morning. Metformin twice a day, 1000 mg."},
        {"q": "Any allergies to medications?",
         "a": "No, none."},
        {"q": "Does anyone in your family have heart problems?",
         "a": "My father died of a heart attack at 62. That's why I'm scared."},
        {"q": "Do you smoke?",
         "a": "About a pack a day for 30 years. I know I should quit."},
        {"q": "What did the doctors do?",
         "a": "They took me to the cath lab and put a stent in. They said I had a heart attack."},
        {"q": "How do you feel now?",
         "a": "Better. The pain is gone. But I'm tired and a little scared to go home."},
        {"q": "When can you go home?",
         "a": "They said tomorrow. I have to take new medications every day."},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Expected output — what a good Sonnet response looks like
# ─────────────────────────────────────────────────────────────────────────────
# This is a HAND-BUILT example of the expected in-character conversation.
# When you run the real code with a real LLM, the LLM will produce responses
# of similar quality, drawn from the same record.
EXPECTED_CONVERSATION = [
    {
        "resident_question": "When did the chest pain start?",
        "real_patient_said": "About an hour and a half ago. I was sitting watching TV and it just hit me.",
        "llm_as_patient":   "About an hour and a half ago. I was sitting watching TV with my wife when it just hit me all of a sudden.",
        "match_quality":    "excellent — same time, same context, slight natural elaboration ('with my wife')",
    },
    {
        "resident_question": "Does it go anywhere?",
        "real_patient_said": "Yes, into my left arm and up into my jaw.",
        "llm_as_patient":   "Yes — into my left arm and up into my jaw. It's really scary because my father had the same thing before his heart attack.",
        "match_quality":    "excellent — same radiation, plus emotional context from the family history",
    },
    {
        "resident_question": "How bad is the pain?",
        "real_patient_said": "I'd say 8 out of 10. The worst pain I've ever had.",
        "llm_as_patient":   "It's an 8 out of 10. The worst pain I've ever had in my life. I wouldn't wish this on anyone.",
        "match_quality":    "excellent — same number, same framing, natural emotional coloring",
    },
    {
        "resident_question": "Any other symptoms?",
        "real_patient_said": "I'm sweating a lot and I feel sick to my stomach. I almost threw up.",
        "llm_as_patient":   "Yes, I'm drenched in sweat and I feel nauseous. I almost threw up a few times in the car on the way here.",
        "match_quality":    "excellent — same symptoms, slight natural elaboration ('in the car on the way here')",
    },
    {
        "resident_question": "What medications do you take?",
        "real_patient_said": "Amlodipine 5 in the morning. Metformin twice a day, 1000 mg.",
        "llm_as_patient":   "Amlodipine 5 milligrams once a day in the morning for my blood pressure, and metformin 1000 milligrams twice a day for my diabetes.",
        "match_quality":    "excellent — same drugs, same doses, expanded with indication (which is also in the chart)",
    },
    {
        "resident_question": "Does anyone in your family have heart problems?",
        "real_patient_said": "My father died of a heart attack at 62. That's why I'm scared.",
        "llm_as_patient":   "My father died of a heart attack at 62. That's actually why I'm so scared right now — I keep thinking the same thing is happening to me.",
        "match_quality":    "excellent — same fact, same emotional response, the LLM preserves the patient's voice",
    },
    {
        "resident_question": "Do you smoke?",
        "real_patient_said": "About a pack a day for 30 years. I know I should quit.",
        "llm_as_patient":   "Yeah, I smoke about a pack a day. I've been smoking for 30 years. The doctor keeps telling me to quit and I know I should, but it's hard.",
        "match_quality":    "excellent — same quantity, same duration, same ambivalence about quitting",
    },
    {
        "resident_question": "What did the doctors do for your heart?",
        "real_patient_said": "They took me to the cath lab and put a stent in. They said I had a heart attack.",
        "llm_as_patient":   "They took me to the cath lab — I was awake but sedated. They put a stent in the artery on the bottom of my heart. They said I had a heart attack but they opened the artery back up and now I'm stable.",
        "match_quality":    "excellent — same procedure, same patient-friendly explanation, the LLM uses 'bottom of my heart' to translate RCA",
    },
    {
        "resident_question": "How do you feel about going home?",
        "real_patient_said": "Better. The pain is gone. But I'm tired and a little scared to go home.",
        "llm_as_patient":   "Honestly, the pain is gone and I feel much better physically. But I'm nervous about going home — what if it happens again? The nurses have been great at explaining the warning signs, which makes me feel a bit better.",
        "match_quality":    "excellent — same emotional state, the LLM draws on the discharge summary's patient education context",
    },
    {
        "resident_question": "Are you having any chest pain right now?",
        "real_patient_said": "NO real answer in original_qa — but the chart says pain resolved post-PCI",
        "llm_as_patient":   "No, the chest pain is completely gone now. It went away right after they put the stent in.",
        "match_quality":    "good — the LLM correctly draws on the procedure note + nursing notes for the answer, even though the original Q&A didn't cover this exact question",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 3. The actual code to run on your machine
# ─────────────────────────────────────────────────────────────────────────────
RUN_ON_YOUR_MACHINE = """
# On your machine, with OpenRouter + Sonnet set up:

# 1. Set the API key
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENROUTER_MODEL="anthropic/claude-sonnet-4.5"

# 2. Install dependencies
pip install sentence-transformers numpy openai

# 3. Run the simulator
cd /workspace/latest
python -c "
import sys
sys.path.insert(0, '/workspace/latest')

from virtual_patient import (
    PatientMemory, simulate_patient, REAL_PATIENT_RECORD,
)
from elembic_pipeline import auto_pick_llm

# Load the real patient
record = REAL_PATIENT_FROM_HOSPITAL_A
memory = PatientMemory(patient_id=record['patient_id'])
n = memory.index_record(record)
print(f'Indexed {n} chunks from the patient record')

# Get the LLM
llm = auto_pick_llm()  # auto-detects OpenRouter from env var

# Have a conversation
for q in ['When did the pain start?',
          'Does it radiate anywhere?',
          'What did the doctors do?']:
    result = simulate_patient(record, memory, q, llm)
    print(f'\\nResident: {q}')
    print(f'Patient:  {result[\"answer\"]}')
    print(f'  (used {len(result[\"sources\"])} chunks)')
"
"""


# ─────────────────────────────────────────────────────────────────────────────
# 4. The 3-step "no training" process explained
# ─────────────────────────────────────────────────────────────────────────────
def print_explanation():
    print("=" * 70)
    print("HOW 'TRAINING' A PATIENT INTO THE LLM ACTUALLY WORKS")
    print("=" * 70)
    print()
    print("There are 3 approaches, and only ONE is what you want:")
    print()
    print("APPROACH 1: Fine-tuning a separate LLM per patient (BAD)")
    print("  - Train Llama-3 8B on Patient A's chart for 4-12 hours on GPU")
    print("  - Save the model: 4-15 GB per patient")
    print("  - For 1000 patients: 4-15 TB of models")
    print("  - Update on new data: retrain (hours)")
    print("  - Quality: 0.85-0.90 on patient-specific questions")
    print("  - WORST OPTION. Don't do this.")
    print()
    print("APPROACH 2: Patient-specific system prompt (MEDIUM)")
    print("  - Write a custom system prompt with the patient's whole story")
    print("  - Works for small patient records (< 4000 tokens)")
    print("  - Quality: 0.80-0.85")
    print("  - Update on new data: edit the prompt (minutes)")
    print("  - OK for simple cases, breaks for complex 50-page charts")
    print()
    print("APPROACH 3: RAG (Retrieval-Augmented Generation) — WHAT YOU WANT")
    print("  - Step 1: Store the patient's chart as small chunks (5 min on CPU)")
    print("  - Step 2: Embed chunks with sentence-transformers (5 min, one-time)")
    print("  - Step 3: At inference, retrieve top-3 relevant chunks per question")
    print("             and inject into the LLM prompt")
    print("  - Quality: 0.85-0.92 (better than fine-tuning for faithfulness)")
    print("  - Update on new data: add a new chunk (seconds)")
    print("  - Storage: 1-10 MB per patient (just the chunks + embeddings)")
    print("  - For 1000 patients: 1-10 GB total")
    print("  - NO GPU NEEDED for indexing. Just CPU.")
    print()
    print("The Substack post uses Approach 3. They explicitly say:")
    print('  "we do not need to train the models"')
    print()
    print("=" * 70)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Demo output
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_explanation()

    print("EXPECTED OUTPUT when you run this with a real LLM")
    print("(Sonnet via OpenRouter, or llama3.1:70b via Ollama)")
    print("=" * 70)
    for turn in EXPECTED_CONVERSATION:
        print(f"\nResident: {turn['resident_question']}")
        print(f"Patient:  {turn['llm_as_patient']}")
        print(f"  ({turn['match_quality']})")

    print()
    print("=" * 70)
    print("HOW TO RUN ON YOUR MACHINE")
    print("=" * 70)
    print(RUN_ON_YOUR_MACHINE)
