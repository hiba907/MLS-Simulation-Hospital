"""Interactive case schema for the patient-simulator mode.

Extends the existing reasoning_canvas case JSON with the fields needed for
the resident to actually have a conversation with the patient, instead of
just seeing a vignette and dragging nodes.

This file is a SCHEMA + WORKED EXAMPLE, not a runner. The actual simulator
UI is built on top of this schema using streamlit-chat or similar.

Drop this into your case authoring UI (extend _render_case_author in
reasoning_canvas_ui.py) to add the new fields.
"""
from __future__ import annotations
import json
from typing import List, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# The interactive case schema
# ─────────────────────────────────────────────────────────────────────────────
INTERACTIVE_CASE_SCHEMA = {
    # Existing fields (already in your system)
    "id":                   "stemi_58m",
    "title":                "Mr. Khalil, 58M — Substernal chest pain",
    "specialty":            "Internal Medicine / Cardiology",
    "difficulty":           "Intern",
    "estimated_min":        8,
    "vignette":             "58M with crushing substernal chest pain × 90 min.",
    "questions_for_student": [...],
    "critical_steps":       [...],
    "critical_node_labels": [...],
    "vitals_initial":       {"hr": 108, "sbp": 92, "dbp": 60, "spo2": 94, "rr": 22},
    "expert_nodes":         [...],
    "expert_edges":         [...],
    "follow_up":            {...},
    "case_source":          {...},

    # NEW: interactive simulation fields
    "interactive": {
        # The LLM uses this as the system prompt — defines the patient persona
        "persona": {
            "name":             "Mr. Youssef Khalil",
            "age":              58,
            "sex":              "M",
            "occupation":       "retired teacher",
            "demeanor":         "anxious, diaphoretic, short answers due to pain",
            "language":         "English (also speaks Arabic)",
            "education_level":  "high — gives detailed history when asked",
            "primary_language": "en",
        },

        # The LLM uses these as the canonical "answers" to common questions
        # (Fallback: if the LLM doesn't have a scripted answer, it improvises
        # based on the persona + vignette.)
        "scripted_responses": {
            "onset":             "About an hour and a half ago. I was watching TV.",
            "character":         "Crushing. Like someone's sitting on my chest.",
            "radiation":         "Yes, to my left arm and my jaw.",
            "severity":          "8 out of 10. The worst pain I've ever had.",
            "associated":        "I'm sweating a lot and I feel nauseous. I almost vomited.",
            "aggravating":       "Nothing makes it better. Breathing doesn't change it.",
            "alleviating":       "Nothing helps. I tried resting but it didn't go away.",
            "prior_episodes":    "No, never had anything like this before.",
            "pmh":               "I have high blood pressure and diabetes. Take medications for both.",
            "medications":       "Amlodipine 5 mg once a day, metformin 1000 mg twice a day.",
            "allergies":         "No known drug allergies.",
            "family_history":    "My father died of a heart attack at 62.",
            "smoking":           "I smoke about a pack a day. Have for 30 years.",
            "alcohol":           "Occasional, maybe a glass of wine on weekends.",
            "last_meal":         "Lunch about 3 hours ago. Rice and chicken.",
        },

        # What the resident finds if they do a physical exam
        # (LLM reveals these when the resident orders "cardiovascular exam",
        # "lung exam", etc.)
        "exam_findings": {
            "general":           "Anxious, diaphoretic, in moderate distress",
            "cardiovascular": {
                "heart_sounds":   "Regular rate and rhythm, no murmurs, no rubs, no gallops",
                "jvp":            "Not elevated",
                "peripheral_pulses": "2+ dorsalis pedis bilaterally, no edema",
                "capillary_refill": "<2 seconds",
            },
            "pulmonary": {
                "inspection":     "No chest wall deformity",
                "auscultation":   "Clear to auscultation bilaterally, no wheezes/rales/rhonchi",
            },
            "abdominal": {
                "inspection":     "Soft, non-distended",
                "auscultation":   "Normoactive bowel sounds",
                "palpation":      "Soft, non-tender, no organomegaly",
            },
            "neurological": {
                "mental_status":  "Alert and oriented x3, anxious",
                "cranial_nerves": "II-XII grossly intact",
            },
        },

        # What the resident gets when they order a test
        "lab_results": {
            "12-lead ECG": {
                "result_text":   "ST elevation in leads II, III, aVF. Reciprocal ST depression in I, aVL. Normal sinus rhythm at 98 bpm.",
                "interpretation": "Acute inferior STEMI",
                "critical":      True,
            },
            "troponin I": {
                "result_text":   "4.8 ng/mL (normal <0.04)",
                "interpretation": "Markedly elevated — consistent with myocardial injury",
                "critical":      True,
            },
            "CBC": {
                "result_text":   "WBC 9.2, Hgb 14.1, Plt 245",
                "interpretation": "Normal",
            },
            "BMP": {
                "result_text":   "Na 138, K 4.2, Cl 101, CO2 24, BUN 18, Cr 1.4, Glc 218",
                "interpretation": "Mildly elevated creatinine and hyperglycemia (poorly controlled DM)",
            },
            "coags": {
                "result_text":   "PT 11.5, INR 1.0, aPTT 28",
                "interpretation": "Normal",
            },
            "type_and_screen": {
                "result_text":   "Type A+, antibody screen negative",
                "interpretation": "Ready for transfusion if needed",
            },
            "magnesium": {
                "result_text":   "2.0 mg/dL",
                "interpretation": "Normal",
            },
            "bedside echo": {
                "result_text":   "EF 30%. Inferior wall hypokinesis. No pericardial effusion. No mechanical complications.",
                "interpretation": "Severely reduced EF with inferior wall motion abnormality",
                "critical":      True,
            },
            "right-sided ECG": {
                "result_text":   "No ST elevation in V4R",
                "interpretation": "No RV involvement — nitrates are safe to consider (though BP may limit)",
            },
        },

        # What happens when the resident prescribes a drug
        "drug_responses": {
            "aspirin 325 mg": {
                "response":     "Given at 04:35. No immediate reaction.",
                "outcome":      "standard care for ACS",
            },
            "ticagrelor 180 mg": {
                "response":     "Loaded at 04:38. No bleeding.",
                "outcome":      "standard DAPT loading",
            },
            "morphine 4 mg IV": {
                "response":     "Pain improved from 8/10 to 4/10 after one dose. No respiratory depression.",
                "outcome":      "appropriate for ongoing ischemic pain",
            },
            "nitroglycerin 0.4 mg SL": {
                "response":     "BP dropped from 92/60 to 78/52 after one dose. Held further nitroglycerin. Pain unchanged.",
                "outcome":      "contraindicated in inferior STEMI with RV involvement OR hypotension",
                "adverse":      True,
                "causality":    "certain (temporal + hemodynamic response)",
            },
            "metoprolol 5 mg IV": {
                "response":     "BP dropped from 88/54 to 72/48. Heart rate 62. Held further beta-blocker. Patient more fatigued.",
                "outcome":      "early IV beta-blocker contraindicated in acute STEMI with HF or hypotension",
                "adverse":      True,
            },
            "normal saline 500 mL": {
                "response":     "BP improved from 88/54 to 96/62. HR 102. Patient less diaphoretic.",
                "outcome":      "appropriate for pre-load in inferior STEMI",
            },
        },

        # What happens at each time point (for the follow-up timeline)
        # This is mostly already in follow_up_timeline, but you can add
        # time-anchored events here
        "scenario_branches": {
            "if_no_ECG_within_30_min": {
                "outcome":     "Patient deteriorates. Becomes hypotensive (BP 70/40), bradycardic (HR 50). VT arrest. Requires CPR + intubation.",
                "teaching":    "Door-to-ECG <10 min is the standard. Every minute of delay = more myocardium.",
            },
            "if_aspirin_not_given": {
                "outcome":     "Patient has VF arrest 8 minutes into ED course. Defibrillated, ROSC. Undergoes PCI with larger infarct, EF 20% post-PCI.",
                "teaching":    "Aspirin 325 mg chewed = mortality benefit, immediate. Never delay.",
            },
            "if_cath_not_activated": {
                "outcome":     "Patient transferred to floor after 'stable' presentation. Found unresponsive 4 hours later. VF arrest, unable to resuscitate.",
                "teaching":    "Door-to-balloon <90 min is the guideline. Activate cath lab based on ECG, not on troponin or stability.",
            },
        },
    },

    # NEW: medication tracking (from medication_tracker.py)
    "medication_timeline": [...],
    "drug_reactions":      [...],
    "lab_monitoring":      [...],
    "adherence_notes":     [...],

    # NEW: follow-up timeline (already in your system, extended with drug events)
    "follow_up_timeline": [...],
}


# ─────────────────────────────────────────────────────────────────────────────
# How to author a case with the interactive schema
# ─────────────────────────────────────────────────────────────────────────────
def author_interactive_case(checklist: bool = True) -> Dict[str, Any]:
    """Return a blank template that an attending fills in to author a new
    interactive case. Each field has a clear comment about what to write."""
    template = {
        "id":                  "your_case_id_here",
        "title":               "age/sex, chief complaint",
        "specialty":           "e.g. Internal Medicine / Cardiology",
        "difficulty":          "Intern | Resident | Senior",
        "estimated_min":       8,
        "vignette":            "1-2 sentence opening (what the resident sees first)",

        "questions_for_student": [
            "3-5 guiding questions, not graded",
        ],

        "critical_steps": [
            {"label": "12-lead ECG within 10 min", "target_by": 120, "done_at": None},
        ],
        "critical_node_labels": [
            "12-lead ECG within 10 min",
        ],
        "vitals_initial": {"hr": 90, "sbp": 120, "dbp": 80, "spo2": 98, "rr": 16},

        # Expert tree (the ideal workup)
        "expert_nodes": [
            {"id": "n_001", "type": "history",   "label": "...",
             "rationale": "..."},
        ],
        "expert_edges": [
            {"from": "n_001", "to": "n_002"},
        ],

        # Interactive simulation fields (NEW)
        "interactive": {
            "persona": {
                "name":             "Patient name (de-identified)",
                "age":              0,
                "sex":              "M | F",
                "occupation":       "...",
                "demeanor":         "anxious / calm / confused / etc.",
                "language":         "English / Arabic / French / mixed",
                "education_level":  "low / medium / high",
                "primary_language": "en | ar | fr",
            },
            "scripted_responses": {
                # The LLM will use these as canonical answers. Fill in
                # the ones that matter for this case. Add more keys as needed.
                "onset":          "When did it start?",
                "character":      "What does it feel like?",
                "radiation":      "Does it go anywhere?",
                "severity":       "How bad is it? (0-10)",
                "associated":     "Any other symptoms?",
                "aggravating":    "What makes it worse?",
                "alleviating":    "What makes it better?",
                "prior_episodes": "Have you had this before?",
                "pmh":            "Any medical conditions?",
                "medications":    "What medications do you take?",
                "allergies":      "Any drug allergies?",
                "family_history": "Family history?",
                "smoking":        "Do you smoke?",
                "alcohol":        "Do you drink?",
            },
            "exam_findings": {
                # What the LLM reveals when the resident orders a system exam
                "general":         "...",
                "cardiovascular":  {"heart_sounds": "...", "jvp": "..."},
                "pulmonary":       {"auscultation": "..."},
                "abdominal":       {"palpation": "..."},
                "neurological":    {"mental_status": "..."},
            },
            "lab_results": {
                # The LLM returns the matching entry when the resident orders it
                "12-lead ECG": {"result_text": "...", "interpretation": "...",
                                "critical": True},
                "troponin I":  {"result_text": "...", "interpretation": "..."},
                # Add more labs as needed
            },
            "drug_responses": {
                # The LLM returns the matching entry when the resident prescribes
                "aspirin 325 mg": {"response": "...", "outcome": "..."},
            },
            "scenario_branches": {
                # Optional: what happens if the resident makes key errors
                "if_X": {"outcome": "...", "teaching": "..."},
            },
        },

        # Drug tracking (from medication_tracker.py)
        "medication_timeline": [],
        "drug_reactions":      [],
        "lab_monitoring":      [],
        "adherence_notes":     [],

        # Follow-up timeline
        "follow_up_timeline": [
            {"at_days": 7,  "type": "follow_up_visit",
             "headline": "...", "data": {}, "prompt": "...",
             "teaching_note": "..."},
        ],

        # Provenance (de-identification + ethics)
        "case_source": {
            "type": "real | published | composite",
            "origin": "...",
            "deidentified": True,
            "consent_note": "...",
        },
        "follow_up": {
            "actual_path": "What actually happened in sequence",
            "complications": "Any, or 'None'",
            "discharge_outcome": "Status at discharge",
            "long_term": "Follow-up status if available",
            "teaching_pearl": "The attending's real-world takeaway",
        },
    }
    if checklist:
        print("""
INTERACTIVE CASE AUTHORING CHECKLIST
=====================================
For the LLM-as-patient simulator to work well, your case MUST have:

[ ] persona            — name, age, sex, demeanor, language
[ ] scripted_responses — at least 8-10 common HPI questions answered
[ ] exam_findings      — at least 4 systems (CV, pulm, abdo, neuro)
[ ] lab_results        — at least 5-6 tests the resident might order
[ ] drug_responses     — at least 3 drugs the resident might prescribe
[ ] scenario_branches  — at least 2-3 "what if" outcomes for common errors
[ ] expert_nodes/edges — the ideal workup tree (REQUIRED — drives scoring)
[ ] critical_steps     — the time-critical steps that drive the vitals sim
[ ] follow_up_timeline — at least 2 updates (e.g. day 7, day 30)
[ ] medication_timeline — all drugs the patient is on
[ ] case_source        — provenance + de-id + consent
[ ] follow_up          — what actually happened

If you skip any of these, the simulator will fall back to improvising
with the LLM, which is fine for some fields (e.g. drug_responses) but
risky for others (e.g. scripted_responses — if you don't write the
patient's story, the LLM will make it up, which is bad for teaching).

Estimated time to author: 45-90 minutes per case for a full interactive case.
                          20-30 minutes if you only fill in the essentials.
""")
    return template


# ─────────────────────────────────────────────────────────────────────────────
# The patient-simulator prompt template (what the LLM receives)
# ─────────────────────────────────────────────────────────────────────────────
PATIENT_SIMULATOR_SYSTEM_PROMPT = """You are a simulated patient in a clinical training scenario.

PATIENT PERSONA:
{persona_json}

SCRIPTED RESPONSES (use these as the canonical answers when the resident
asks questions; improvise consistent answers for other reasonable questions):
{scripted_responses_json}

EXAM FINDINGS (reveal these when the resident performs a physical exam):
{exam_findings_json}

VITALS: {vitals_initial}

INSTRUCTIONS:
- Stay in character. You are this patient. Use first person.
- Match the patient's demeanor (anxious, calm, etc.).
- Be realistic. Don't volunteer information the patient wouldn't naturally offer.
  If the resident doesn't ask the right question, they don't get the answer.
- For scripted questions, give the scripted response (paraphrase naturally).
- For unscripted reasonable questions, improvise a consistent answer.
- For irrelevant or out-of-scope questions, politely decline ("I don't know" / "I don't understand the question").
- Never break character. Never reveal you're an AI. Never give medical advice.
- Use {primary_language} as your primary language, but switch to {secondary_language} if the resident does.

EXAMPLES:
Resident: "When did the pain start?"
You: "{onset_response}"

Resident: "Does the pain radiate anywhere?"
You: "{radiation_response}"

Resident: "Do you have any medical conditions?"
You: "{pmh_response}"
"""


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    case = author_interactive_case(checklist=True)
    # Don't print the whole template, just the structure
    print(f"Template has {len(case)} top-level fields")
    print(f"Interactive schema has {len(case['interactive'])} sub-fields")
    print(f"  persona:           {len(case['interactive']['persona'])} fields")
    print(f"  scripted_responses: {len(case['interactive']['scripted_responses'])} questions")
    print(f"  exam_findings:     {len(case['interactive']['exam_findings'])} systems")
