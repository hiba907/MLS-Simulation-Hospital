# Integration Guide — Reasoning Canvas + MLS Hospital

This zip contains the **AI MLS Simulation Hospital** with the **Reasoning Canvas**
department integrated. You can drop the contents into your GitHub repo
(`MLS-Simulation-Hospital/`) directly.

## What was changed

| File | Change |
|---|---|
| `hospital_portal.py` | Added "🧠 Reasoning Canvas" to the lobby. Routes to `reasoning_canvas_ui.render(sid)`. |
| `shared_session.py` | Added `"reasoning"` to `DEPT_KEYS`. Added 2 new badges: "🧠 Reasoner" (≥80) and "🎓 Master Reasoner" (≥95). |
| `analytics_dashboard.py` | No change needed — it reads `DEPT_KEYS` dynamically, so "Reasoning" automatically appears in the radar chart and leaderboard. |
| `instructions_module.py` | Added a new "🧠 Reasoning Canvas" tab with the full user guide. |
| `test_smoke.py` | Added 7 new tests for the Reasoning Canvas + elembic_pipeline integration. |
| `requirements.txt` | Added `openai>=1.0`, `sentence-transformers>=2.2`, `anthropic>=0.18`. |
| `.env.example` | Template for OpenRouter / OpenAI / Anthropic API keys. |
| `setup_llm.py` | One-shot script to test your LLM connection. |

## What was added (new files)

| File | Purpose |
|---|---|
| `reasoning_canvas.py` | The Reasoning Canvas engine: tree model, scoring, VitalsSim, case library. |
| `reasoning_canvas_ui.py` | The Streamlit UI for the Reasoning Canvas. |
| `reasoning_followup.py` | Multi-month case evolution engine (updates fire at day 7/30/90). |
| `elembic_pipeline.py` | The v2 NER+L pipeline — extracts clinical events from free-text notes. |
| `journey_timeline.py` | Patient journey timeline renderer (the Substack-style viz). |
| `medication_tracker.py` | Drug timeline + ADR (adverse drug reaction) tracking. |
| `virtual_patient.py` | RAG-based virtual patient simulator (turns real charts into LLM characters). |
| `confidence_scorer.py` | The 5-technique accuracy-maximizing wrapper. |
| `interactive_case_schema.py` | Schema for interactive patient simulation cases. |
| `real_patient_example.py` | Worked example showing expected virtual-patient conversation. |
| `cases/pa_pregnancy_bmj2025.json` | The published BMJ case report. |
| `cases/patient_test_minimal.json` | A minimal test patient for the virtual patient module. |
| `cases/elembic_extracted_case.json` | Sample output of the elembic pipeline. |
| `PATIENT_TEMPLATE.json` | Empty template for authoring new patients. |
| `setup_llm.py` | One-shot LLM connection tester. |
| `INTEGRATION.md` | This file. |

## How to use it

### Step 1: Install dependencies

```bash
cd /path/to/MLS-Simulation-Hospital
pip install -r requirements.txt
```

This installs the existing deps + the new ones for Reasoning Canvas
(openai, sentence-transformers, anthropic).

### Step 2: Set up your LLM (one of these)

**Option A — OpenRouter (free tier, easiest)**
```bash
cp .env.example .env
# Edit .env: paste your OpenRouter key
# Get one free at https://openrouter.ai/keys

# Test the connection
python setup_llm.py
# Should print: ✅ OpenRouter   model=meta-llama/llama-3.3-70b-instruct:free
```

**Option B — Ollama (local, no API key, needs GPU)**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama serve &  # leave running

python setup_llm.py
# Should print: ✅ Ollama (local)   model=llama3.1:8b
```

**Option C — OpenAI direct**
```bash
export OPENAI_API_KEY="sk-..."
python setup_llm.py
```

**Option D — Anthropic direct**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python setup_llm.py
```

### Step 3: Run the hospital

```bash
streamlit run hospital_portal.py
```

The lobby will now show:
- 🔊 Ultrasound Lab
- 🔪 Surgery Suite
- 🧲 FLS Peg Transfer
- 🩻 CT Reading Room
- 🔭 Endoscopy Suite
- **🧠 Reasoning Canvas** ← NEW
- 📊 Analytics Dashboard
- 📖 Instructions

### Step 4: Try the Reasoning Canvas

1. Click **🧠 Reasoning Canvas** in the lobby
2. The case picker appears with the STEMI case pre-loaded
3. Click on the STEMI case
4. Use the palette (❓ history, 🩺 exam, 🧪 lab, 🩻 imaging, 🔀 decision,
   💊 treatment, 📍 diagnosis, 🚪 disposition) to build your workup
5. Connect nodes with arrows
6. Hit **"Finish & See X-Ray"** to see your score vs the expert

### Step 5: Author your own case

1. On the case picker screen, click **"🩺 Leader: Author a case"** (top right)
2. Enter the leader passphrase: **`mls-leader-2026`**
3. Fill in: title, specialty, difficulty, vignette, workup steps,
   critical steps (with time targets), follow-up timeline, what actually happened
4. Hit **"💾 Save case & hide it"**
5. The case is saved to `cases/<your_case_id>.json` and appears in the picker

### Step 6: Try the virtual patient (the "no-training" part)

This is the part that turns your real cases into LLM-simulated patients:

```bash
# 1. Author a patient record (use PATIENT_TEMPLATE.json as a starting point)
cp PATIENT_TEMPLATE.json cases/patient_001.json
# Edit cases/patient_001.json — paste the chart text

# 2. Build the virtual patient (5 min on CPU, no GPU)
python -c "
import json, os
from virtual_patient import PatientMemory
record = json.loads(open('cases/patient_001.json').read())
memory = PatientMemory(patient_id=record['patient_id'])
n = memory.index_record(record)
os.makedirs('patient_memories', exist_ok=True)
memory.save(f'patient_memories/{record[\"patient_id\"]}.pkl')
print(f'Done: {n} chunks indexed')
"

# 3. Talk to the virtual patient
python -c "
import json
from virtual_patient import PatientMemory, simulate_patient
from elembic_pipeline import auto_pick_llm
record = json.loads(open('cases/patient_001.json').read())
memory = PatientMemory(patient_id=record['patient_id'])
memory.load(f'patient_memories/{record[\"patient_id\"]}.pkl')
llm = auto_pick_llm()
for q in ['What brings you to the hospital?', 'How are you feeling?']:
    r = simulate_patient(record, memory, q, llm)
    print(f'\\nResident: {q}')
    print(f'Patient:  {r[\"answer\"]}')
"
```

The LLM will respond AS the patient, using the chart text as context.
No model training. The patient data is the "memory" of the LLM character.

### Step 7: Test the integration

```bash
python test_smoke.py
```

Expected: 21+ tests pass (depending on which optional modules are installed).
Streamlit-related tests skip gracefully if streamlit isn't installed.

## What runs in the app vs as scripts

| Mode | What it does | Command |
|---|---|---|
| **Lobby** (default) | Shows all departments | `streamlit run hospital_portal.py` |
| **Reasoning Canvas** | Student draws workup, gets X-ray | Click 🧠 in lobby |
| **Authoring mode** | Leader adds a new case | Click 🩺 Leader: Author a case |
| **Cohort view** | Aggregate resident attempts | Click 📊 Cohort view |
| **Admin dashboard** | Engagement stats | Click 👥 Admin dashboard |
| **Virtual patient (script)** | Talk to a real patient's LLM twin | `python -c "from virtual_patient import ...; ..."` |
| **Eleembic pipeline (script)** | Extract timeline from free-text notes | `python elembic_pipeline.py` |
| **Medication timeline (script)** | Render drug + ADR viz | `python medication_tracker.py` |

## What's the ROI?

For your **teaching tool**:
- 3 cases today, 10 by next week, 50 by end of month
- 5-10 residents per session, 30+ attempts/day
- Engagement loop proven by 1-week pilot (see DEMO_SCRIPT.md)

For your **research dataset** (the 1000-case corpus):
- Every resident attempt stored as JSON in `hospital_data/`
- Includes: questions asked, orders placed, scoring, drug events, follow-up responses
- 30,000+ attempts after 6 months → ready for fine-tuning your own model

For the **virtual patient** module:
- 5-10 min per real patient to convert to LLM character
- After 50 patients, you have a useful clinical-reasoning training simulator
- The LLM has the patient's actual answers (because you gave it the chart)

## How this connects to your GitHub project

The integration is minimal and surgical:

1. **Add the new files** from this zip to your repo (preserve directory structure)
2. **Merge the 4 modified files** (`hospital_portal.py`, `shared_session.py`,
   `instructions_module.py`, `test_smoke.py`, `requirements.txt`) — the
   diffs are small, see the table at the top of this file
3. **Commit** with a clear message
4. **Deploy** — no other config changes needed

The Reasoning Canvas works WITHOUT an LLM (the v1 system uses deterministic
graph-diff scoring). The LLM is only needed for the v2 modules (elembic
pipeline, virtual patient, accuracy scorer's self-consistency).

## OpenRouter setup (recommended)

The user explicitly requested this. To use OpenRouter as your LLM provider
without calling API functions in PowerShell:

1. Sign up at https://openrouter.ai/keys (free, 1 minute)
2. Copy your key (starts with `sk-or-v1-...`)
3. In your project folder, copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Edit `.env` and paste your key:
   ```
   OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
   OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
   ```
5. Test the connection:
   ```bash
   python setup_llm.py
   ```
   Should print `✅ OpenRouter  model=...`
6. Now run anything — `streamlit run hospital_portal.py`,
   `python virtual_patient.py`, `python elembic_pipeline.py` — the
   pipeline auto-detects your key from the environment.

You never need to set the env var manually in PowerShell again. The
`.env` file is loaded automatically by the pipeline.

## When you want to push to GitHub

```bash
cd /path/to/MLS-Simulation-Hospital
git add hospital_portal.py shared_session.py instructions_module.py test_smoke.py requirements.txt
git add reasoning_canvas.py reasoning_canvas_ui.py reasoning_followup.py
git add elembic_pipeline.py journey_timeline.py medication_tracker.py
git add virtual_patient.py confidence_scorer.py interactive_case_schema.py
git add real_patient_example.py PATIENT_TEMPLATE.json
git add cases/ patient_memories/ .env.example setup_llm.py INTEGRATION.md
git add CHANGELOG.md  # update with integration notes
git commit -m "Add Reasoning Canvas department + LLM-backed virtual patient module

- New department: 🧠 Reasoning Canvas (clinical decision tree training)
- New script modules: elembic_pipeline, journey_timeline, medication_tracker
- New script module: virtual_patient (turns real charts into LLM characters)
- New script module: confidence_scorer (5 accuracy-maximizing techniques)
- New shared_session key: 'reasoning' + 2 new badges (Reasoner, Master Reasoner)
- New instructions tab: 🧠 Reasoning Canvas
- New tests: 7 new tests for the integrated modules
- New .env.example + setup_llm.py for LLM configuration
- requirements.txt: added openai, sentence-transformers, anthropic
"
git push origin main
```
