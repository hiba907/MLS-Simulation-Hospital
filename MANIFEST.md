# Patch Manifest — Reasoning Canvas Integration

This zip contains **only** the files you need to add or change in your existing
MLS-Simulation-Hospital GitHub repo. The 15+ existing files (ultrasound, surgery,
FLS, CT, endoscopy, suture modules, mls_hospital_module, analytics_dashboard,
make_test_dicom, run_hospital.bat/.sh) are NOT included — keep your existing
copies.

## How to use this zip

```bash
# 1. Unzip into your existing repo
cd /path/to/your/MLS-Simulation-Hospital
unzip -o /path/to/reasoning_canvas_patch.zip

# 2. The 16 NEW files go into the project root (or their target subdir)
ls new_files/       # 16 files to add
ls changed_files/   # 5 files to replace
ls MANIFEST.md      # this file

# 3. Copy the new files
cp new_files/*.py .
cp new_files/*.json .
cp new_files/.env.example .
cp new_files/INTEGRATION.md .
cp new_files/MANIFEST.md .
cp -r new_files/cases/ ./cases/   # adds the 3 new case files

# 4. Replace the changed files
cp changed_files/hospital_portal.py .
cp changed_files/shared_session.py .
cp changed_files/instructions_module.py .
cp changed_files/test_smoke.py .
cp changed_files/requirements.txt .

# 5. Add .env to .gitignore (NEVER commit API keys)
grep -q "^\.env$" .gitignore || echo ".env" >> .gitignore

# 6. Install + test
pip install -r requirements.txt
python test_smoke.py
# Expected: 21/21 tests passed

# 7. Set your API key (one-time, local)
cp .env.example .env
# Edit .env: paste your OpenRouter key from https://openrouter.ai/keys
python setup_llm.py
# Expected: ✅ OpenRouter  model=...

# 8. Run the hospital
streamlit run hospital_portal.py
# Lobby now has: 🧠 Reasoning Canvas
```

## Files in new_files/ (16 total — add all)

| File | What it does |
|---|---|
| `reasoning_canvas.py` | The Reasoning Canvas engine: tree model, scoring, VitalsSim, case library |
| `reasoning_canvas_ui.py` | The Streamlit UI for Reasoning Canvas |
| `reasoning_followup.py` | Multi-month case evolution (day 7/30/90 updates) |
| `elembic_pipeline.py` | v2 NER+L pipeline — extracts clinical events from free-text |
| `journey_timeline.py` | Patient journey timeline renderer (Substack-style viz) |
| `medication_tracker.py` | Drug timeline + ADR (adverse drug reaction) tracking |
| `virtual_patient.py` | RAG-based virtual patient (turns real charts into LLM characters) |
| `confidence_scorer.py` | 5-technique accuracy-maximizing wrapper |
| `interactive_case_schema.py` | Schema for interactive patient simulation cases |
| `real_patient_example.py` | Worked example with expected conversation output |
| `setup_llm.py` | One-shot LLM connection tester (reads .env, tests OpenAI/Anthropic/Ollama) |
| `PATIENT_TEMPLATE.json` | Empty template for authoring new virtual patients |
| `.env.example` | Template for OpenRouter/OpenAI/Anthropic keys (COMMIT THIS to git) |
| `INTEGRATION.md` | The full integration guide |
| `cases/pa_pregnancy_bmj2025.json` | Published BMJ case report (may already be in your repo) |
| `cases/elembic_extracted_case.json` | Sample elembic pipeline output (large) |
| `cases/patient_test_minimal.json` | A minimal test patient for the virtual patient module |

## Files in changed_files/ (5 total — replace all in your repo)

| File | What changed | Lines |
|---|---|---|
| `hospital_portal.py` | Added "🧠 Reasoning Canvas" to lobby + route handler | ~5 |
| `shared_session.py` | Added "reasoning" to DEPT_KEYS + 2 new badges (Reasoner, Master Reasoner) | ~3 |
| `instructions_module.py` | Added new "🧠 Reasoning Canvas" tab with user guide | ~30 |
| `test_smoke.py` | Added 7 new tests; wrapped streamlit tests with skip-helper | ~80 |
| `requirements.txt` | Added openai, sentence-transformers, anthropic Python packages | ~3 |

## Files NOT in this zip (you keep your existing copies)

These are the original MLS Hospital files you already have:

- `analytics_dashboard.py` (no change, but listed for reference)
- `ct_hospital_module.py`
- `endoscopy_module.py`
- `make_test_dicom.py`
- `mls_hospital_module.py`
- `run_hospital.bat`, `run_hospital.sh`
- `surgery_module.py`
- `suture_module.py`
- `requirements.txt` (REPLACED with the version in changed_files/)

The `analytics_dashboard.py` doesn't need any changes because it dynamically
reads `DEPT_KEYS` from `shared_session.py` — when you add "reasoning" there,
it automatically appears in the radar chart and leaderboard.

## About the API key (read this carefully)

You do NOT need both OpenRouter AND Anthropic. Pick ONE.

| What you want | Set in `.env` | Python package needed |
|---|---|---|
| Free tier, no card, many models | `OPENROUTER_API_KEY=sk-or-v1-...` | `openai` (already in requirements) |
| Best clinical NER quality (paid) | `ANTHROPIC_API_KEY=sk-ant-...` | `anthropic` (already in requirements) |
| Fallback between two | Set both | Both |

API keys go in `.env` (gitignored), NOT in `requirements.txt`, NOT in code.
The `.env.example` file has placeholders — copy it to `.env` and replace the
placeholders with your real keys.

## Verify the integration

```bash
python test_smoke.py
```

Expected output:
```
Running smoke tests (data dir: /tmp/mls_test_...)...
  ✅  shared_session: register / save / load round-trip
  ✅  shared_session: record_session rejects unknown dept
  ✅  shared_session: record_session accepts valid dept
  ✅  shared_session: Top Surgeon badge requires all depts attempted
  ✅  shared_session: atomic write survives mid-write crash
  ✅  shared_session: reasoning dept in DEPT_KEYS   ← NEW
  ✅  ct_hospital_module: synthetic volumes are correct shape & dtype
  ✅  ct_hospital_module: brain volume has expected HU ranges
  ✅  ct_hospital_module: kidney has all three layers
  ✅  ct_hospital_module: synthetic generation is fast (<2s per case)
  ✅  ct_hospital_module: _extract_spacing handles missing tags
  ✅  suture_module: 8 suture patterns are defined
  ✅  suture_module: score_stitch gives full marks at target
  ✅  suture_module: score_stitch gives 0 beyond max_dist
  ✅  suture_module: render_schematic produces a figure for every type
  ✅  reasoning_canvas: empty tree mutates correctly   ← NEW
  ✅  reasoning_canvas: scoring on perfect tree = ~100   ← NEW
  ✅  reasoning_canvas: scoring on bad tree = low        ← NEW
  ✅  reasoning_canvas: VitalsSim deteriorates on missed critical step   ← NEW
  ✅  elembic_pipeline: section segmentation finds HPI   ← NEW
  ✅  elembic_pipeline: temporal anchor parses 'day N' and 'before admission'   ← NEW

21/21 tests passed
```

If you get less than 21, the most common cause is a missing dependency —
run `pip install -r requirements.txt` and try again.
