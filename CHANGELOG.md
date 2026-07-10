# AI MLS Simulation Hospital — Patch Notes

## What was fixed

### Tier 1 — real bugs (priority fixes)

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `ct_hospital_module.py` | **DICOM slices sorted alphabetically** — broke for `slice1.dcm, slice10.dcm, slice2.dcm` and scrambled anatomy | Sort by `InstanceNumber` then `SliceLocation` (DICOM tags), fall back to filename only as last resort |
| 2 | `ct_hospital_module.py` | **Voxel volume hardcoded to 0.5 mm isotropic** — segmentation `mL` was wrong for real DICOMs with anisotropic spacing | Read `PixelSpacing` + `SliceThickness` from DICOM header, default 0.5 mm only if tags missing. Show spacing in MPR caption. |
| 3 | `shared_session.py` | **"Top Surgeon" badge permanently locked** — required `best >= 80` in all 5 depts, so anyone who never tried Endoscopy could never earn it | Badge now requires mastery only in depts the student has actually attempted (must have ≥1 session in each) |
| 4 | `shared_session.py` | **`save_student()` was non-atomic** — Ctrl+C / OOM mid-write corrupted the student JSON and user got logged in as Guest | Temp-file + `os.replace()` rename. Survives interruption. |

### Tier 2 — UX / scoring

| # | File | Issue | Fix |
|---|------|-------|-----|
| 5 | `mls_hospital_module.py` | FLS handoff threshold hardcoded to **60 px** — never triggered on 4K camera, always triggered on 640 px webcam | Adaptive: `0.05 × frame_width` |
| 6 | `endoscopy_module.py` | Scope navigation scoring was **binary cliff** (90 if in zone, 30 otherwise) — 2 px off the boundary dropped you 60 points | Continuous: `100 − distance_to_zone_center × 200` |
| 7 | `mls_hospital_module.py`, `endoscopy_module.py` | **Silent failure** when `cv2.aruco` not installed — user saw blank screen | `_detect_aruco()` now always returns the raw captured frame with a clear warning overlay; detection runs if available |

### Tier 3 — performance & polish

| # | File | Improvement |
|---|------|-------------|
| 8 | `ct_hospital_module.py` | Vectorized all 6 synthetic CT generators with `np.indices()` + boolean masking — **~100× speedup** (5–10 s → ~50 ms per load) |
| 9 | `run_hospital.bat`, `run_hospital.sh`, `instructions_module.py` | Pinned `opencv-contrib-python>=4.7` (required by `cv2.aruco.ArucoDetector`) and `streamlit>=1.30` |
| 10 | `shared_session.py` | `STUDENTS_DIR` anchored to script location instead of cwd — survives being launched from any folder |

### Tier 4 — testing & docs

| # | File | Added |
|---|------|-------|
| 11 | `test_smoke.py` | New smoke test suite covering: register/save/load, dept whitelist (both accept and reject), Top Surgeon badge logic, atomic-write safety, synthetic volume shapes/dtypes/speed, DICOM spacing fallback |
| 12 | `analytics_dashboard.py` | Renamed misleading "Overall Best" → "Avg Best Score" everywhere (KPI card + leaderboard y-axis) |

---

## Deferred — known limitations (not bugs, would need bigger changes)

### A. Snapshot-only camera (drills feel less "real")
`st.camera_input()` produces single snapshots. Surgery tremor and FLS rate drills rely on the user manually clicking the camera multiple times. Real surgical simulators use continuous video with timestamps.

**To upgrade:** add `streamlit-webrtc` for browser-side webcam streaming, or run a sidecar FastAPI server that ingests frames from OpenCV `VideoCapture` and pushes to Streamlit. ~2 days of work, ~200 lines.

### B. 2D tracking only — no real 3D probe pose
The whole ArUco approach gives you 2D pixel position, not 3D space. Probe could be 30 cm above the phantom and score the same. Inherent ceiling of single-camera setups.

**To upgrade:** stereo cameras (compute depth via triangulation), an RGB-D sensor (Intel RealSense, Apple LiDAR), or an electromagnetic tracker (NDI Aurora — what real IGT systems use). All require hardware investment.

### C. Synthetic volumes are toy shapes, not anatomy
`_make_brain` etc. are stacked ellipsoids. Good enough for teaching window/level and segmentation, not for teaching anatomy.

**To upgrade:** load real segmented atlases (e.g. TotalSegmentator exports, ~1000 structures from any CT), or generate from parametric shape models. Out of scope for v13.

### D. No password / auth on student login
Student ID is the only auth — anyone who knows another student's ID can sign in as them. Fine for a teaching demo, not fine for production.

**To upgrade:** add bcrypt password hashing to `register_student`, store `password_hash` in JSON, validate on `login_student`. ~30 lines.

### E. No persistence beyond local JSON
Student data lives in `hospital_data/*.json` next to the script. Lose the folder = lose all students. No backup, no sync across machines.

**To upgrade:** swap `save_student` to write to SQLite or a small FastAPI + Postgres backend. Existing function signatures stay the same; only the body changes.

---

## How to verify the fixes

```bash
python test_smoke.py
```
Expected: `12/12 tests passed` (or 11/12 if pydicom isn't installed locally — one test is skipped in that case, see test runner output).

```bash
pip install -r requirements.txt
streamlit run hospital_portal.py
```

## Files changed
- `shared_session.py` (atomic write, dept whitelist, badge fix, dir anchor)
- `ct_hospital_module.py` (DICOM ordering, voxel volume, vectorized generators)
- `endoscopy_module.py` (retroflexion logic, continuous scoring, fallback image)
- `analytics_dashboard.py` (label rename)
- `mls_hospital_module.py` (adaptive pixel thresholds, fallback image)
- `run_hospital.bat`, `run_hospital.sh` (OpenCV version pin)
- `instructions_module.py` (updated install commands + drill instructions)

## Files added
- `test_smoke.py` — smoke test suite
- `CHANGELOG.md` — this file
