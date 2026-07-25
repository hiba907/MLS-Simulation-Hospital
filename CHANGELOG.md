## v14-Final — MLS Merge (Perlin CT + surgery_module)

### Changes from v14 base

| # | File | Change |
|---|------|--------|
| 1 | `ct_hospital_module.py` | Added `_perlin_like_noise()` helper using `scipy.ndimage.gaussian_filter`. All 6 synthetic CT generators now use multi-scale Perlin-like noise to perturb organ boundaries — no more perfect spheres/ellipsoids. HU tissue values and anatomy (sulci, ventricles, ribs, portal vein, etc.) preserved from v14. |
| 2 | `surgery_module.py` | Added full surgery module from MLS-Hospital-Updated: ArUco wound-edge markers (IDs 20 & 21), incision scoring (tremor 40% + path 40% + endpoint 20%), Suture Lab with 6 patterns (Simple Interrupted, Running, Locking, Figure-8, Vertical Mattress, Horizontal Mattress), reference schematics, and clinical notes. |
| 3 | `hospital_portal.py` | Surgery department now routes to `surgery_module.run()` instead of `mls_hospital_module.render("surgery", ...)`. |
| 4 | `requirements.txt` | Added `pandas`, `matplotlib` to dependency list. |

# AI MLS Simulation Hospital — Patch Notes

## v14 — Anatomical CT phantoms + Wound-marker surgery + Suture Lab

This version tackles the three user-reported issues: the brain (and other CT
cases) looking like nested circles, the missing wound-marker concept in the
surgery drill, and the lack of a suture practice module.

### Major changes

| # | File | Change |
|---|------|--------|
| 1 | `ct_hospital_module.py` | **All 6 synthetic CT generators rewritten as anatomical phantoms.** Brain now has skull, CSF, gray/white matter, lateral & third ventricles, falx cerebri, sulci/gyri surface texture, cerebellum, brainstem. Chest has body wall, 2 lungs, heart, aorta, trachea, spine, sternum, 10 ribs. Abdomen has body wall, liver with portal vein, spleen, 2 kidneys, aorta, IVC, spine. Bone has cortical ring + trabecular cancellous + medullary canal. Kidney has cortex/medulla/pelvis. Liver has parenchyma + portal vein + hepatic veins + gallbladder + IVC. |
| 2 | `mls_hospital_module.py` | **Surgery tab is now 3 sub-tabs:** (1) **Incision drill with wound markers** — 2 new marker IDs (20 & 21) taped at the 2 ends of the cut; simulator draws the wound line and scores tremor (40 %) + path-following (40 %) + endpoint coverage (20 %); works for any cut size 1 cm to 20+ cm. (2) **🪡 Suture Lab** — see #3. (3) **Tremor analysis** (the old drill, kept as-is). |
| 3 | `suture_module.py` (new) | **Suture Lab with 8 patterns**: Simple Interrupted, Simple Continuous (Running), Continuous Locking (Ford), Figure-of-8, Vertical Mattress (Donati), Horizontal Mattress, Subcuticular (Running Subcutaneous), Purse-string. Each has a reference schematic (matplotlib drawing showing the wound, stitches, knots, and numbered bite order), clinical description, indications, and a target list. The simulator lets the student practise the bites with the camera — each capture is scored by distance to the target (0-100, normalized to frame size). |
| 4 | `instructions_module.py` | Added the wound-marker explanation, the new ID 20/21 marker table, and a dedicated **🪡 Suture Lab** tab with the full pattern table, scoring system, and clinical context. |
| 5 | `test_smoke.py` | Updated HU-range assertions to match the new anatomical phantoms. Added 4 new tests covering suture patterns, scoring, and schematic rendering. Now **14/14 tests pass.** |

### Why 2 markers for the wound?

A *cut* is a break in the skin — it's a line, and a line needs 2 points to
define it. With markers at the 2 ends (ID 20 + ID 21), the simulator knows the
wound's position, length, and orientation — regardless of whether the cut is
1 cm or 20 cm. For very small cuts where 2 printed markers would crowd, place
them slightly off the wound on either side, along the wound's long axis. The
math still works.

For a curved cut you'd need 3+ markers; that's a future extension.

### Performance
- New CT generators run in **130-200 ms** each (was ~50 ms for the toy
  ellipsoids). Still well under the 2 s smoke-test budget.
- Suture gallery renders in <100 ms; schematics are pure NumPy/matplotlib.
- Total test suite: 14/14 pass in <2 s.

### Known limitations (unchanged from v13)
- A. Snapshot-only camera (`st.camera_input`)
- B. 2D tracking only — no real 3D probe pose
- C. ~~Synthetic volumes are toy shapes~~ ✅ **Fixed in v14 — now anatomical**
- D. No password / auth on student login
- E. No persistence beyond local JSON

---

## What was fixed (v13 — prior version)

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
| 11 | `test_smoke.py` | New smoke test suite covering: register/save/load, dept whitelist (both accept and reject), Top Surgeon badge logic, atomic-write safety, synthetic volume shapes/dtypes/speed, DICOM spacing fallback, suture patterns & scoring |
| 12 | `analytics_dashboard.py` | Renamed misleading "Overall Best" → "Avg Best Score" everywhere (KPI card + leaderboard y-axis) |

---

## How to verify the fixes

```bash
python test_smoke.py
```
Expected: `14/14 tests passed`

```bash
pip install -r requirements.txt
streamlit run hospital_portal.py
```

## Files changed
- `shared_session.py` (atomic write, dept whitelist, badge fix, dir anchor)
- `ct_hospital_module.py` (DICOM ordering, voxel volume, **anatomical phantoms**)
- `endoscopy_module.py` (retroflexion logic, continuous scoring, fallback image)
- `analytics_dashboard.py` (label rename)
- `mls_hospital_module.py` (adaptive pixel thresholds, fallback image, **3-tab surgery**, **wound-marker incision drill**, **suture lab**)
- `suture_module.py` (new — 8 suture patterns + schematic renderer + scoring)
- `instructions_module.py` (wound-marker docs, suture lab tab, full pattern table)
- `run_hospital.bat`, `run_hospital.sh` (OpenCV version pin)

## Files added
- `test_smoke.py` — smoke test suite (now 14 tests)
- `suture_module.py` — Suture Lab reference gallery + simulator
- `CHANGELOG.md` — this file
