import streamlit as st
from shared_session import dept_header


def render(student_id: str):
    dept_header("Instructions & Setup Guide", "📖", student_id)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚀 Quick Start", "🖨️ ArUco Markers", "🔊 Ultrasound",
        "🔪 Surgery", "🪡 Suture Lab", "🩻 CT & DICOM"])

    # ── Quick Start ────────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
## Quick Start (5 minutes)
```bash
pip install "streamlit>=1.30" "opencv-contrib-python>=4.7" numpy scipy pillow plotly pydicom pandas
streamlit run hospital_portal.py
```
1. Open http://localhost:8501
2. Click **🚀 Enter Hospital** (guest mode — no signup)
3. Pick a department from the lobby
4. Print ArUco markers first if you want camera-based drills

## What each department does
| Department | What you train |
|---|---|
| 🔊 Ultrasound | Probe positioning on organ phantom zones |
| 🔪 Surgery | Incision along a marked wound, tremor analysis, suture practice |
| 🧲 FLS | Two-instrument peg transfer timing |
| 🩻 CT | Slice / segment / upload real DICOM |
| 🔭 Endoscopy | Scope navigation, biopsy, retroflexion |
| 📊 Analytics | Your scores, charts, badges |
| 🪡 Suture Lab | 8 common suture patterns with reference diagrams |
        """)

    # ── ArUco Markers ──────────────────────────────────────────────────────
    with tab2:
        st.markdown(r"""
## Print ArUco Markers
Go to **https://chev.me/arucogen/** → Dictionary: **4x4** → Size: **100mm** → Print at 100%

### Marker table

| ID | Where to tape it | Module |
|---|---|---|
| 0  | Probe TIP | Ultrasound |
| 1  | Probe TAIL | Ultrasound |
| 2  | Instrument tip (scalpel, needle holder, pen) | Surgery (all 3 tabs) |
| 3  | LEFT instrument | FLS |
| 4  | RIGHT instrument | FLS |
| 5  | Scope tip | Endoscopy |
| 6  | Scope handle | Endoscopy |
| 10 | Phantom TOP-LEFT | All |
| 11 | Phantom TOP-RIGHT | All |
| 12 | Phantom BOTTOM-RIGHT | All |
| 13 | Phantom BOTTOM-LEFT | All |
| **20** | **Wound endpoint A** | **Surgery — Incision drill** |
| **21** | **Wound endpoint B** | **Surgery — Incision drill** |

## Phantom setup
A **phantom** is any physical object acting as the patient body:
- **$0 option:** thick hardcover book
- **Better:** foam yoga block or kitchen sponge
- **Best:** gelatin block (10g gelatin powder + 500 ml water)

Tape corner markers IDs 10–13 to the 4 corners of your phantom.

## ⚠️ ALWAYS print on paper — never show on screen
Screens emit light that causes glare. ArUco detection only works on flat printed paper.
        """)

    # ── Ultrasound ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
## Ultrasound Lab
1. Tape ID 0 to probe tip, ID 1 to probe tail (same stick)
2. Tape IDs 10–13 to phantom corners
3. Select organ and target zone
4. Hold probe over phantom → click camera → score shown
5. "✅ Correct zone" = probe correctly placed

## Liver Couinaud Segments
| Segment | Location | Approach |
|---|---|---|
| I Caudate | Deep, behind portal vein | Steep, aim toward IVC |
| II–III | Left lobe | Subcostal flat |
| IV | Central | Slight left tilt |
| V–VIII | Right lobe | 45° subcostal or intercostal |
        """)

    # ── Surgery (Incision + Tremor) ───────────────────────────────────────
    with tab4:
        st.markdown(r"""
## Surgery Suite — 3 tabs

### Tab 1: Incision Drill (wound markers)
**The wound is a break in the skin** — a *cut*. To teach the simulator where
the cut is, you place **2 ArUco markers at the 2 ends of the cut**:

| Marker | Where |
|---|---|
| **ID 20** | One end of the cut (wound endpoint A) |
| **ID 21** | The other end (wound endpoint B) |
| **ID 2**  | Instrument tip |

The simulator draws a yellow line between markers 20 and 21 — **that is the wound**.
You then move the instrument along the line to make the incision.

**Why 2 markers, not 1?** A cut is a line, and a line needs 2 points to define
it. With 2 markers the simulator knows the wound's *position*, *length*, and
*orientation* — regardless of whether the cut is 1 cm or 20 cm.

**For a very small cut** (1-2 cm) the 2 markers will sit right next to each
other at the wound ends — that's fine, the math still works. If the cut is so
small that 2 printed markers would overlap, place them slightly off the wound
on either side, along the wound's long axis.

**No wound yet?** Draw a line on a piece of tape with a marker, then put
the 2 ArUco markers at the ends of the line. You can also use the corner
markers (ID 10/11) of the phantom as endpoints for a long wound.

The score is **40 % tremor, 40 % path-following, 20 % endpoint coverage**.

### Tab 2: Suture Lab
See the **🪡 Suture Lab** tab in the Instructions panel.

### Tab 3: Tremor Analysis (legacy)
- Tape ID 2 to pen tip (your instrument)
- Click camera → move slightly → click again → repeat 5+ times
- Tremor = standard deviation of frame-to-frame movement
- Hold completely still → score 90+. Shake → score drops.
- Score is normalized against frame width so it works on any camera.

## FLS Peg Transfer
- Tape ID 3 to LEFT instrument, ID 4 to RIGHT
- Bring both markers within 5% of frame width → handoff counted
- Rate per minute and total handoffs scored
        """)

    # ── Suture Lab (new tab) ──────────────────────────────────────────────
    with tab5:
        st.markdown(r"""
## 🪡 Suture Lab — pick a pattern, study it, practise it

The Suture Lab gives you **8 common suture patterns** with a reference
schematic for each. Pick a pattern, study the diagram, then practise the
bites in the simulator.

### How to use
1. Go to **Surgery Suite → 🪡 Suture Lab** tab
2. Pick a pattern from the dropdown — the reference schematic appears below
3. Read the **Indications** to understand when this suture is used clinically
4. Click **Try this pattern** to start the simulator
5. Hover the instrument (ID 2) over each numbered target dot, then **capture**
6. Each bite is scored by distance to the target. The session score is the average

### The 8 patterns

| # | Pattern | Bites / stitch | When to use |
|---|---|---|---|
| 1 | **Simple Interrupted** | 2 | General skin closure, laparotomy |
| 2 | **Simple Continuous (Running)** | 1 | Long incisions, peritoneum |
| 3 | **Continuous Locking (Ford)** | 1 | Fascia, areas under tension |
| 4 | **Figure-of-8** | 2 | Hemostasis, small defects, vessel ligation |
| 5 | **Vertical Mattress (Donati)** | 4 | Skin under tension, edge eversion |
| 6 | **Horizontal Mattress** | 4 | High-tension closures, tendons |
| 7 | **Subcuticular** | 1 | Cosmetic closures (face, neck) |
| 8 | **Purse-string** | 1 | Appendiceal stump, trocar sites, stoma |

### Scoring
- Each bite is scored `100 × (1 − distance / 18 % of frame)`
- Distance is measured in normalized frame coordinates (so it works on any camera)
- Excellent ≥ 90 · Good ≥ 75 · Fair ≥ 55 · Poor < 55 · Miss ≥ 18 %

### Tips
- The wound is shown as a virtual line in the camera frame — you don't need
  a real suturing pad for this drill
- For "Purse-string", the wound is a small circle in the middle
- Print & tape the **ID 2** marker to a pen, forceps, or actual needle holder
        """)

    # ── CT & DICOM ─────────────────────────────────────────────────────────
    with tab6:
        st.markdown("""
## CT Reading Room

### Built-in cases (no files needed)
🧠 Brain / 🫁 Chest / 🫀 Abdomen / 🦴 Bone / 🫘 Kidney / 🫀 Liver
All 6 are now **anatomical phantoms** with real structures (skull, ventricles,
lungs, heart, kidneys, vessels, ribs, etc.) — not nested ellipsoids anymore.

| Case | What you'll see |
|---|---|
| **Brain** | Skull, CSF, gray/white matter, ventricles, falx, sulci, cerebellum, brainstem |
| **Chest** | Body wall, 2 lungs, heart, aorta, trachea, spine, sternum, 10 ribs |
| **Abdomen** | Liver (with portal vein), spleen, 2 kidneys, aorta, IVC, spine |
| **Bone** | Cortical ring, cancellous interior with trabecular pattern, medullary canal |
| **Kidney** | Body wall + 2 kidneys with cortex, medulla, renal pelvis |
| **Liver** | Liver parenchyma with portal vein, hepatic veins, gallbladder, IVC |

### Upload real DICOM
1. Export CT series from PACS as ZIP of .dcm files
2. Enter CT Reading Room → Load Case tab → Upload DICOM ZIP
3. Slices are sorted by InstanceNumber / SliceLocation (DICOM tags), not by filename
4. Voxel volume is read from PixelSpacing + SliceThickness — no hardcoded 0.5 mm assumption

### HU Reference
| Tissue | HU |
|---|---|
| Air | −1000 |
| Lung | −600 to −400 |
| Fat | −100 to −50 |
| Soft tissue | 20 to 80 |
| Blood | 45 to 65 |
| Liver | 55 to 65 |
| Bone | 400+ |

### Free public DICOM datasets
- **cancerimagingarchive.net** — thousands of anonymised cases
- **radiopaedia.org/cases** — teaching cases
- Always anonymise patient data before sharing with students
        """)
