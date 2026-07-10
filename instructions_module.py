import streamlit as st
from shared_session import dept_header


def render(student_id: str):
    dept_header("Instructions & Setup Guide", "📖", student_id)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Quick Start", "🖨️ ArUco Markers", "🔊 Ultrasound", "🔪 Surgery & FLS", "🩻 CT & DICOM"])

    with tab1:
        st.markdown("""
## Quick Start (5 minutes)
```bash
pip install "streamlit>=1.30" "opencv-contrib-python>=4.7" numpy scipy pillow plotly pydicom pandas
streamlit run hospital_portal.py
```
1. Open http://localhost:8501
2. Register → write down your Student ID
3. Choose a department from the lobby
4. Enter as Guest to try without saving

## What each department does
| Department | What you train |
|---|---|
| 🔊 Ultrasound | Probe positioning on organ phantom zones |
| 🔪 Surgery | Instrument tremor and incision precision |
| 🧲 FLS | Two-hand peg transfer timing |
| 🩻 CT | Slice, segment and upload real DICOM |
| 🔭 Endoscopy | Scope navigation, biopsy and retroflexion |
| 📊 Analytics | Your scores, charts and badges |
        """)

    with tab2:
        st.markdown("""
## Print ArUco Markers
Go to **https://chev.me/arucogen/** → Dictionary: **4x4** → Size: **100mm** → Print at 100%

| ID | Tape to | Module |
|---|---|---|
| 0 | Probe TIP | Ultrasound |
| 1 | Probe TAIL | Ultrasound |
| 2 | Surgical tip | Surgery |
| 3 | LEFT instrument | FLS |
| 4 | RIGHT instrument | FLS |
| 5 | Scope tip | Endoscopy |
| 6 | Scope handle | Endoscopy |
| 10 | Phantom TOP-LEFT | All |
| 11 | Phantom TOP-RIGHT | All |
| 12 | Phantom BOTTOM-RIGHT | All |
| 13 | Phantom BOTTOM-LEFT | All |

## Phantom setup
A **phantom** is any physical object acting as the patient body:
- **$0 option:** thick hardcover book
- **Better:** foam yoga block or kitchen sponge
- **Best:** gelatin block (10g gelatin powder + 500ml water)

Tape corner markers IDs 10–13 to the 4 corners of your phantom.

## ⚠️ ALWAYS print on paper — never show on screen
Screens emit light that causes glare. ArUco detection only works on flat printed paper.
        """)

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

    with tab4:
        st.markdown("""
## Surgery Suite
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

    with tab5:
        st.markdown("""
## CT Reading Room

### Built-in cases (no files needed)
Brain / Chest / Abdomen / Bone / Kidney / Liver — all synthetic, HU-accurate

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
