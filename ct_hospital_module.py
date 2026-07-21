"""CT Reading Room — slice view, multi-planar reconstruction, virtual knife, manual segmentation.

TotalSegmentator auto-segmentation is disabled for free-tier deployment (Streamlit Cloud
has only 1 GB RAM, TotalSegmentator needs 4-8 GB). The manual HU-threshold painter in the
Segmentation tab still works and is the main training tool. To re-enable AI auto-seg,
host the app somewhere with 4+ GB RAM (Render $7/mo, Oracle Cloud Free 24 GB, etc.)
and set AUTOSEG_OK = True in the import block below.
"""

import streamlit as st
import numpy as np
import math, io, zipfile, datetime
from PIL import Image
from shared_session import dept_header, record_session

# ---- DICOM support (pydicom) ----
try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    DICOM_OK = True
except ImportError:
    DICOM_OK = False

# ---- TotalSegmentator support (DISABLED on free tier — see file docstring) ----
AUTOSEG_OK = False
try:
    import nibabel as nib
    # from totalsegmentator.python_api import totalsegmentator  # not loaded
except ImportError:
    pass


# ── Vectorized synthetic volume generators (~100x faster than nested loops) ──

def _make_brain():
    nz, ny, nx = 80, 128, 128
    cz, cy, cx = 40, 64, 64
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    r = np.sqrt((zz - cz) ** 2 / 4 + (yy - cy) ** 2 + (xx - cx) ** 2)
    v = np.full((nz, ny, nx), -1000.0, dtype=np.float32)
    v[r < 50] = 30.0
    v[r < 40] = 40.0
    v[r < 20] = 45.0
    return v


def _make_chest():
    nz, ny, nx = 100, 128, 128
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    r = np.sqrt((yy - 64) ** 2 + (xx - 64) ** 2)
    v = np.full((nz, ny, nx), -800.0, dtype=np.float32)
    v[r < 45] = -500.0
    v[r < 15] = 50.0
    rib_mask = (r > 45) & (r < 55) & (zz < 80)
    v[rib_mask] = 600.0
    return v


def _make_abdomen():
    nz, ny, nx = 90, 128, 128
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    v = np.full((nz, ny, nx), -100.0, dtype=np.float32)
    liver  = ((xx - 50) ** 2 / 900 + (yy - 55) ** 2 / 400 + (zz - 45) ** 2 / 400) < 1
    v[liver] = 60.0
    spleen = ((xx - 80) ** 2 / 225 + (yy - 50) ** 2 / 225 + (zz - 40) ** 2 / 225) < 1
    v[spleen] = 50.0
    aorta  = ((xx - 64) ** 2 + (yy - 78) ** 2) < 25
    v[aorta] = 250.0
    return v


def _make_bone():
    nz, ny, nx = 80, 128, 128
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    r = np.sqrt((yy - 64) ** 2 + (xx - 64) ** 2)
    v = np.full((nz, ny, nx), -200.0, dtype=np.float32)
    v[(r > 28) & (r < 38)] = 800.0
    v[r < 10] = 50.0
    return v


def _make_kidney():
    nz, ny, nx = 80, 128, 128
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    v = np.full((nz, ny, nx), -100.0, dtype=np.float32)
    for ox in (40, 88):
        r = np.sqrt((xx - ox) ** 2 / 400 + (yy - 64) ** 2 / 900 + (zz - 40) ** 2 / 900)
        v[r < 1] = -50.0
        v[(r > 0.3) & (r < 1)] = 0.0
        v[(r > 0.6) & (r < 1)] = 30.0
    return v


def _make_liver():
    nz, ny, nx = 80, 128, 128
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    liver = ((xx - 52) ** 2 / 1600 + (yy - 58) ** 2 / 900 + (zz - 40) ** 2 / 900) < 1
    v = np.full((nz, ny, nx), -100.0, dtype=np.float32)
    v[liver] = 62.0
    portal = liver & (np.abs(xx - 52) < 4) & (np.abs(yy - 58) < 4)
    v[portal] = 80.0
    return v


CASES = {
    "🧠 Brain":   (_make_brain,   "Brain / Neuro"),
    "🫁 Chest":   (_make_chest,   "Chest / Thorax"),
    "🫀 Abdomen": (_make_abdomen, "Abdomen"),
    "🦴 Bone":    (_make_bone,    "MSK / Bone"),
    "🫘 Kidney":  (_make_kidney,  "Kidney detail"),
    "🫀 Liver":   (_make_liver,   "Liver & vessels"),
}


WINDOWS = {
    "Brain":        (40,  80),
    "Lung":         (-600, 1500),
    "Bone":         (400, 1800),
    "Soft Tissue":  (50,  350),
    "Abdomen":      (60,  400),
}


def _slice_to_img(arr2d, wl, ww):
    lo, hi = wl - ww / 2, wl + ww / 2
    clipped = np.clip(arr2d, lo, hi)
    scaled = ((clipped - lo) / (ww + 1e-6) * 255).astype(np.uint8)
    return Image.fromarray(scaled)


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Defensive: pydicom pixel_array can be 2D, 3D, or 4D depending on modality
    and frame count. Downstream code assumes 3D (slices, rows, cols).
    - 2D  -> add a leading axis
    - 3D  -> assume (slices, rows, cols) and return as-is
    - 4D  -> squeeze leading singleton (e.g. single-frame 4D cine)
    - 5+D -> keep squeezing leading axes until 3D
    """
    while arr.ndim > 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            # Multi-frame 4D: take the first volume
            arr = arr[0]
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    return arr


def _extract_spacing(ds):
    """Read (row_mm, col_mm, slice_mm) from a pydicom dataset."""
    try:
        ps = getattr(ds, "PixelSpacing", [0.5, 0.5])
        row_mm = float(ps[0])
        col_mm = float(ps[1])
    except (TypeError, ValueError, IndexError):
        row_mm = col_mm = 0.5
    try:
        st_mm = float(getattr(ds, "SliceThickness", 0.5))
    except (TypeError, ValueError):
        st_mm = 0.5
    return (row_mm, col_mm, st_mm)


def _load_dicom_zip(raw_bytes):
    """Load DICOM slices, sort by InstanceNumber/SliceLocation — not filename."""
    if not DICOM_OK:
        st.error("pydicom not installed. Run: pip install pydicom")
        return None, None
    loaded = []
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".dcm"):
                continue
            data = zf.read(name)
            try:
                ds = pydicom.dcmread(io.BytesIO(data))
                arr = ds.pixel_array.astype(np.float32)
                slope = float(getattr(ds, "RescaleSlope", 1))
                inter = float(getattr(ds, "RescaleIntercept", 0))
                arr = arr * slope + inter
                arr = _ensure_3d(arr)
                sort_key = (
                    getattr(ds, "InstanceNumber", None)
                    or getattr(ds, "SliceLocation", None)
                    or name
                )
                loaded.append((sort_key, arr, ds))
            except Exception:
                pass
    if not loaded:
        st.error("No valid DICOM slices found in ZIP.")
        return None, None

    def sortfunc(item):
        k = item[0]
        try:
            return (0, float(k))
        except (TypeError, ValueError):
            return (1, str(k))

    loaded.sort(key=sortfunc)
    vol = np.concatenate([item[1] for item in loaded], axis=0).astype(np.float32)
    vol = _ensure_3d(vol)
    spacing = _extract_spacing(loaded[0][2])
    return vol, spacing


def render(student_id: str):
    dept_header("CT Reading Room", "🩻", student_id)

    tab_load, tab_mpr, tab_knife, tab_seg = st.tabs(
        ["📂 Load Case", "🖼️ Multi-Planar", "🔪 Virtual Knife", "🎨 Segmentation"]
    )

    with tab_load:
        st.subheader("Load a CT Case")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Built-in synthetic cases**")
            choice = st.selectbox("Choose case", list(CASES.keys()))
            if st.button("▶️ Load Case", key="load_synth"):
                fn, label = CASES[choice]
                with st.spinner(f"Generating {label}..."):
                    st.session_state.ct_vol = fn()
                    st.session_state.ct_label = label
                    st.session_state.ct_spacing = (0.5, 0.5, 0.5)
                st.success(
                    f"✅ {label} loaded — {st.session_state.ct_vol.shape[0]} slices"
                )

        with col2:
            st.markdown("**Upload real DICOM (.dcm or .zip)**")
            up = st.file_uploader("Upload DICOM ZIP", type=["zip", "dcm"])
            if up and st.button("📤 Load DICOM", key="load_dicom"):
                raw = up.getvalue()
                if up.name.lower().endswith(".zip"):
                    vol, spacing = _load_dicom_zip(raw)
                else:
                    if not DICOM_OK:
                        st.error("pydicom not installed")
                        vol, spacing = None, None
                    else:
                        ds = pydicom.dcmread(io.BytesIO(raw))
                        arr = ds.pixel_array.astype(np.float32)
                        arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(
                            getattr(ds, "RescaleIntercept", 0)
                        )
                        vol = _ensure_3d(arr)
                        spacing = _extract_spacing(ds)
                if vol is not None:
                    vol = _ensure_3d(vol)
                    st.session_state.ct_vol = vol
                    st.session_state.ct_label = up.name
                    st.session_state.ct_spacing = spacing
                    r, c, s = spacing
                    st.success(
                        f"✅ DICOM loaded — {vol.shape[0]} slices, "
                        f"{vol.shape[1]}×{vol.shape[2]}, "
                        f"spacing {r:.2f}×{c:.2f}×{s:.2f} mm"
                    )

        if "ct_vol" not in st.session_state:
            st.info("Load a case above to unlock the other tabs.")
            return

    vol = st.session_state.ct_vol
    vol = _ensure_3d(vol)
    st.session_state.ct_vol = vol
    spacing = st.session_state.get("ct_spacing", (0.5, 0.5, 0.5))
    nz, ny, nx = vol.shape
    voxel_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0

    with tab_mpr:
        st.subheader("Multi-Planar Reconstruction")
        st.caption(
            f"Volume spacing: {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} mm  "
            f"(1 voxel = {voxel_ml * 1000:.3f} mm³ = {voxel_ml:.4f} mL)"
        )
        win_name = st.selectbox("Window preset", list(WINDOWS.keys()), key="mpr_win")
        wl, ww = WINDOWS[win_name]
        c1, c2, c3 = st.columns(3)
        with c1:
            z = st.slider("Axial slice", 0, nz - 1, nz // 2, key="ax")
            img = _slice_to_img(vol[z], wl, ww)
            st.image(img, caption=f"Axial z={z}", use_column_width=True)
        with c2:
            y = st.slider("Coronal slice", 0, ny - 1, ny // 2, key="cor")
            img = _slice_to_img(vol[:, y, :], wl, ww)
            st.image(img, caption=f"Coronal y={y}", use_column_width=True)
        with c3:
            x = st.slider("Sagittal slice", 0, nx - 1, nx // 2, key="sag")
            img = _slice_to_img(vol[:, :, x], wl, ww)
            st.image(img, caption=f"Sagittal x={x}", use_column_width=True)
        if st.button("💾 Save CT session", key="ct_save_mpr"):
            record_session(student_id, "ct", 75.0, f"MPR:{st.session_state.ct_label}")
            st.success("Saved!")

    with tab_knife:
        st.subheader("Virtual Knife — drag cut plane through volume")
        axis = st.radio("Cut axis", ["Axial", "Coronal", "Sagittal"], horizontal=True)
        win_name2 = st.selectbox("Window preset", list(WINDOWS.keys()), key="knife_win")
        wl2, ww2 = WINDOWS[win_name2]
        if axis == "Axial":
            depth = st.slider("Depth (slice)", 0, nz - 1, nz // 2)
            sl = vol[depth]
        elif axis == "Coronal":
            depth = st.slider("Depth (slice)", 0, ny - 1, ny // 2)
            sl = vol[:, depth, :]
        else:
            depth = st.slider("Depth (slice)", 0, nx - 1, nx // 2)
            sl = vol[:, :, depth]
        st.image(_slice_to_img(sl, wl2, ww2),
                 caption=f"{axis} cut at {depth}", use_column_width=True)
        flat = sl.flatten()
        counts, _ = np.histogram(flat, bins=50)
        st.bar_chart({"HU distribution": counts})

    with tab_seg:
        st.subheader("Segmentation Painter")
        st.caption(f"Voxel volume = {voxel_ml:.4f} mL (from DICOM spacing)")
        HU_PRESETS = {
            "Liver":           (40,  80),
            "Kidney Cortex":   (20,  50),
            "Spleen":          (40,  60),
            "Fat":             (-120, -60),
            "Blood/Vessel":    (45,  80),
            "Soft Tissue":     (20,  80),
            "Cortical Bone":   (400, 2000),
            "Cancellous Bone": (100, 400),
            "Lung":            (-700, -500),
            "Air":             (-1100, -900),
        }
        preset = st.selectbox("Tissue preset", list(HU_PRESETS.keys()))
        lo, hi = HU_PRESETS[preset]
        z_seg = st.slider("Axial slice", 0, nz - 1, nz // 2, key="seg_z")
        win_s = st.selectbox("Window", list(WINDOWS.keys()), key="seg_win")
        wls, wws = WINDOWS[win_s]
        sl = vol[z_seg]
        base = _slice_to_img(sl, wls, wws).convert("RGB")
        mask = (sl >= lo) & (sl <= hi)
        overlay = np.array(base)
        overlay[mask] = [255, 140, 0]
        vol_ml_seg = float(mask.sum()) * voxel_ml
        st.image(Image.fromarray(overlay),
                 caption=f"{preset} overlay — {vol_ml_seg:.2f} mL",
                 use_column_width=True)
        st.metric("Segmented volume", f"{vol_ml_seg:.2f} mL")
        st.metric("HU range", f"{lo} to {hi}")
        if st.button("💾 Save segmentation session", key="seg_save"):
            record_session(
                student_id, "ct", 80.0,
                f"Seg:{preset} vol:{vol_ml_seg:.2f}mL"
            )
            st.success("Saved!")
