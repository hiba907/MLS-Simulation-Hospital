"""CT Reading Room — slice view, multi-planar reconstruction, virtual knife, manual segmentation.

ENHANCED Virtual Knife with proper aspect ratio, zoom, +/- step buttons, and 3D orientation inset.

Drop-in replacement for ct_hospital_module.py. Requires matplotlib in requirements.txt.

TotalSegmentator auto-segmentation is disabled for free-tier deployment (Streamlit Cloud
has only 1 GB RAM, TotalSegmentator needs 4-8 GB). The manual HU-threshold painter in the
Segmentation tab still works and is the main training tool. To re-enable AI auto-seg,
host the app somewhere with 4+ GB RAM (Render $7/mo, Oracle Cloud Free 24 GB, etc.)
and set AUTOSEG_OK = True in the import block below.
"""

import streamlit as st
import numpy as np
import math, io, zipfile, datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
except ImportError:
    pass


# ── Realistic synthetic CT generators with Perlin-like noise ──────────────────
# Uses multi-scale Gaussian-smoothed noise to produce organic, irregular tissue
# boundaries instead of perfect spheres. HU values are anatomically calibrated.
from scipy.ndimage import gaussian_filter as _gf

def _perlin_like_noise(shape, scales=(4, 8, 16, 32), weights=(0.45, 0.30, 0.18, 0.07), seed=42):
    """Multi-scale Gaussian-smoothed noise — fast Perlin approximation."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, dtype=np.float32)
    for sc, w in zip(scales, weights):
        n = rng.standard_normal(shape).astype(np.float32)
        sigma = max(shape[0] // sc, 1)
        n = _gf(n, sigma=sigma)
        std = n.std()
        if std > 0:
            n /= std
        out += w * n
    return out


def _make_brain():
    """Brain CT with Perlin-like noise: skull diploe, sulci, lateral/third
    ventricles, gray/white matter, falx cerebri, cerebellum, brainstem.
    HU: skull 820-1180, CSF 6-10, GM 35-45, WM 26-34, ventricles ~6."""
    import numpy as np
    nz, ny, nx = 100, 144, 144
    cz, cy, cx = 50, 72, 72
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    noise = _perlin_like_noise((nz, ny, nx), scales=(4, 8, 16, 32),
                               weights=(0.45, 0.30, 0.18, 0.07), seed=7)
    r_head = np.sqrt(((zz - cz) / 0.95) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    r_noisy = r_head - 2.5 * noise
    v = np.full((nz, ny, nx), -1000.0, dtype=np.float32)
    R_SKULL_OUT, R_SKULL_IN, R_CSF_IN, R_GRAY_IN = 60, 57, 51, 44
    skull_m = (r_noisy >= R_SKULL_IN) & (r_noisy < R_SKULL_OUT)
    v[skull_m] = 1000.0 + 180.0 * noise[skull_m]
    csf_m = (r_noisy >= R_CSF_IN) & (r_noisy < R_SKULL_IN)
    v[csf_m] = 8.0 + 2.0 * noise[csf_m]
    brain_m = r_noisy < R_CSF_IN
    v[brain_m] = 38.0
    wm = brain_m & (r_noisy < R_GRAY_IN)
    v[wm] = 30.0 + 4.0 * noise[wm]
    gm = brain_m & (r_noisy >= R_GRAY_IN)
    v[gm] = 40.0 + 5.0 * noise[gm]
    ang = np.arctan2(yy - cy, xx - cx)
    sulci = (0.6 * np.cos(8 * ang + 0.5 * np.sin(3 * zz / nz * np.pi)) +
             0.4 * np.cos(5 * ang + 1.7) +
             0.3 * np.cos(11 * ang - 0.7 * zz / nz * np.pi))
    sulci_m = brain_m & (r_noisy > R_CSF_IN - 5) & (r_noisy < R_CSF_IN - 0.5)
    v[sulci_m] = np.maximum(v[sulci_m] - sulci[sulci_m] * 7 - 3 * noise[sulci_m], 8.0)
    falx = brain_m & (np.abs(xx - cx) < 1.8) & (zz < 68) & (zz > 30)
    v[falx] = 68.0 + 6.0 * noise[falx]
    lv_l = brain_m & (((yy-65)/9)**2 + ((xx-60)/4)**2 + ((zz-48)/11)**2 - 0.15*noise < 1)
    lv_r = brain_m & (((yy-65)/9)**2 + ((xx-84)/4)**2 + ((zz-48)/11)**2 - 0.15*noise < 1)
    v[lv_l | lv_r] = 6.0 + 2.0 * noise[lv_l | lv_r]
    v3 = brain_m & (((yy-75)/2)**2 + ((xx-cx)/2)**2 + ((zz-42)/5)**2 - 0.10*noise < 1)
    v[v3] = 6.0 + 2.0 * noise[v3]
    cereb = brain_m & (((zz-35)/11)**2 + ((yy-80)/10)**2 + ((xx-cx)/14)**2 - 0.10*noise < 1)
    v[cereb] = 35.0 + 4.0 * noise[cereb]
    bs = brain_m & (np.abs(xx-cx) < 5) & (np.abs(yy-75) < 4) & (zz < 32) & (zz > 18)
    v[bs] = 30.0 + 3.0 * noise[bs]
    rng = np.random.default_rng(99)
    v += rng.normal(0, 5, v.shape).astype(np.float32)   # scanner noise ~5 HU
    return v


def _make_chest():
    """Chest CT: body wall (fat + muscle), 2 lungs with vascular texture,
    heart, aorta, trachea, spine with canal, sternum, 10 ribs.
    All boundaries perturbed by Perlin-like noise."""
    import numpy as np
    nz, ny, nx = 110, 144, 144
    cz, cy, cx = 55, 72, 72
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    noise = _perlin_like_noise((nz, ny, nx), scales=(4, 8, 16),
                               weights=(0.5, 0.32, 0.18), seed=13)
    r_body = np.sqrt(((zz-cz)/1.2)**2 + (yy-cy)**2 + (xx-cx)**2)
    r_noisy = r_body - 2.0 * noise
    v = np.full((nz, ny, nx), -1000.0, dtype=np.float32)
    R_SKIN, R_MUSC, R_LUNG = 52, 49, 44
    fat_m  = (r_noisy >= R_MUSC) & (r_noisy < R_SKIN)
    musc_m = (r_noisy >= R_LUNG) & (r_noisy < R_MUSC)
    v[fat_m]  = -90.0 + 15.0 * noise[fat_m]
    v[musc_m] =  45.0 + 12.0 * noise[musc_m]
    lung_l = ((((yy-65)/14)**2 + ((xx-56)/10)**2 + ((zz-50)/26)**2 - 0.10*noise) < 1) & (r_noisy < R_LUNG)
    lung_r = ((((yy-65)/14)**2 + ((xx-88)/10)**2 + ((zz-50)/26)**2 - 0.10*noise) < 1) & (r_noisy < R_LUNG)
    v[lung_l] = -800.0 + 90.0 * np.abs(noise[lung_l])
    v[lung_r] = -800.0 + 90.0 * np.abs(noise[lung_r])
    heart = ((((yy-70)/11)**2 + ((xx-70)/10)**2 + ((zz-50)/14)**2 - 0.12*noise) < 1) & (r_noisy < R_LUNG)
    v[heart] = 40.0 + 12.0 * noise[heart]
    aorta = (((xx-70)/4)**2 + ((yy-80)/4)**2 < 1) & (zz > 35) & (zz < 75)
    v[aorta] = 60.0 + 10.0 * noise[aorta]
    trachea = (((xx-cx)/3)**2 + ((yy-58)/3)**2 < 1) & (zz > 55) & (zz < 100)
    v[trachea] = -950.0
    spy = cy + 45
    spine_m = ((((xx-cx)/7)**2 + ((yy-spy)/8)**2 - 0.12*noise) < 1) & (r_noisy < R_LUNG)
    v[spine_m] = 700.0 + 120.0 * noise[spine_m]
    canal_m = (((xx-cx)/3)**2 + ((yy-spy)/3)**2 < 1) & (r_noisy < R_LUNG)
    v[canal_m] = 10.0
    stern = (np.abs(xx-cx) < 4) & ((yy-cy) < -30) & (zz > 35) & (zz < 90) & (r_noisy < R_MUSC)
    v[stern] = 500.0 + 80.0 * noise[stern]
    for rzc in np.linspace(30, 95, 10):
        rib_ang = np.arctan2(yy-cy, xx-cx)
        rib_d = np.sqrt((xx-cx)**2 + (yy-cy)**2)
        rib_b = (np.abs(zz-rzc) < 2.5) & (rib_d > 28) & (rib_d < 50) & (r_noisy < R_SKIN)
        rib_b = rib_b & (np.abs(np.sin(rib_ang)) > 0.3)
        v[rib_b] = 600.0 + 90.0 * noise[rib_b]
    rng = np.random.default_rng(77)
    v += rng.normal(0, 6, v.shape).astype(np.float32)
    return v


def _make_abdomen():
    """Abdomen CT: body wall, liver + portal vein, spleen, 2 kidneys,
    aorta, IVC, spine, visceral fat — all with Perlin noise boundaries."""
    import numpy as np
    nz, ny, nx = 110, 144, 144
    cz, cy, cx = 55, 72, 72
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    noise = _perlin_like_noise((nz, ny, nx), scales=(4, 8, 16),
                               weights=(0.5, 0.32, 0.18), seed=21)
    r_body = np.sqrt(((zz-cz)/1.1)**2 + (yy-cy)**2 + (xx-cx)**2)
    r_noisy = r_body - 2.2 * noise
    v = np.full((nz, ny, nx), -1000.0, dtype=np.float32)
    R_SKIN, R_MUSC, R_VISC = 55, 51, 46
    v[(r_noisy >= R_MUSC) & (r_noisy < R_SKIN)] = -100.0 + 20.0 * noise[(r_noisy >= R_MUSC) & (r_noisy < R_SKIN)]
    v[(r_noisy >= R_VISC) & (r_noisy < R_MUSC)] =   52.0 + 12.0 * noise[(r_noisy >= R_VISC) & (r_noisy < R_MUSC)]
    inside = r_noisy < R_VISC
    v[inside] = 50.0 + 10.0 * noise[inside]
    liver_r = (((yy-60)/22)**2 + ((xx-85)/18)**2 + ((zz-55)/26)**2 - 0.10*noise)
    liver = (liver_r < 1) & inside
    v[liver] = 58.0 + 8.0 * noise[liver]
    pv = liver & (np.abs(xx-82) < 3) & (zz > 40) & (zz < 70)
    v[pv] = 130.0 + 20.0 * noise[pv]
    spleen_r = (((yy-60)/16)**2 + ((xx-58)/12)**2 + ((zz-55)/18)**2 - 0.10*noise)
    spleen = (spleen_r < 1) & inside
    v[spleen] = 50.0 + 7.0 * noise[spleen]
    lk = ((((yy-68)/10)**2 + ((xx-92)/8)**2 + ((zz-58)/12)**2 - 0.10*noise) < 1) & inside
    rk = ((((yy-68)/10)**2 + ((xx-52)/8)**2 + ((zz-56)/12)**2 - 0.10*noise) < 1) & inside
    v[lk] = 40.0 + 8.0 * noise[lk]
    v[rk] = 40.0 + 8.0 * noise[rk]
    spy = cy + 30
    aorta = (((xx-cx)/5)**2 + ((yy-spy)/5)**2 < 1) & inside
    v[aorta] = 200.0 + 30.0 * noise[aorta]
    ivc = (((xx-cx-8)/5)**2 + ((yy-spy)/5)**2 < 1) & inside
    v[ivc] = 40.0 + 8.0 * noise[ivc]
    spine_y = cy + 40
    spine_m = ((((xx-cx)/7)**2 + ((yy-spine_y)/8)**2 - 0.12*noise) < 1) & (r_noisy < R_MUSC)
    v[spine_m] = 680.0 + 110.0 * noise[spine_m]
    canal_m = (((xx-cx)/3)**2 + ((yy-spine_y)/3)**2 < 1) & (r_noisy < R_MUSC)
    v[canal_m] = 10.0
    rng = np.random.default_rng(33)
    v += rng.normal(0, 5, v.shape).astype(np.float32)
    return v


def _make_bone():
    """Long bone CT cross-section: cortical ring (900+ HU), trabecular
    lattice (300-500 HU), medullary canal (-80 HU fat marrow) — Perlin noise."""
    import numpy as np
    nz, ny, nx = 80, 128, 128
    cz, cy, cx = 40, 64, 64
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    noise = _perlin_like_noise((nz, ny, nx), scales=(4, 8, 16),
                               weights=(0.5, 0.32, 0.18), seed=55)
    v = np.full((nz, ny, nx), -1000.0, dtype=np.float32)
    r2d = np.sqrt(((yy-cy)/0.72)**2 + (xx-cx)**2) - 1.8 * noise[nz//2, :, :]
    r3d = np.broadcast_to(r2d[np.newaxis], (nz, ny, nx)).copy()
    z_frac = (zz - cz) / cz
    r3d += 1.5 * z_frac**2   # slight taper
    R_COUT, R_CIN, R_MED = 35, 29, 12
    cortex_m  = r3d < R_COUT
    trabecu_m = r3d < R_CIN
    medull_m  = r3d < R_MED
    v[cortex_m]  = 900.0 + 100.0 * noise[cortex_m]
    trab_pat = 0.5 * (np.sin(12*np.pi*(xx-cx)/nx) * np.sin(12*np.pi*(yy-cy)/ny) + 1)
    v[trabecu_m] = 300.0 + 200.0 * trab_pat[trabecu_m] + 50.0 * noise[trabecu_m]
    v[medull_m]  = -80.0 + 20.0 * noise[medull_m]
    rng = np.random.default_rng(66)
    v += rng.normal(0, 4, v.shape).astype(np.float32)
    return v


def _make_kidney():
    """Detailed kidney CT: perinephric fat, cortex, medullary pyramids,
    renal pelvis/sinus — bilateral — with Perlin noise organ surfaces."""
    import numpy as np
    nz, ny, nx = 80, 128, 128
    cz, cy, cx = 40, 64, 64
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    noise = _perlin_like_noise((nz, ny, nx), scales=(4, 8, 16),
                               weights=(0.5, 0.32, 0.18), seed=88)
    v = np.full((nz, ny, nx), -200.0, dtype=np.float32)
    for dx in [-25, 25]:
        kr = (((xx-cx-dx)/18)**2 + ((yy-cy)/22)**2 + ((zz-cz)/20)**2 - 0.08*noise)
        cortex_m = kr < 1.0
        medull_m = kr < 0.75
        pelvis_m = kr < 0.40
        v[cortex_m] = 30.0 + 10.0 * noise[cortex_m]
        v[medull_m] = 20.0 +  6.0 * noise[medull_m]
        v[pelvis_m] = -50.0 + 5.0 * noise[pelvis_m]
    rng = np.random.default_rng(44)
    v += rng.normal(0, 4, v.shape).astype(np.float32)
    return v


def _make_liver():
    """Detailed liver CT: parenchyma, portal vein + branches, 3 hepatic
    veins, gallbladder, IVC — all with Perlin noise organ surfaces."""
    import numpy as np
    nz, ny, nx = 90, 144, 144
    cz, cy, cx = 45, 72, 72
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    noise = _perlin_like_noise((nz, ny, nx), scales=(4, 8, 16),
                               weights=(0.5, 0.32, 0.18), seed=11)
    r_body = np.sqrt(((zz-cz)/1.1)**2 + (yy-cy)**2 + (xx-cx)**2)
    v = np.full((nz, ny, nx), 30.0, dtype=np.float32)
    R_BODY, R_FAT, R_MUSC = 58, 54, 50
    fat_m  = (r_body >= R_MUSC) & (r_body < R_FAT)
    musc_m = (r_body >= R_FAT)  & (r_body < R_BODY)
    v[fat_m]  = -80.0 + 15.0 * noise[fat_m]
    v[musc_m] =  10.0 + 8.0  * noise[musc_m]
    liver_r = (((xx-56)/30)**2 + ((yy-65)/22)**2 + ((zz-45)/22)**2 - 0.08*noise)
    liver = (liver_r < 1) & (r_body < R_MUSC - 1)
    v[liver] = 62.0 + 8.0 * noise[liver]
    portal_m = liver & (np.abs(xx-56) < 3) & (np.abs(yy-65) < 3) & (zz > 40) & (zz < 60)
    v[portal_m] = 35.0 + 6.0 * noise[portal_m]
    for direction in (-1, 1):
        branch = liver & (np.abs(yy-(65+direction*10)) < 2) & (np.abs(xx-56) < 2) & (zz > 40) & (zz < 60)
        v[branch] = 35.0 + 5.0 * noise[branch]
    for vx, vy in ((46, 55), (66, 55), (56, 60)):
        hepatic = liver & (((xx-vx)/3)**2 + ((yy-vy)/3)**2 + ((zz-50)/6)**2 - 0.06*noise < 1)
        v[hepatic] = 30.0 + 4.0 * noise[hepatic]
    gb = (((xx-48)/5)**2 + ((yy-80)/4)**2 + ((zz-50)/8)**2 - 0.06*noise < 1) & liver
    v[gb] = 5.0 + 2.0 * noise[gb]
    ivc = (((xx-70)/3)**2 + ((yy-80)/3)**2 + ((zz-45)/20)**2 - 0.06*noise < 1) & (r_body < R_MUSC)
    v[ivc] = 45.0 + 6.0 * noise[ivc]
    rng = np.random.default_rng(22)
    v += rng.normal(0, 5, v.shape).astype(np.float32)
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

# ── Tissue HU reference table for Virtual Knife annotations ──
TISSUE_BANDS = [
    # (name, lo_HU, hi_HU, label_center, color)
    ("Air",           -1100, -900,  -1000, "lightblue"),
    ("Lung",           -700, -500,   -600, "skyblue"),
    ("Fat",            -120,  -60,    -90, "lightyellow"),
    ("Water",            -5,    5,      0, "cyan"),
    ("Soft Tissue",      20,   80,     50, "lightgreen"),
    ("Liver",            40,   80,     60, "green"),
    ("Cortical Bone",   400, 2000,    700, "lightcoral"),
]


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


# ── Helper: annotated HU histogram with tissue bands ──

def _render_annotated_histogram(sl_flat, show_labels=True):
    """Draw a matplotlib HU histogram with optional tissue band annotations.
    Returns the matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    fig.set_dpi(150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Draw histogram
    counts, bin_edges, patches = ax.hist(
        sl_flat, bins=80, color="#3D5A80", alpha=0.75, edgecolor="none"
    )
    ax.set_xlabel("Hounsfield Units (HU)", fontsize=10)
    ax.set_ylabel("Pixel count", fontsize=10)
    ax.set_title("HU Distribution — Current Slice", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    y_max = counts.max() if counts.max() > 0 else 1.0

    if show_labels:
        for name, lo, hi, center, color in TISSUE_BANDS:
            ax.axvspan(lo, hi, alpha=0.18, color=color, zorder=0)
            # Only show label text if the band is within the data x-range
            x_min_data = sl_flat.min()
            x_max_data = sl_flat.max()
            if center >= x_min_data - 200 and center <= x_max_data + 200:
                ax.text(
                    center, y_max * 0.88,
                    name,
                    ha="center", va="top",
                    fontsize=6.5, color="#333333",
                    rotation=90,
                    clip_on=True,
                )

    return fig


def _detect_tissues_in_slice(sl_flat):
    """Return list of tissue names whose HU band has ≥1 pixel in the slice."""
    detected = []
    for name, lo, hi, center, color in TISSUE_BANDS:
        if np.any((sl_flat >= lo) & (sl_flat <= hi)):
            detected.append(name)
    return detected


# ── Helper: 3-view orientation inset (axial / coronal / sagittal) with crosshairs ──

def _render_3view_inset(vol, axis, depth, wl, ww, nz, ny, nx):
    """Render three small thumbnails with crosshairs showing 3D orientation.
    The active view (the one currently displayed in the main panel) has a red
    border and red crosshairs at the cut position on the other two axes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    fig.patch.set_facecolor("white")

    # Positions on each axis
    cur_z = depth if axis == "Axial" else nz // 2
    cur_y = depth if axis == "Coronal" else ny // 2
    cur_x = depth if axis == "Sagittal" else nx // 2

    views = [
        ("Axial",    vol[cur_z],        f"Axial z={cur_z}"),
        ("Coronal",  vol[:, cur_y, :],  f"Coronal y={cur_y}"),
        ("Sagittal", vol[:, :, cur_x],  f"Sagittal x={cur_x}"),
    ]

    for ax, view_axis, sl_data, title in zip(axes, [v[0] for v in views],
                                              [v[1] for v in views],
                                              [v[2] for v in views]):
        is_active = (view_axis == axis)
        ax.imshow(sl_data, cmap="gray",
                  vmin=wl - ww / 2, vmax=wl + ww / 2,
                  aspect="equal")
        if is_active:
            # Red crosshairs showing where the cut is on the OTHER two axes
            if view_axis == "Axial":
                ax.axhline(cur_y, color="red", linewidth=0.9, alpha=0.85)
                ax.axvline(cur_x, color="red", linewidth=0.9, alpha=0.85)
            elif view_axis == "Coronal":
                ax.axhline(cur_z, color="red", linewidth=0.9, alpha=0.85)
                ax.axvline(cur_x, color="red", linewidth=0.9, alpha=0.85)
            else:  # Sagittal
                ax.axhline(cur_z, color="red", linewidth=0.9, alpha=0.85)
                ax.axvline(cur_y, color="red", linewidth=0.9, alpha=0.85)
            # Red border on active view
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2.0)
        ax.set_title(title, fontsize=9,
                     fontweight="bold" if is_active else "normal",
                     color="red" if is_active else "black")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render(student_id: str):
    dept_header("CT Reading Room", "🩻", student_id)

    tab_load, tab_mpr, tab_knife, tab_seg, tab_code, tab_challenge, tab_3d = st.tabs(
        ["📂 Load Case", "🖼️ Multi-Planar", "🔪 Virtual Knife",
         "🎨 Segmentation", "🐍 Code Lab", "🎯 Challenges", "🧊 3-D Volume"]
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
                    st.session_state["ct_volume"] = st.session_state.ct_vol  # for 3D tab
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
                    st.session_state["ct_volume"] = vol  # for 3D tab
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
    st.session_state["ct_volume"] = vol  # for 3D tab
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

    # ═══════════════════════════════════════════════════════════════════════════
    # ENHANCED Virtual Knife — aspect ratio, zoom, +/- buttons, 3D orientation
    # ═══════════════════════════════════════════════════════════════════════════

    # Lazy scipy import for z-interpolation (optional feature)
    try:
        from scipy.ndimage import zoom as scipy_zoom
        SCIPY_OK = True
    except ImportError:
        SCIPY_OK = False

    with tab_knife:
        st.subheader("🔪 Virtual Knife — Interactive CT Slicer")

        with st.expander("ℹ️ How to use the knife", expanded=False):
            st.markdown(
                """
**The knife is a plane that cuts through the 3D CT volume.**

1. **Pick a cut axis** — Axial (top-down), Coronal (front-back), or Sagittal (left-right).
2. **Move the cut plane** — drag the slider, or use the ◀ ▶ buttons to step one slice at a time, or click ⟲ to jump back to the middle.
3. **Zoom** — use + / − / Fit to inspect anatomy in detail.
4. **Check the 3D Orientation panel below** — three small thumbnails (axial / coronal / sagittal) with red crosshairs showing where your cut is in the volume. The active view has a red border.
5. **Read the HU histogram** — identify which tissues appear in the current slice.

💡 *Think of it like slicing a loaf of bread: the axis is the direction of the cut, the depth is which slice you're looking at.*
                """
            )

        # Axis + window
        col_axis, col_win = st.columns([1, 2])
        with col_axis:
            axis = st.radio(
                "Cut axis",
                ["Axial", "Coronal", "Sagittal"],
                horizontal=True,
                key="knife_axis",
            )
        with col_win:
            win_name2 = st.selectbox(
                "Window preset", list(WINDOWS.keys()), key="knife_win"
            )
        wl2, ww2 = WINDOWS[win_name2]

        # Determine slice range for the selected axis
        if axis == "Axial":
            max_depth = nz - 1
            default_depth = nz // 2
        elif axis == "Coronal":
            max_depth = ny - 1
            default_depth = ny // 2
        else:
            max_depth = nx - 1
            default_depth = nx // 2

        # Per-axis depth in session state (so flipping axes keeps your position)
        depth_key = f"knife_depth_{axis.lower()}"
        if depth_key not in st.session_state:
            st.session_state[depth_key] = default_depth
        # Clamp in case volume shape changed
        st.session_state[depth_key] = max(0, min(max_depth, st.session_state[depth_key]))

        # Depth control: prev / slider / next / reset
        st.markdown("##### Cut depth")
        col_prev, col_slider, col_next, col_reset = st.columns([1, 6, 1, 1])
        with col_prev:
            if st.button("◀", key="knife_prev", help="Previous slice"):
                st.session_state[depth_key] = max(
                    0, st.session_state[depth_key] - 1
                )
        with col_next:
            if st.button("▶", key="knife_next", help="Next slice"):
                st.session_state[depth_key] = min(
                    max_depth, st.session_state[depth_key] + 1
                )
        with col_reset:
            if st.button("⟲", key="knife_reset", help="Jump to middle"):
                st.session_state[depth_key] = default_depth
        with col_slider:
            depth = st.slider(
                "Slice",
                0,
                max_depth,
                st.session_state[depth_key],
                key=depth_key,
                label_visibility="collapsed",
            )

        # Zoom control
        zoom_key = "knife_zoom"
        if zoom_key not in st.session_state:
            st.session_state[zoom_key] = 1.0
        zoom = st.session_state[zoom_key]

        st.markdown("##### Zoom")
        col_z1, col_z2, col_z3, col_z4 = st.columns([1, 1, 1, 3])
        with col_z1:
            if st.button("−", key="zoom_out", help="Zoom out"):
                st.session_state[zoom_key] = max(
                    0.5, st.session_state[zoom_key] - 0.25
                )
        with col_z2:
            if st.button("+", key="zoom_in", help="Zoom in"):
                st.session_state[zoom_key] = min(
                    4.0, st.session_state[zoom_key] + 0.25
                )
        with col_z3:
            if st.button("Fit", key="zoom_fit", help="Reset zoom"):
                st.session_state[zoom_key] = 1.0
        with col_z4:
            st.markdown(f"**{st.session_state[zoom_key]:.2f}×**")

        # Z-interpolation toggle — makes coronal/sagittal views square for
        # thin-z DICOMs (e.g. head CTs with only 30-60 z-slices). This is what
        # real DICOM viewers (Horos, 3D Slicer, your Chrome viewer) do by default.
        interp_key = "knife_z_interp"
        if interp_key not in st.session_state:
            st.session_state[interp_key] = False

        c_int1, c_int2 = st.columns([1, 3])
        with c_int1:
            interp_z = st.checkbox(
                "Z-interp",
                value=st.session_state[interp_key],
                key=interp_key,
                help="Interpolate slices in z so coronal/sagittal views look square. "
                     "Recommended for thin-z DICOMs (e.g. head CT).",
            )
        with c_int2:
            if interp_z and not SCIPY_OK:
                st.caption("⚠️ scipy not installed — skipping interpolation")
            elif interp_z and axis != "Axial":
                st.caption(
                    f"Original z={nz} → interpolated to ~{max(nz, ny, nx)} slices"
                )

        # Extract the slice for the selected axis/depth
        # Apply z-interpolation for coronal/sagittal to fix the thin-band issue
        if interp_z and SCIPY_OK and axis != "Axial":
            target_nz = max(nz, ny, nx)
            zoom_factor = target_nz / nz
            vol_view = scipy_zoom(vol, (zoom_factor, 1, 1), order=1)
        else:
            vol_view = vol

        if axis == "Axial":
            sl = vol_view[depth]
            slice_label = f"Axial cut at z={depth}"
        elif axis == "Coronal":
            sl = vol_view[:, depth, :]
            slice_label = f"Coronal cut at y={depth}"
        else:
            sl = vol_view[:, :, depth]
            slice_label = f"Sagittal cut at x={depth}"

        # Render with PIL + zoom — preserves aspect ratio
        img = _slice_to_img(sl, wl2, ww2)
        if zoom != 1.0:
            new_w = max(1, int(img.width * zoom))
            new_h = max(1, int(img.height * zoom))
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Main image + side info panel
        col_main, col_side = st.columns([3, 1])
        with col_main:
            st.image(img, caption=slice_label, use_column_width=True)
        with col_side:
            st.markdown(
                f"""
**Slice info**

- **Axis:** {axis}
- **Depth:** {depth} / {max_depth}
- **Shape:** {sl.shape[0]} × {sl.shape[1]}
- **Window:** {win_name2}
- **W / L:** {ww2} / {wl2}
- **Zoom:** {zoom:.2f}×
- **Voxel:** {voxel_ml * 1000:.2f} mm³
                """
            )

        # 3D orientation inset — three small thumbnails with crosshairs
        st.markdown("---")
        st.markdown("##### 3D Orientation — Where am I in the volume?")
        st.caption(
            "Red crosshairs mark the current cut position. Red border = the active view."
        )
        _render_3view_inset(vol, axis, depth, wl2, ww2, nz, ny, nx)

        # Annotated HU histogram
        st.markdown("---")
        st.markdown("##### HU Histogram — Tissue Analysis")
        show_tissue_labels = st.checkbox(
            "Show tissue labels",
            value=True,
            key="knife_show_tissue_labels",
            help="Overlay colored HU bands for known tissue types on the histogram.",
        )

        flat = sl.flatten()
        fig_hist = _render_annotated_histogram(flat, show_labels=show_tissue_labels)
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        detected = _detect_tissues_in_slice(flat)
        if detected:
            st.markdown(
                f"**Tissues detected in this slice:** "
                + ", ".join(f"`{t}`" for t in detected)
            )
        else:
            st.markdown("*No standard tissue HU bands detected in this slice.*")

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

    # ── Feature B: Code Lab Tab ────────────────────────────────────────────────
    with tab_code:
        _render_code_lab(vol, nz, ny, nx, student_id)

    # ── Feature C: Challenges Tab ──────────────────────────────────────────────
    with tab_challenge:
        _render_challenges(vol, nz, ny, nx, spacing, voxel_ml, student_id)

    # ── 3D Volume Rendering Tab ─────────────────────────────────────────────────
    with tab_3d:
        st.subheader("🧊 3-D Volume Rendering")
        st.caption("Rotate: drag | Zoom: scroll | Pan: right-click drag")

        # ── Window presets ──────────────────────────────────────────────────────
        st.markdown("**⚡ Quick Window Presets**")
        _pcols = st.columns(5)
        _presets = [
            ("🧠 Brain", 0, 80),
            ("🦴 Bone", 300, 1000),
            ("🫁 Lung", -900, -500),
            ("🩸 Soft Tissue", -100, 150),
            ("👁️ Head/Neck", -200, 200),
        ]
        for _pi, (_plbl, _plo, _phi) in enumerate(_presets):
            with _pcols[_pi]:
                if st.button(_plbl, key=f"preset_{_pi}"):
                    st.session_state["ct3d_lower"] = _plo
                    st.session_state["ct3d_upper"] = _phi

        st.markdown("---")

        # ── Volume availability ─────────────────────────────────────────────────
        _vol3d = st.session_state.get("ct_volume", st.session_state.get("ct_vol"))
        if _vol3d is None:
            st.warning("⬅️ Load a phantom or upload a DICOM in the **📂 Load Case** tab first.")
        else:
            _shape = _vol3d.shape
            _hu_min, _hu_max = int(_vol3d.min()), int(_vol3d.max())
            st.info(f"📊 Volume: {_shape[2]}×{_shape[1]}×{_shape[0]} voxels | HU range: {_hu_min} → {_hu_max}")

            # ── Controls ────────────────────────────────────────────────────────
            _c1, _c2, _c3 = st.columns(3)
            with _c1:
                _lo = st.slider("Lower HU threshold", -1000, 500,
                                st.session_state.get("ct3d_lower", -200), 10, key="ct3d_lower_sl")
            with _c2:
                _hi = st.slider("Upper HU threshold", -500, 2000,
                                st.session_state.get("ct3d_upper", 400), 10, key="ct3d_upper_sl")
            with _c3:
                _ds = st.slider("Downsample (1=full, 4=fast)", 1, 4, 2, key="ct3d_ds")

            _render_mode = st.radio("Render mode", ["🌫️ Volume Cloud", "🗿 Isosurface (solid)"],
                                    horizontal=True, key="ct3d_mode")

            if st.button("🚀 Render 3-D", key="ct3d_render", type="primary"):
                import plotly.graph_objects as go
                import numpy as np

                # Downsample
                _v = _vol3d[::_ds, ::_ds, ::_ds].astype(np.float32)
                # Clamp to threshold
                _v = np.clip(_v, _lo, _hi)
                _nz, _ny, _nx = _v.shape
                _zz, _yy, _xx = np.mgrid[0:_nz, 0:_ny, 0:_nx]

                if "Volume" in _render_mode:
                    _fig = go.Figure(go.Volume(
                        x=_xx.flatten(), y=_yy.flatten(), z=_zz.flatten(),
                        value=_v.flatten(),
                        isomin=_lo, isomax=_hi,
                        opacity=0.15,
                        opacityscale=[[0,0],[0.1,0.01],[0.3,0.05],[0.6,0.2],[1.0,0.6]],
                        surface_count=25,
                        colorscale="Hot",
                        lighting=dict(ambient=0.4, diffuse=0.8, roughness=0.5,
                                      specular=0.5, fresnel=0.2),
                        lightposition=dict(x=100, y=200, z=0),
                    ))
                    _fig.update_layout(
                        scene=dict(
                            bgcolor="black",
                            xaxis=dict(showticklabels=False, showgrid=False),
                            yaxis=dict(showticklabels=False, showgrid=False),
                            zaxis=dict(showticklabels=False, showgrid=False),
                        ),
                        paper_bgcolor="black",
                        font_color="white",
                        margin=dict(l=0, r=0, t=30, b=0),
                        title=dict(text="Volume Cloud Render", font=dict(color="white")),
                        height=600,
                    )
                else:
                    _iso_val = int((_lo + _hi) / 2)
                    _fig = go.Figure(go.Isosurface(
                        x=_xx.flatten(), y=_yy.flatten(), z=_zz.flatten(),
                        value=_v.flatten(),
                        isomin=_lo, isomax=_hi,
                        surface_count=3,
                        colorscale="Plasma",
                        caps=dict(x_show=False, y_show=False, z_show=False),
                        lighting=dict(ambient=0.3, diffuse=0.9, roughness=0.4,
                                      specular=0.8, fresnel=0.1),
                        lightposition=dict(x=100, y=200, z=0),
                    ))
                    _fig.update_layout(
                        scene=dict(
                            bgcolor="black",
                            xaxis=dict(showticklabels=False, showgrid=False),
                            yaxis=dict(showticklabels=False, showgrid=False),
                            zaxis=dict(showticklabels=False, showgrid=False),
                        ),
                        paper_bgcolor="black",
                        font_color="white",
                        margin=dict(l=0, r=0, t=30, b=0),
                        title=dict(text="Isosurface Render", font=dict(color="white")),
                        height=600,
                    )

                st.plotly_chart(_fig, use_container_width=True)

            # ── HU reference table ────────────────────────────────────────────
            with st.expander("📖 HU Tissue Reference"):
                import pandas as pd
                st.dataframe(pd.DataFrame({
                    "Tissue": ["Air","Lung","Fat","Water/CSF","Soft Tissue","Blood","Contrast","Bone Trabecular","Bone Cortical"],
                    "HU Min": [-1000,-900,-120,-10,20,30,100,200,700],
                    "HU Max": [-950,-600,-50,15,80,45,300,400,1900],
                }), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature B — Code Lab
# ═══════════════════════════════════════════════════════════════════════════════

# Number of Code Lab exercises
CODE_LAB_TOTAL = 8


def _render_code_lab(vol, nz, ny, nx, student_id):
    st.subheader("🐍 Code Lab — Python Array Slicing")
    st.markdown(
        "Master CT-volume indexing by solving these exercises. "
        "Each correct answer unlocks a real slice from the loaded volume."
    )

    # Initialise session state
    if "ct_exercises_done" not in st.session_state:
        st.session_state.ct_exercises_done = set()

    done = st.session_state.ct_exercises_done
    n_done = len(done)
    st.progress(n_done / CODE_LAB_TOTAL, text=f"Progress: {n_done}/{CODE_LAB_TOTAL} exercises completed")

    if n_done == CODE_LAB_TOTAL:
        st.balloons()
        st.success("🏆 Code Lab Complete! All 8 exercises solved.")

    st.markdown("---")

    # Exercise selector
    ex_num = st.selectbox(
        "Choose an exercise",
        list(range(1, CODE_LAB_TOTAL + 1)),
        format_func=lambda n: (
            f"Exercise {n} {'✅' if n in done else ''}"
        ),
        key="codelab_ex_select",
    )

    # Precompute slice indices for visual confirmations
    mid_z = nz // 2
    mid_y = ny // 2
    mid_x = nx // 2
    WL_DEFAULT, WW_DEFAULT = 40, 400

    # Helper: show image inline
    def _show_slice(arr2d, caption):
        img = _slice_to_img(arr2d, WL_DEFAULT, WW_DEFAULT)
        st.image(img, caption=caption, width=300)

    # ── Exercise 1 ──────────────────────────────────────────────────────────
    if ex_num == 1:
        st.markdown("### Exercise 1 — Axial Slice")
        st.info(
            "A CT volume is stored as `vol` with shape `(slices, rows, cols)`.  \n"
            "**Write the expression to get axial slice number 30.**"
        )
        ans1 = st.radio(
            "Select the correct expression:",
            ["A) `vol[30]`", "B) `vol[:, 30, :]`", "C) `vol[:, :, 30]`", "D) `vol[30, :, :]`"],
            key="ex1_radio",
        )
        if st.button("✅ Submit Answer", key="ex1_submit"):
            if "A)" in ans1 or "D)" in ans1:
                st.success(
                    "✅ Correct! Both `vol[30]` and `vol[30, :, :]` retrieve the axial "
                    "slice at index 30. In NumPy, a bare integer index selects along the "
                    "first axis — which is the slice (z) axis in a CT volume."
                )
                done.add(1)
                st.session_state.ct_exercises_done = done
                _show_slice(vol[min(30, nz - 1)], "vol[30, :, :] — Axial slice 30")
            else:
                st.error(
                    "❌ Try again. Hint: The first axis (axis 0) is the slice direction. "
                    "`vol[30]` = `vol[30, :, :]` selects all rows and columns at z=30."
                )

    # ── Exercise 2 ──────────────────────────────────────────────────────────
    elif ex_num == 2:
        st.markdown("### Exercise 2 — Coronal Slice")
        st.info("Get the coronal slice at row index `y`.")
        ans2 = st.radio(
            "Select the correct expression:",
            ["A) `vol[y, :, :]`", "B) `vol[:, y, :]`", "C) `vol[:, :, y]`", "D) `vol[y]`"],
            key="ex2_radio",
        )
        if st.button("✅ Submit Answer", key="ex2_submit"):
            if "B)" in ans2:
                st.success(
                    "✅ Correct! `vol[:, y, :]` selects all slices and all columns at "
                    "row y — producing a coronal (front-to-back) cross-section."
                )
                done.add(2)
                st.session_state.ct_exercises_done = done
                _show_slice(vol[:, mid_y, :], f"vol[:, {mid_y}, :] — Coronal slice at midpoint y={mid_y}")
            else:
                st.error(
                    "❌ Try again. Hint: Axis 1 is the row (y / anterior-posterior) direction. "
                    "Fix the second index to y and leave axis 0 and 2 as `:` (all)."
                )

    # ── Exercise 3 ──────────────────────────────────────────────────────────
    elif ex_num == 3:
        st.markdown("### Exercise 3 — Sagittal Slice")
        st.info("Get the sagittal slice at column index `x`.")
        ans3 = st.radio(
            "Select the correct expression:",
            ["A) `vol[:, :, x]`", "B) `vol[x, :, :]`", "C) `vol[:, x, :]`", "D) `vol[x]`"],
            key="ex3_radio",
        )
        if st.button("✅ Submit Answer", key="ex3_submit"):
            if "A)" in ans3:
                st.success(
                    "✅ Correct! `vol[:, :, x]` keeps all slices and all rows, selecting "
                    "column x — this gives you a left-right (sagittal) cross-section."
                )
                done.add(3)
                st.session_state.ct_exercises_done = done
                _show_slice(vol[:, :, mid_x], f"vol[:, :, {mid_x}] — Sagittal slice at midpoint x={mid_x}")
            else:
                st.error(
                    "❌ Try again. Hint: Axis 2 is the column (x / left-right) direction. "
                    "Fix the third index to x and leave axes 0 and 1 as `:` (all)."
                )

    # ── Exercise 4 ──────────────────────────────────────────────────────────
    elif ex_num == 4:
        st.markdown("### Exercise 4 — Step Slicing (Every Other Slice)")
        st.info(
            "Get every other axial slice from a volume (reduces radiation dose by "
            "skipping alternating slices). **Which expression is correct?**"
        )
        ans4 = st.radio(
            "Select the correct expression:",
            ["A) `vol[::2, :, :]`", "B) `vol[2:, :, :]`", "C) `vol[:2, :, :]`", "D) `vol[2::2, :, :]`"],
            key="ex4_radio",
        )
        if st.button("✅ Submit Answer", key="ex4_submit"):
            if "A)" in ans4:
                st.success(
                    "✅ Correct! `vol[::2, :, :]` uses a step of 2 starting from index 0, "
                    "selecting slices 0, 2, 4, 6, … This halves the radiation dose model."
                )
                done.add(4)
                st.session_state.ct_exercises_done = done
                st.markdown(
                    f"**Full stack:** `vol.shape = {vol.shape}`  \n"
                    f"**Every other slice:** `vol[::2].shape = {vol[::2].shape}`"
                )
            else:
                st.error(
                    "❌ Try again. Hint: Use start:stop:step notation — `::2` means "
                    "start at 0, go to end, step by 2."
                )

    # ── Exercise 5 ──────────────────────────────────────────────────────────
    elif ex_num == 5:
        st.markdown("### Exercise 5 — Reverse Axial Order")
        st.info(
            "DICOM slices sometimes come in head-to-toe order. "
            "**Reverse the stack so it runs toe-to-head.**"
        )
        ans5 = st.radio(
            "Select the correct expression:",
            ["A) `vol[-1, :, :]`", "B) `vol[::-1, :, :]`", "C) `vol[0:-1, :, :]`", "D) `vol.flip()`"],
            key="ex5_radio",
        )
        if st.button("✅ Submit Answer", key="ex5_submit"):
            if "B)" in ans5:
                st.success(
                    "✅ Correct! `vol[::-1, :, :]` steps backward through axis 0 "
                    "(the slice axis), reversing the stack order. This is a standard "
                    "DICOM normalisation step before 3D reconstruction."
                )
                done.add(5)
                st.session_state.ct_exercises_done = done
                _show_slice(vol[-1], f"vol[-1] — first slice after reversal (original last slice)")
            else:
                st.error(
                    "❌ Try again. Hint: `[::-1]` reverses a Python list — the same "
                    "trick works on NumPy axes. Apply it only to axis 0."
                )

    # ── Exercise 6 ──────────────────────────────────────────────────────────
    elif ex_num == 6:
        st.markdown("### Exercise 6 — Boolean Masking (Segmentation)")
        st.info(
            "You want a mask of all voxels with HU between 40 and 80 (liver). "
            "**Which expression creates this boolean mask?**"
        )
        ans6 = st.radio(
            "Select the correct expression:",
            [
                "A) `vol[40:80]`",
                "B) `(vol >= 40) & (vol <= 80)`",
                "C) `vol == [40, 80]`",
                "D) `vol.mask(40, 80)`",
            ],
            key="ex6_radio",
        )
        if st.button("✅ Submit Answer", key="ex6_submit"):
            if "B)" in ans6:
                st.success(
                    "✅ Correct! Element-wise comparison `(vol >= 40) & (vol <= 80)` "
                    "produces a boolean array — True where HU is in the liver range. "
                    "Note: use `&` (bitwise AND), not `and` (Python keyword), for NumPy arrays."
                )
                done.add(6)
                st.session_state.ct_exercises_done = done
                # Show liver segmentation overlay
                sl_liver = vol[mid_z]
                liver_mask = (sl_liver >= 40) & (sl_liver <= 80)
                base_img = _slice_to_img(sl_liver, WL_DEFAULT, WW_DEFAULT).convert("RGB")
                overlay = np.array(base_img)
                overlay[liver_mask] = [255, 140, 0]
                st.image(Image.fromarray(overlay), caption="Liver mask (HU 40–80) shown in orange", width=300)
            else:
                st.error(
                    "❌ Try again. Hint: Use comparison operators (`>=`, `<=`) with "
                    "the bitwise `&` operator. `vol[40:80]` would slice the first axis "
                    "from index 40 to 80 — not an HU threshold."
                )

    # ── Exercise 7 ──────────────────────────────────────────────────────────
    elif ex_num == 7:
        st.markdown("### Exercise 7 — Count Masked Voxels")
        st.info(
            "Given a boolean mask `mask`, how do you count the number of True voxels "
            "(e.g., to compute organ volume)?"
        )
        ans7 = st.radio(
            "Select the correct expression:",
            ["A) `len(mask)`", "B) `mask.count()`", "C) `mask.sum()`", "D) `np.count(mask)`"],
            key="ex7_radio",
        )
        if st.button("✅ Submit Answer", key="ex7_submit"):
            if "C)" in ans7:
                st.success(
                    "✅ Correct! `mask.sum()` treats `True` as 1 and `False` as 0, "
                    "so it counts the True voxels. Multiply by voxel volume (mm³) "
                    "and divide by 1000 to get volume in mL."
                )
                done.add(7)
                st.session_state.ct_exercises_done = done
            else:
                st.error(
                    "❌ Try again. Hint: In NumPy, `True == 1` and `False == 0`. "
                    "`len(mask)` gives the length of the first axis, not the count of True values."
                )

    # ── Exercise 8 ──────────────────────────────────────────────────────────
    elif ex_num == 8:
        st.markdown("### Exercise 8 — Volume Calculation")
        st.info(
            "Each voxel is **0.5 mm × 0.5 mm × 1.0 mm**.  \n"
            "You have **4800 liver voxels**.  \n"
            "**What is the liver volume in mL?** (1 mL = 1000 mm³)"
        )
        st.code("volume_mL = n_voxels × voxel_x_mm × voxel_y_mm × voxel_z_mm / 1000", language="python")
        user_answer = st.number_input(
            "Your answer (mL):",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.01,
            key="ex8_number",
        )
        CORRECT_ANSWER_ML = 4800 * 0.5 * 0.5 * 1.0 / 1000  # = 1.2 mL
        TOLERANCE = 0.05  # accept 1.15 to 1.25
        if st.button("✅ Submit Answer", key="ex8_submit"):
            if abs(user_answer - CORRECT_ANSWER_ML) <= TOLERANCE:
                st.success(
                    f"✅ Correct! The liver volume is **{CORRECT_ANSWER_ML:.2f} mL**.  \n"
                    f"Formula: `4800 × 0.5 × 0.5 × 1.0 / 1000 = {CORRECT_ANSWER_ML} mL`  \n"
                    "This is how radiologists estimate organ volume from CT — count the voxels "
                    "and multiply by the voxel size."
                )
                done.add(8)
                st.session_state.ct_exercises_done = done
            else:
                st.error(
                    f"❌ Try again. Hint: voxel volume = 0.5 × 0.5 × 1.0 = 0.25 mm³. "
                    f"Total = 4800 × 0.25 = 1200 mm³ = ? mL."
                )

    # Refresh progress bar after any answer
    n_done_new = len(st.session_state.ct_exercises_done)
    if n_done_new > n_done:
        if n_done_new == CODE_LAB_TOTAL:
            st.balloons()
            st.success("🏆 Code Lab Complete! All 8 exercises solved.")
        else:
            st.info(f"📊 Progress: {n_done_new}/{CODE_LAB_TOTAL} completed.")


# ═══════════════════════════════════════════════════════════════════════════════
# Feature C — Guided Challenge Mode
# ═══════════════════════════════════════════════════════════════════════════════

CHALLENGE_TOTAL = 5


def _challenge_status_icon(cnum):
    """Return emoji for a challenge's unlock status."""
    unlocked = st.session_state.get("ct_challenge_unlocked", {})
    passed = st.session_state.get("ct_challenge_passed", {})
    if passed.get(cnum, False):
        return "✅"
    if unlocked.get(cnum, False) or cnum == 1:
        return "🎯"
    return "🔒"


def _init_challenge_state():
    """Ensure all challenge session-state keys exist."""
    if "ct_challenge_unlocked" not in st.session_state:
        st.session_state.ct_challenge_unlocked = {1: True}
    if "ct_challenge_passed" not in st.session_state:
        st.session_state.ct_challenge_passed = {}
    if "ct_challenge_scores" not in st.session_state:
        st.session_state.ct_challenge_scores = {}
    if "ct_challenge_tries" not in st.session_state:
        st.session_state.ct_challenge_tries = {}


def _render_challenges(vol, nz, ny, nx, spacing, voxel_ml, student_id):
    st.subheader("🎯 Guided Challenge Mode")

    _init_challenge_state()

    unlocked = st.session_state.ct_challenge_unlocked
    passed = st.session_state.ct_challenge_passed
    scores = st.session_state.ct_challenge_scores
    tries = st.session_state.ct_challenge_tries

    # Total score header
    total_score = sum(scores.values())
    st.metric("Challenge Score", f"{total_score} / 500")
    st.progress(total_score / 500.0, text=f"{total_score}/500 points")

    st.markdown(
        "Complete challenges in order. Pass one to unlock the next. "
        "Your score is saved when all 5 are complete."
    )
    st.markdown("---")

    # ── Challenge 1: Find the Organ ─────────────────────────────────────────
    c1_unlocked = unlocked.get(1, True)
    c1_passed = passed.get(1, False)
    c1_icon = _challenge_status_icon(1)

    with st.expander(f"{c1_icon} Challenge 1 — Find the Organ (MPR Navigation)", expanded=(c1_unlocked and not c1_passed)):
        if not c1_unlocked:
            st.warning("🔒 Complete Challenge 0 first (no predecessor — always unlocked).")
        else:
            st.markdown(
                "**Task:** Using the Axial slider below, find the slice where the **liver "
                "appears LARGEST** (most orange/highlighted area). Note that liver voxels "
                "are in the HU range **40–80** (shown in orange).  \n\n"
                "When you think you've found it, hit Submit."
            )

            # Build abdomen volume for this challenge (use loaded vol or generate abdomen)
            abdomen_vol = _make_abdomen()
            c1_nz = abdomen_vol.shape[0]

            # Precompute liver pixel count per slice
            liver_counts = np.array([
                int(np.sum((abdomen_vol[z] >= 40) & (abdomen_vol[z] <= 80)))
                for z in range(c1_nz)
            ])
            true_max_slice = int(np.argmax(liver_counts))
            true_max_count = int(liver_counts[true_max_slice])

            c1_z = st.slider(
                "Axial slice (Abdomen case)", 0, c1_nz - 1, c1_nz // 2,
                key="c1_slider"
            )

            # Live liver overlay
            sl_c1 = abdomen_vol[c1_z]
            liver_mask_c1 = (sl_c1 >= 40) & (sl_c1 <= 80)
            base_c1 = _slice_to_img(sl_c1, 60, 400).convert("RGB")
            ov_c1 = np.array(base_c1)
            ov_c1[liver_mask_c1] = [255, 140, 0]
            st.image(Image.fromarray(ov_c1), caption=f"Axial z={c1_z}", width=300)

            live_count = int(liver_counts[c1_z])
            st.metric("Liver pixels in this slice", live_count)

            if not c1_passed:
                if st.button("📤 Submit: this is the maximum-liver slice", key="c1_submit"):
                    diff = abs(c1_z - true_max_slice)
                    if diff <= 2:
                        sc = 100
                    elif diff <= 5:
                        sc = 80
                    else:
                        sc = 60
                    scores[1] = sc
                    passed[1] = True
                    unlocked[2] = True
                    st.session_state.ct_challenge_scores = scores
                    st.session_state.ct_challenge_passed = passed
                    st.session_state.ct_challenge_unlocked = unlocked
                    st.success(
                        f"✅ Challenge 1 complete! Score: **{sc}/100**  \n"
                        f"Your slice: {c1_z} | True max-liver slice: {true_max_slice} "
                        f"(liver pixels: {true_max_count})"
                    )
                    # Show the true max slice for reference
                    sl_ref = abdomen_vol[true_max_slice]
                    lm_ref = (sl_ref >= 40) & (sl_ref <= 80)
                    base_ref = _slice_to_img(sl_ref, 60, 400).convert("RGB")
                    ov_ref = np.array(base_ref)
                    ov_ref[lm_ref] = [255, 140, 0]
                    st.image(Image.fromarray(ov_ref),
                             caption=f"True max-liver slice z={true_max_slice}", width=300)
                    st.info("👉 Challenge 2 is now unlocked!")
            else:
                st.success(f"✅ Already passed! Score: {scores.get(1, 0)}/100")

    # ── Challenge 2: Window the Brain ───────────────────────────────────────
    c2_unlocked = unlocked.get(2, False)
    c2_passed = passed.get(2, False)
    c2_icon = _challenge_status_icon(2)

    with st.expander(f"{c2_icon} Challenge 2 — Window the Brain", expanded=(c2_unlocked and not c2_passed)):
        if not c2_unlocked:
            st.warning("🔒 Complete Challenge 1 first.")
        else:
            st.markdown(
                "**Task:** A head CT has been loaded. Apply the correct Window/Level "
                "preset to see **grey matter vs. white matter** clearly. Which preset do you choose?"
            )
            brain_vol = _make_brain()
            b_nz = brain_vol.shape[0]
            b_mid = b_nz // 2

            # Show current view with user-chosen window
            c2_preset = st.radio(
                "Choose window preset:",
                ["Brain (W:80, L:40)", "Lung (W:1500, L:-600)",
                 "Bone (W:2000, L:400)", "Soft Tissue (W:400, L:40)"],
                key="c2_preset",
            )
            preset_map = {
                "Brain (W:80, L:40)":       (40,  80),
                "Lung (W:1500, L:-600)":    (-600, 1500),
                "Bone (W:2000, L:400)":     (400, 2000),
                "Soft Tissue (W:400, L:40)":(40, 400),
            }
            wl_c2, ww_c2 = preset_map[c2_preset]
            img_c2 = _slice_to_img(brain_vol[b_mid], wl_c2, ww_c2)
            st.image(img_c2, caption=f"Brain CT with {c2_preset}", width=300)

            if not c2_passed:
                tries_c2 = tries.get(2, 0)
                if st.button("📤 Submit window choice", key="c2_submit"):
                    tries[2] = tries_c2 + 1
                    st.session_state.ct_challenge_tries = tries
                    if "Brain" in c2_preset:
                        sc2 = 100 if tries_c2 == 0 else 50
                        scores[2] = sc2
                        passed[2] = True
                        unlocked[3] = True
                        st.session_state.ct_challenge_scores = scores
                        st.session_state.ct_challenge_passed = passed
                        st.session_state.ct_challenge_unlocked = unlocked
                        st.success(
                            f"✅ Correct! Score: **{sc2}/100**  \n"
                            "The **Brain preset** (W:80, L:40) gives narrow contrast "
                            "centered on grey matter HU (~40 HU), separating grey (30–40 HU) "
                            "from white matter (20–30 HU). Wide windows wash out this contrast."
                        )
                        st.info("👉 Challenge 3 is now unlocked!")
                    else:
                        st.error(
                            f"❌ Incorrect (attempt {tries_c2 + 1}). Hint: Grey and white matter "
                            "differ by only ~10 HU. Which preset has the narrowest window?"
                        )
            else:
                st.success(f"✅ Already passed! Score: {scores.get(2, 0)}/100")

    # ── Challenge 3: Segment the Lung ───────────────────────────────────────
    c3_unlocked = unlocked.get(3, False)
    c3_passed = passed.get(3, False)
    c3_icon = _challenge_status_icon(3)

    with st.expander(f"{c3_icon} Challenge 3 — Segment the Lung", expanded=(c3_unlocked and not c3_passed)):
        if not c3_unlocked:
            st.warning("🔒 Complete Challenge 2 first.")
        else:
            st.markdown(
                "**Task:** Using HU thresholding, segment the lungs from the Chest case. "
                "The lungs appear at HU range **[-700, -500]**. "
                "Adjust the HU sliders and submit your segmented volume."
            )
            chest_vol = _make_chest()
            c3_nz = chest_vol.shape[0]
            c3_mid = c3_nz // 2

            # Reference lung volume
            REF_LO, REF_HI = -700, -500
            ref_mask = (chest_vol >= REF_LO) & (chest_vol <= REF_HI)
            ref_voxel_ml = (0.5 * 0.5 * 0.5) / 1000.0
            ref_vol_ml = float(ref_mask.sum()) * ref_voxel_ml

            lo_c3 = st.slider("Low HU threshold", -1100, 0, -750, 10, key="c3_lo")
            hi_c3 = st.slider("High HU threshold", -1000, 200, -400, 10, key="c3_hi")

            sl_c3 = chest_vol[c3_mid]
            seg_mask_c3 = (sl_c3 >= lo_c3) & (sl_c3 <= hi_c3)
            base_c3 = _slice_to_img(sl_c3, -600, 1500).convert("RGB")
            ov_c3 = np.array(base_c3)
            ov_c3[seg_mask_c3] = [0, 180, 255]
            st.image(Image.fromarray(ov_c3),
                     caption=f"Lung segmentation overlay (HU {lo_c3} to {hi_c3})", width=300)

            # Compute full-volume segmentation
            full_mask_c3 = (chest_vol >= lo_c3) & (chest_vol <= hi_c3)
            student_vol_ml = float(full_mask_c3.sum()) * ref_voxel_ml
            st.metric("Your segmented lung volume", f"{student_vol_ml:.2f} mL")

            if not c3_passed:
                if st.button("📤 Submit segmentation", key="c3_submit"):
                    if ref_vol_ml > 0:
                        pct_err = abs(student_vol_ml - ref_vol_ml) / ref_vol_ml
                    else:
                        pct_err = 1.0
                    if pct_err <= 0.10:
                        sc3 = 100
                    elif pct_err <= 0.20:
                        sc3 = 75
                    elif pct_err <= 0.30:
                        sc3 = 50
                    else:
                        sc3 = 0

                    scores[3] = sc3
                    if sc3 > 0:
                        passed[3] = True
                        unlocked[4] = True
                    st.session_state.ct_challenge_scores = scores
                    st.session_state.ct_challenge_passed = passed
                    st.session_state.ct_challenge_unlocked = unlocked

                    if sc3 > 0:
                        st.success(
                            f"✅ Challenge 3 complete! Score: **{sc3}/100**  \n"
                            f"Your lung volume: {student_vol_ml:.2f} mL | "
                            f"Reference (HU -700 to -500): {ref_vol_ml:.2f} mL  \n"
                            f"Error: {pct_err*100:.1f}%"
                        )
                        st.info("👉 Challenge 4 is now unlocked!")
                    else:
                        st.error(
                            f"❌ Volume too far off ({pct_err*100:.1f}% error). "
                            f"Reference volume: {ref_vol_ml:.2f} mL. Try adjusting HU thresholds closer to [-700, -500]."
                        )
            else:
                st.success(f"✅ Already passed! Score: {scores.get(3, 0)}/100")

    # ── Challenge 4: Identify the Tissue ────────────────────────────────────
    c4_unlocked = unlocked.get(4, False)
    c4_passed = passed.get(4, False)
    c4_icon = _challenge_status_icon(4)

    with st.expander(f"{c4_icon} Challenge 4 — Identify the Tissue", expanded=(c4_unlocked and not c4_passed)):
        if not c4_unlocked:
            st.warning("🔒 Complete Challenge 3 first.")
        else:
            st.markdown(
                "**Task:** Look at the HU histogram of this slice. "
                "**Which tissues are present?** Select all that apply."
            )
            # Use a fixed abdomen slice for reproducibility
            abd_vol = _make_abdomen()
            c4_z = abd_vol.shape[0] // 2
            sl_c4 = abd_vol[c4_z]

            # Show annotated histogram
            fig_c4 = _render_annotated_histogram(sl_c4.flatten(), show_labels=True)
            st.pyplot(fig_c4)
            plt.close(fig_c4)
            st.image(
                _slice_to_img(sl_c4, 60, 400),
                caption=f"Axial slice z={c4_z}", width=300
            )

            # Ground truth tissues
            ground_truth = set(_detect_tissues_in_slice(sl_c4.flatten()))
            all_tissue_names = [t[0] for t in TISSUE_BANDS]

            c4_selected = st.multiselect(
                "Select all tissues present in this slice:",
                all_tissue_names,
                key="c4_multiselect",
            )

            if not c4_passed:
                if st.button("📤 Submit tissue identification", key="c4_submit"):
                    selected_set = set(c4_selected)
                    wrong = len(selected_set.symmetric_difference(ground_truth))
                    sc4 = max(0, 100 - 10 * wrong)
                    scores[4] = sc4
                    if sc4 >= 60:
                        passed[4] = True
                        unlocked[5] = True
                    st.session_state.ct_challenge_scores = scores
                    st.session_state.ct_challenge_passed = passed
                    st.session_state.ct_challenge_unlocked = unlocked

                    if sc4 >= 60:
                        st.success(
                            f"✅ Challenge 4 complete! Score: **{sc4}/100**  \n"
                            f"Ground truth: {', '.join(sorted(ground_truth))}  \n"
                            f"Your answer: {', '.join(sorted(selected_set))}  \n"
                            f"Incorrect selections/misses: {wrong}"
                        )
                        st.info("👉 Challenge 5 is now unlocked!")
                    else:
                        st.error(
                            f"❌ Score: {sc4}/100 — too many wrong answers ({wrong}). "
                            f"Ground truth: {', '.join(sorted(ground_truth))}. Try again after reviewing the histogram."
                        )
            else:
                st.success(f"✅ Already passed! Score: {scores.get(4, 0)}/100")

    # ── Challenge 5: Write the Slice Expression ──────────────────────────────
    c5_unlocked = unlocked.get(5, False)
    c5_passed = passed.get(5, False)
    c5_icon = _challenge_status_icon(5)

    with st.expander(f"{c5_icon} Challenge 5 — Write the Slice Expression", expanded=(c5_unlocked and not c5_passed)):
        if not c5_unlocked:
            st.warning("🔒 Complete Challenge 4 first.")
        else:
            st.markdown(
                "**Task:** You need to extract every **3rd coronal slice** from the volume "
                "to create a faster preview. Write the correct numpy expression."
            )
            ans_c5 = st.radio(
                "Select the correct expression:",
                [
                    "A) `vol[:, ::3, :]`",
                    "B) `vol[::3, :, :]`",
                    "C) `vol[:, :, ::3]`",
                    "D) `vol[::3]`",
                ],
                key="c5_radio",
            )
            if not c5_passed:
                tries_c5 = tries.get(5, 0)
                if st.button("📤 Submit expression", key="c5_submit"):
                    tries[5] = tries_c5 + 1
                    st.session_state.ct_challenge_tries = tries
                    if "A)" in ans_c5:
                        sc5 = 100 if tries_c5 == 0 else 50
                        scores[5] = sc5
                        passed[5] = True
                        st.session_state.ct_challenge_scores = scores
                        st.session_state.ct_challenge_passed = passed
                        st.success(
                            f"✅ Challenge 5 complete! Score: **{sc5}/100**  \n"
                            "`vol[:, ::3, :]` — axis 1 is the row (y / coronal) direction. "
                            "`::3` selects every 3rd index. Axes 0 and 2 are left intact."
                        )
                    else:
                        st.error(
                            f"❌ Incorrect (attempt {tries_c5 + 1}). Hint: Coronal slices vary "
                            "along axis 1 (rows). Use `::3` on axis 1 only."
                        )
            else:
                st.success(f"✅ Already passed! Score: {scores.get(5, 0)}/100")

    # ── Final score + save ───────────────────────────────────────────────────
    total_now = sum(st.session_state.ct_challenge_scores.values())
    if len(st.session_state.ct_challenge_passed) == CHALLENGE_TOTAL:
        st.markdown("---")
        st.balloons()
        st.success(f"🏆 All challenges complete! Final score: **{total_now} / 500**")
        if st.button("💾 Save challenge results", key="challenge_save"):
            record_session(
                student_id, "ct", total_now / 5.0,
                f"Challenges complete: {total_now}/500"
            )
            st.success("Results saved!")
