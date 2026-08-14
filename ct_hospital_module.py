"""CT Reading Room — slice view, multi-planar reconstruction, virtual knife, manual segmentation."""

import streamlit as st
import numpy as np
import pandas as pd
import math, io, zipfile, datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from shared_session import dept_header, record_session

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    DICOM_OK = True
except ImportError:
    DICOM_OK = False

AUTOSEG_OK = False
try:
    import nibabel as nib
except ImportError:
    pass

try:
    import SimpleITK as sitk
    from lungmask import LMInferer
    LUNGMASK_OK = True
except ImportError:
    LUNGMASK_OK = False

try:
    from totalsegmentator.python_api import totalsegmentator
    TOTALSEG_OK = True
except ImportError:
    TOTALSEG_OK = False

TOTALSEG_STRUCTURES = [
    "brain", "liver", "spleen", "pancreas", "kidney_left", "kidney_right",
    "stomach", "gallbladder", "urinary_bladder", "prostate", "uterus",
    "colon", "small_bowel", "duodenum", "esophagus", "trachea",
    "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right",
    "heart", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
    "adrenal_gland_left", "adrenal_gland_right", "thyroid_gland",
    "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
    "hip_left", "hip_right", "femur_left", "femur_right", "sacrum",
    "skull", "spinal_cord", "rib_left_1", "rib_right_1",
]

from scipy.ndimage import gaussian_filter as _gf

def _perlin_like_noise(shape, scales=(4, 8, 16, 32), weights=(0.45, 0.30, 0.18, 0.07), seed=42):
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, dtype=np.float32)
    for sc, w in zip(scales, weights):
        n = rng.standard_normal(shape).astype(np.float32)
        sigma = max(shape[0] // sc, 1)
        n = _gf(n, sigma=sigma)
        std = n.std()
        if std > 0: n /= std
        out += w * n
    return out

def _make_brain():
    nz, ny, nx = 100, 144, 144
    cz, cy, cx = 50, 72, 72
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    noise = _perlin_like_noise((nz, ny, nx), scales=(4, 8, 16, 32), weights=(0.45, 0.30, 0.18, 0.07), seed=7)
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
    sulci = (0.6 * np.cos(8 * ang + 0.5 * np.sin(3 * zz / nz * np.pi)) + 0.4 * np.cos(5 * ang + 1.7) + 0.3 * np.cos(11 * ang - 0.7 * zz / nz * np.pi))
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
    v += rng.normal(0, 5, v.shape).astype(np.float32)
    return v

def _make_chest():
    nz, ny, nx = 110, 144, 144
    cz, cy, cx = 55, 72, 72
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    noise = _perlin_like_noise((nz, ny, nx), scales=(4, 8, 16), weights=(0.5, 0.32, 0.18), seed=13)
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
    nz, ny, nx = 110, 144, 144
    cz, cy, cx = 55, 72, 72
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    noise = _perlin_like_noise((nz, ny, nx), scales=(4, 8, 16), weights=(0.5, 0.32, 0.18), seed=21)
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
    nz, ny, nx = 80, 128, 128
    cz, cy, cx = 40, 64, 64
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    noise = _perlin_like_noise((nz, ny, nx), scales=(4, 8, 16), weights=(0.5, 0.32, 0.18), seed=55)
    v = np.full((nz, ny, nx), -1000.0, dtype=np.float32)
    r2d = np.sqrt(((yy-cy)/0.72)**2 + (xx-cx)**2) - 1.8 * noise[nz//2, :, :]
    r3d = np.broadcast_to(r2d[np.newaxis], (nz, ny, nx)).copy()
    z_frac = (zz - cz) / cz
    r3d += 1.5 * z_frac**2
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
    nz, ny, nx = 80, 128, 128
    cz, cy, cx = 40, 64, 64
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    noise = _perlin_like_noise((nz, ny, nx), scales=(4, 8, 16), weights=(0.5, 0.32, 0.18), seed=88)
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
    nz, ny, nx = 90, 144, 144
    cz, cy, cx = 45, 72, 72
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    noise = _perlin_like_noise((nz, ny, nx), scales=(4, 8, 16), weights=(0.5, 0.32, 0.18), seed=11)
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

TISSUE_BANDS = [
    ("Air",           -1100, -900,  -1000, "lightblue"),
    ("Lung",           -700, -500,   -600, "skyblue"),
    ("Fat",            -120,  -60,    -90, "lightyellow"),
    ("Water",            -5,    5,      0, "cyan"),
    ("Soft Tissue",      20,   80,     50, "lightgreen"),
    ("Liver",            40,   80,     60, "green"),
    ("Cortical Bone",   400, 2000,    700, "lightcoral"),
]

@st.cache_resource(show_spinner=False)
def _get_lungmask_inferer():
    return LMInferer()  # loads the pretrained U-net(R231) model, cached once per session

def _run_lungmask(vol, spacing):
    """Run the pretrained lungmask U-Net on a HU volume.
    vol: numpy array (nz, ny, nx) in Hounsfield Units.
    spacing: (dz, dy, dx) in mm, matching this app's existing spacing convention.
    Returns an integer mask array of the same shape: 0=background, 1=right lung, 2=left lung.
    """
    img = sitk.GetImageFromArray(vol.astype(np.int16))
    img.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))  # SimpleITK wants (x, y, z)
    inferer = _get_lungmask_inferer()
    return inferer.apply(img)

def _run_totalsegmentator(vol, spacing, roi_subset, fast=True):
    """Run the pretrained TotalSegmentator model on a HU volume.
    vol: numpy array (nz, ny, nx) in Hounsfield Units.
    spacing: (dz, dy, dx) in mm.
    roi_subset: list of structure names to segment (keeps inference fast/light).
    Returns a dict {structure_name: boolean mask array (nz, ny, nx)}.
    """
    import tempfile, os
    import nibabel as _nib
    affine = np.diag([spacing[2], spacing[1], spacing[0], 1]).astype(np.float64)
    nifti_data = np.transpose(vol, (2, 1, 0)).astype(np.int16)  # nibabel wants (x, y, z)
    img_nifti = _nib.Nifti1Image(nifti_data, affine)
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.nii.gz")
        out_dir = os.path.join(tmpdir, "output")
        img_nifti.to_filename(in_path)
        totalsegmentator(in_path, out_dir, fast=fast, roi_subset=roi_subset, quiet=True)
        masks = {}
        for name in roi_subset:
            mpath = os.path.join(out_dir, f"{name}.nii.gz")
            if os.path.exists(mpath):
                m = _nib.load(mpath).get_fdata()
                masks[name] = np.transpose(m, (2, 1, 0)) > 0  # back to (nz, ny, nx)
    return masks

def _slice_to_img(arr2d, wl, ww):
    lo, hi = wl - ww / 2, wl + ww / 2
    clipped = np.clip(arr2d, lo, hi)
    scaled = ((clipped - lo) / (ww + 1e-6) * 255).astype(np.uint8)
    return Image.fromarray(scaled)

def _scale_up_img(img):
    min_dim = min(img.width, img.height)
    if min_dim < 512:
        scale = 512.0 / min_dim
        # Cap the resulting longer side so a razor-thin slice (e.g. from a
        # single-image DICOM upload) can't get blown up into a gigantic
        # image that crashes PIL/Streamlit when it tries to encode it.
        max_dim = max(img.width, img.height)
        if max_dim > 0:
            scale = min(scale, 2048.0 / max_dim)
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        return img.resize((new_w, new_h), Image.LANCZOS)
    return img

def _apply_z_stretch(img, factor):
    """Stretch a coronal/sagittal image vertically so it reflects real-world
    proportions instead of raw voxel counts. Without this, a scan with few
    axial slices but a large in-plane resolution renders as a flat, squished
    strip even though it's not a bug -- it's just missing the physical
    slice-spacing correction."""
    factor = max(0.05, min(factor, 50.0))  # sane bounds against bad spacing metadata
    if abs(factor - 1.0) < 1e-3:
        return img
    new_h = max(1, int(round(img.height * factor)))
    return img.resize((img.width, new_h), Image.LANCZOS)

def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    while arr.ndim > 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[0]
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
        
    # Ensure each dimension is at least 2 so Streamlit sliders don't crash
    if arr.shape[0] < 2:
        arr = np.concatenate([arr, arr], axis=0)
    if arr.shape[1] < 2:
        arr = np.concatenate([arr, arr], axis=1)
    if arr.shape[2] < 2:
        arr = np.concatenate([arr, arr], axis=2)
        
    return arr

def _extract_spacing(ds):
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

def _load_dicom_zip(raw_bytes, max_voxels=100_000_000):
    if not DICOM_OK:
        st.error("pydicom not installed. Run: pip install pydicom")
        return None, None, None
    loaded = []
    meta_ds = None  # only keep ONE dataset around, for spacing/modality -- not all of them
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
                arr = arr[np.newaxis, :, :]  # (H, W) -> (1, H, W); the full volume gets the ensure_3d padding once, after concatenation
                sort_key = (
                    getattr(ds, "InstanceNumber", None)
                    or getattr(ds, "SliceLocation", None)
                    or name
                )
                if meta_ds is None:
                    meta_ds = ds
                # Don't keep the full Dataset per slice -- it internally caches its own
                # decoded pixel buffer, so holding hundreds of them alongside our own
                # arr copies roughly doubles memory use on real multi-slice series and
                # can OOM-crash a free-tier deployment.
                loaded.append((sort_key, arr))
            except Exception:
                pass
    if not loaded:
        st.error("No valid DICOM slices found in ZIP.")
        return None, None, None

    def sortfunc(item):
        k = item[0]
        try:
            return (0, float(k))
        except (TypeError, ValueError):
            return (1, str(k))

    loaded.sort(key=sortfunc)
    vol = np.concatenate([item[1] for item in loaded], axis=0).astype(np.float32)
    loaded = None  # free the per-slice list promptly rather than waiting for GC
    vol = _ensure_3d(vol)

    if vol.size > max_voxels:
        orig_voxels = vol.size
        step = int(np.ceil((orig_voxels / max_voxels) ** (1 / 3)))
        vol = np.ascontiguousarray(vol[::step, ::step, ::step])
        st.info(f"ℹ️ This series is large ({orig_voxels:,} voxels) -- automatically downsampled by {step}x in each direction to fit comfortably in memory. Coronal/Sagittal/3D will be slightly lower-resolution as a result.")

    spacing = _extract_spacing(meta_ds)
    modality = str(getattr(meta_ds, "Modality", "") or "").upper()
    return vol, spacing, modality

def _render_annotated_histogram(sl_flat, show_labels=True):
    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    fig.set_dpi(150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    counts, bin_edges, patches = ax.hist(sl_flat, bins=80, color="#3D5A80", alpha=0.75, edgecolor="none")
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
            x_min_data = sl_flat.min()
            x_max_data = sl_flat.max()
            if center >= x_min_data - 200 and center <= x_max_data + 200:
                ax.text(center, y_max * 0.88, name, ha="center", va="top", fontsize=6.5, color="#333333", rotation=90, clip_on=True)
    return fig

def _detect_tissues_in_slice(sl_flat):
    detected = []
    for name, lo, hi, center, color in TISSUE_BANDS:
        if np.any((sl_flat >= lo) & (sl_flat <= hi)):
            detected.append(name)
    return detected

def _render_3view_inset(vol, axis, depth, wl, ww, nz, ny, nx):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    fig.patch.set_facecolor("white")
    cur_z = depth if axis == "Axial" else nz // 2
    cur_y = depth if axis == "Coronal" else ny // 2
    cur_x = depth if axis == "Sagittal" else nx // 2
    views = [
        ("Axial",    vol[cur_z],        f"Axial z={cur_z}"),
        ("Coronal",  vol[:, cur_y, :],  f"Coronal y={cur_y}"),
        ("Sagittal", vol[:, :, cur_x],  f"Sagittal x={cur_x}"),
    ]
    for ax, view_axis, sl_data, title in zip(axes, [v[0] for v in views], [v[1] for v in views], [v[2] for v in views]):
        is_active = (view_axis == axis)
        ax.imshow(sl_data, cmap="gray", vmin=wl - ww / 2, vmax=wl + ww / 2, aspect="equal")
        if is_active:
            if view_axis == "Axial":
                ax.axhline(cur_y, color="red", linewidth=0.9, alpha=0.85)
                ax.axvline(cur_x, color="red", linewidth=0.9, alpha=0.85)
            elif view_axis == "Coronal":
                ax.axhline(cur_z, color="red", linewidth=0.9, alpha=0.85)
                ax.axvline(cur_x, color="red", linewidth=0.9, alpha=0.85)
            else:
                ax.axhline(cur_z, color="red", linewidth=0.9, alpha=0.85)
                ax.axvline(cur_y, color="red", linewidth=0.9, alpha=0.85)
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2.0)
        ax.set_title(title, fontsize=9, fontweight="bold" if is_active else "normal", color="red" if is_active else "black")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def render(student_id: str):
    dept_header("CT Reading Room", "🩻", student_id)

    tab_load, tab_mpr, tab_knife, tab_seg, tab_code, tab_challenge, tab_3d = st.tabs(
        ["📂 Load Case", "🖼️ Multi-Planar", "🔪 Virtual Knife", "🎨 Segmentation", "🐍 Code Lab", "🎯 Challenges", "🧊 3-D Volume"]
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
                    st.session_state["ct_volume"] = st.session_state.ct_vol
                    st.session_state.ct_label = label
                    st.session_state.ct_spacing = (0.5, 0.5, 0.5)
                    st.session_state["ct_modality"] = "CT"
                st.success(f"✅ {label} loaded — {st.session_state.ct_vol.shape[0]} slices")

        with col2:
            st.markdown("**Upload real DICOM (.dcm or .zip)**")
            up = st.file_uploader("Upload DICOM ZIP", type=["zip", "dcm"])
            if up and st.button("📤 Load DICOM", key="load_dicom"):
                raw = up.getvalue()
                modality = None
                if up.name.lower().endswith(".zip"):
                    vol, spacing, modality = _load_dicom_zip(raw)
                else:
                    if not DICOM_OK:
                        st.error("pydicom not installed")
                        vol, spacing = None, None
                    else:
                        ds = None
                        try:
                            ds = pydicom.dcmread(io.BytesIO(raw))
                        except Exception as _e:
                            st.error(f"⚠️ Could not read DICOM file: {_e}.")
                            vol, spacing = None, None
                        if ds is not None:
                            try:
                                modality = str(getattr(ds, "Modality", "") or "").upper()
                                arr = ds.pixel_array.astype(np.float32)
                                arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
                                vol = _ensure_3d(arr)
                                spacing = _extract_spacing(ds)
                            except Exception as _e:
                                st.error(f"⚠️ Could not read pixel data from DICOM: {_e}")
                                vol, spacing = None, None
                if vol is not None:
                    vol = _ensure_3d(vol)
                    st.session_state.ct_vol = vol
                    st.session_state["ct_volume"] = vol
                    st.session_state.ct_label = up.name
                    st.session_state.ct_spacing = spacing
                    st.session_state["ct_modality"] = modality
                    r, c, s = spacing
                    if modality and modality != "CT":
                        st.warning(
                            f"⚠️ This file's DICOM Modality tag says **{modality}**, not CT. "
                            "This module assumes pixel values are Hounsfield Units, which only CT scans have. "
                            "For MR (and other non-CT modalities), the HU windows, presets, and 3D thresholds below "
                            "won't correspond to real tissue density -- the render will look strange because it's "
                            "reading the wrong kind of data, not because anything crashed."
                        )
                    st.success(f"✅ DICOM loaded — {vol.shape[0]} slices, {vol.shape[1]}×{vol.shape[2]}, spacing {r:.2f}×{c:.2f}×{s:.2f} mm")

        if "ct_vol" not in st.session_state:
            st.info("Load a case above to unlock the other tabs.")
            return

    vol = st.session_state.ct_vol
    vol = _ensure_3d(vol)
    st.session_state.ct_vol = vol
    st.session_state["ct_volume"] = vol
    spacing = st.session_state.get("ct_spacing", (0.5, 0.5, 0.5))
    _modality = st.session_state.get("ct_modality")
    if _modality and _modality != "CT":
        st.warning(f"⚠️ Loaded data is **{_modality}**, not CT -- HU-based windowing/thresholds below won't be meaningful for this modality.")
    nz, ny, nx = vol.shape
    voxel_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0

    with tab_mpr:
        st.subheader("Multi-Planar Reconstruction")
        st.caption(f"Volume spacing: {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} mm  (1 voxel = {voxel_ml * 1000:.3f} mm³ = {voxel_ml:.4f} mL)")
        if nz <= 2:
            st.info("ℹ️ This looks like a single 2D image rather than a multi-slice CT series, so there's only one real Axial slice to view. Coronal/Sagittal reconstructions and slice scrolling need a full DICOM series (multiple .dcm files in a .zip) to be meaningful.")
        win_name = st.selectbox("Window preset", list(WINDOWS.keys()), key="mpr_win")
        wl, ww = WINDOWS[win_name]
        c1, c2, c3 = st.columns(3)
        
        # Bulletproof sliders: max(1, dim-1) ensures max_value is always at least 1
        with c1:
            max_z = max(1, nz - 1)
            z = st.slider("Axial slice", 0, max_z, min(max_z, nz // 2), key="ax")
            img = _scale_up_img(_slice_to_img(vol[z], wl, ww))
            st.image(img, caption=f"Axial z={z}", use_container_width=True)
        with c2:
            max_y = max(1, ny - 1)
            y = st.slider("Coronal slice", 0, max_y, min(max_y, ny // 2), key="cor")
            img = _slice_to_img(vol[:, y, :], wl, ww)
            img = _apply_z_stretch(img, spacing[2] / spacing[0]) if spacing[0] else img
            img = _scale_up_img(img)
            st.image(img, caption=f"Coronal y={y}", use_container_width=True)
        with c3:
            max_x = max(1, nx - 1)
            x = st.slider("Sagittal slice", 0, max_x, min(max_x, nx // 2), key="sag")
            img = _slice_to_img(vol[:, :, x], wl, ww)
            img = _apply_z_stretch(img, spacing[2] / spacing[1]) if spacing[1] else img
            img = _scale_up_img(img)
            st.image(img, caption=f"Sagittal x={x}", use_container_width=True)
            
        if st.button("💾 Save CT session", key="ct_save_mpr"):
            record_session(student_id, "ct", 75.0, f"MPR:{st.session_state.ct_label}")
            st.success("Saved!")

    try:
        from scipy.ndimage import zoom as scipy_zoom
        SCIPY_OK = True
    except ImportError:
        SCIPY_OK = False

    with tab_knife:
        st.subheader("🔪 Virtual Knife — Interactive CT Slicer")
        with st.expander("ℹ️ How to use the knife", expanded=False):
            st.markdown("**The knife is a plane that cuts through the 3D CT volume.**")
            
        col_axis, col_win = st.columns([1, 2])
        with col_axis:
            axis = st.radio("Cut axis", ["Axial", "Coronal", "Sagittal"], horizontal=True, key="knife_axis")
        with col_win:
            win_name2 = st.selectbox("Window preset", list(WINDOWS.keys()), key="knife_win")
        wl2, ww2 = WINDOWS[win_name2]

        # Bulletproof depth calculation
        if axis == "Axial":
            max_depth = max(1, nz - 1)
            default_depth = min(max_depth, nz // 2)
        elif axis == "Coronal":
            max_depth = max(1, ny - 1)
            default_depth = min(max_depth, ny // 2)
        else:
            max_depth = max(1, nx - 1)
            default_depth = min(max_depth, nx // 2)

        depth_key = f"knife_depth_{axis.lower()}"
        if depth_key not in st.session_state:
            st.session_state[depth_key] = default_depth
        st.session_state[depth_key] = max(0, min(max_depth, st.session_state[depth_key]))

        st.markdown("##### Cut depth")
        col_prev, col_slider, col_next, col_reset = st.columns([1, 6, 1, 1])
        with col_prev:
            if st.button("◀", key="knife_prev", help="Previous slice"):
                st.session_state[depth_key] = max(0, st.session_state[depth_key] - 1)
        with col_next:
            if st.button("▶", key="knife_next", help="Next slice"):
                st.session_state[depth_key] = min(max_depth, st.session_state[depth_key] + 1)
        with col_reset:
            if st.button("⟲", key="knife_reset", help="Jump to middle"):
                st.session_state[depth_key] = default_depth
        with col_slider:
            depth = st.slider("Slice", 0, max_depth, st.session_state[depth_key], key=depth_key, label_visibility="collapsed")

        zoom_key = "knife_zoom"
        if zoom_key not in st.session_state:
            st.session_state[zoom_key] = 1.0
        zoom = st.session_state[zoom_key]

        st.markdown("##### Zoom")
        col_z1, col_z2, col_z3, col_z4 = st.columns([1, 1, 1, 3])
        with col_z1:
            if st.button("−", key="zoom_out", help="Zoom out"):
                st.session_state[zoom_key] = max(0.5, st.session_state[zoom_key] - 0.25)
        with col_z2:
            if st.button("+", key="zoom_in", help="Zoom in"):
                st.session_state[zoom_key] = min(4.0, st.session_state[zoom_key] + 0.25)
        with col_z3:
            if st.button("Fit", key="zoom_fit", help="Reset zoom"):
                st.session_state[zoom_key] = 1.0
        with col_z4:
            st.markdown(f"**{st.session_state[zoom_key]:.2f}×**")

        interp_key = "knife_z_interp"
        if interp_key not in st.session_state:
            st.session_state[interp_key] = False
        c_int1, c_int2 = st.columns([1, 3])
        with c_int1:
            interp_z = st.checkbox("Z-interp", value=st.session_state[interp_key], key=interp_key, help="Interpolate slices in z so coronal/sagittal views look square.")
        with c_int2:
            if interp_z and not SCIPY_OK:
                st.caption("⚠️ scipy not installed — skipping interpolation")
            elif interp_z and axis != "Axial":
                st.caption(f"Original z={nz} → interpolated to ~{max(nz, ny, nx)} slices")

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

        img = _slice_to_img(sl, wl2, ww2)
        img = _scale_up_img(img)
        if zoom != 1.0:
            new_w = max(1, int(img.width * zoom))
            new_h = max(1, int(img.height * zoom))
            img = img.resize((new_w, new_h), Image.LANCZOS)

        col_main, col_side = st.columns([3, 1])
        with col_main:
            st.image(img, caption=slice_label, use_container_width=True)
        with col_side:
            st.markdown(f"**Slice info**\n- **Axis:** {axis}\n- **Depth:** {depth} / {max_depth}\n- **Shape:** {sl.shape[0]} × {sl.shape[1]}\n- **Window:** {win_name2}\n- **W / L:** {ww2} / {wl2}\n- **Zoom:** {zoom:.2f}×\n- **Voxel:** {voxel_ml * 1000:.2f} mm³")

        st.markdown("---")
        st.markdown("##### 3D Orientation — Where am I in the volume?")
        _render_3view_inset(vol, axis, depth, wl2, ww2, nz, ny, nx)

        st.markdown("---")
        st.markdown("##### HU Histogram — Tissue Analysis")
        show_tissue_labels = st.checkbox("Show tissue labels", value=True, key="knife_show_tissue_labels")
        flat = sl.flatten()
        fig_hist = _render_annotated_histogram(flat, show_labels=show_tissue_labels)
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        detected = _detect_tissues_in_slice(flat)
        if detected:
            st.markdown(f"**Tissues detected in this slice:** " + ", ".join(f"`{t}`" for t in detected))

    with tab_seg:
        st.subheader("Segmentation Painter")
        st.caption(f"Voxel volume = {voxel_ml:.4f} mL")
        HU_PRESETS = {
            "Liver": (40, 80), "Kidney Cortex": (20, 50), "Spleen": (40, 60),
            "Fat": (-120, -60), "Blood/Vessel": (45, 80), "Soft Tissue": (20, 80),
            "Cortical Bone": (400, 2000), "Cancellous Bone": (100, 400),
            "Lung": (-700, -500), "Air": (-1100, -900),
        }
        preset = st.selectbox("Tissue preset", list(HU_PRESETS.keys()))
        lo, hi = HU_PRESETS[preset]
        max_seg_z = max(1, nz - 1)
        z_seg = st.slider("Axial slice", 0, max_seg_z, min(max_seg_z, nz // 2), key="seg_z")
        win_s = st.selectbox("Window", list(WINDOWS.keys()), key="seg_win")
        wls, wws = WINDOWS[win_s]
        sl = vol[z_seg]
        base = _slice_to_img(sl, wls, wws).convert("RGB")
        mask = (sl >= lo) & (sl <= hi)
        overlay = np.array(base)
        overlay[mask] = [255, 140, 0]
        vol_ml_seg = float(mask.sum()) * voxel_ml
        st.image(_scale_up_img(Image.fromarray(overlay)), caption=f"{preset} overlay — {vol_ml_seg:.2f} mL", use_container_width=True)
        st.metric("Segmented volume", f"{vol_ml_seg:.2f} mL")
        if st.button("💾 Save segmentation session", key="seg_save"):
            record_session(student_id, "ct", 80.0, f"Seg:{preset} vol:{vol_ml_seg:.2f}mL")
            st.success("Saved!")

    with tab_code:
        _render_code_lab(vol, nz, ny, nx, student_id)
    with tab_challenge:
        _render_challenges(vol, nz, ny, nx, spacing, voxel_ml, student_id)

    with tab_3d:
        st.subheader("🧊 3-D Volume Rendering")
        st.markdown("**⚡ Quick Window Presets**")
        _pcols = st.columns(5)
        _presets = [("🧠 Brain", 0, 80), ("🦴 Bone", 300, 1000), ("🫁 Lung", -900, -500), ("🩸 Soft Tissue", -100, 150), ("👁️ Head/Neck", -200, 200)]
        for _pi, (_plbl, _plo, _phi) in enumerate(_presets):
            with _pcols[_pi]:
                if st.button(_plbl, key=f"preset_{_pi}"):
                    st.session_state["ct3d_lower"] = _plo
                    st.session_state["ct3d_upper"] = _phi
        _vol3d = st.session_state.get("ct_volume", st.session_state.get("ct_vol"))
        if _vol3d is None:
            st.warning("⬅️ Load a phantom or upload a DICOM in the **📂 Load Case** tab first.")
        else:
            if _vol3d.shape[0] <= 2:
                st.info("ℹ️ You've loaded a single 2D image rather than a full CT series, so there's almost no depth to build a 3D shape from. The render below will look like a thin, mostly flat wafer rather than a full anatomical structure -- upload a multi-slice DICOM series (a .zip of many .dcm files) for a realistic 3D model.")
            _c1, _c2, _c3 = st.columns(3)
            with _c1:
                _lo = st.slider("Lower HU threshold", -1000, 500, st.session_state.get("ct3d_lower", -200), 10, key="ct3d_lower_sl")
            with _c2:
                _hi = st.slider("Upper HU threshold", -500, 2000, st.session_state.get("ct3d_upper", 400), 10, key="ct3d_upper_sl")
            with _c3:
                _ds = st.slider("Downsample (1=full, 4=fast)", 1, 4, 2, key="ct3d_ds")
            _render_mode = st.radio("Render mode", ["🌫️ Volume Cloud", "🗿 Isosurface (solid)"], horizontal=True, key="ct3d_mode")

            _use_ai_lung = False
            if LUNGMASK_OK:
                _use_ai_lung = st.checkbox(
                    "🫁 Use AI lung segmentation (pretrained lungmask model)",
                    help="Instead of just an HU threshold, run a pretrained U-Net (lungmask/R231) to find the actual lung boundaries and render only those voxels.",
                    key="ct3d_use_lungmask",
                )
            else:
                st.caption("ℹ️ AI lung segmentation isn't available -- install the `lungmask` and `SimpleITK` packages to enable it.")

            _organ_picks = []
            _ts_fast = True
            if TOTALSEG_OK:
                _organ_picks = st.multiselect(
                    "🧠 Use AI multi-organ segmentation (pretrained TotalSegmentator)",
                    options=TOTALSEG_STRUCTURES,
                    help="Segments the exact structure(s) you pick (brain, liver, spleen, kidneys, bones, pelvic organs, etc.) using a pretrained model, and renders only those voxels instead of a raw HU-threshold blob.",
                    key="ct3d_ts_structures",
                )
                if _organ_picks:
                    _ts_fast = st.checkbox("⚡ Fast mode (lower resolution, quicker)", value=True, key="ct3d_ts_fast")
                    st.caption("⏳ Multi-organ segmentation can take anywhere from ~30s to a few minutes depending on the server -- it's a real model doing real inference, not instant.")
            else:
                st.caption("ℹ️ AI multi-organ segmentation isn't available -- install the `totalsegmentator` package to enable it.")

            if st.button("🚀 Render 3-D", key="ct3d_render", type="primary"):
                import plotly.graph_objects as go
                # Don't let the downsample stride collapse a thin axis (e.g. z=2
                # from a single duplicated slice) down to a single layer -- an
                # isosurface/volume render needs at least 2 layers per axis to
                # produce any visible geometry at all.
                _ds_z = min(_ds, max(1, _vol3d.shape[0] // 2))
                _ds_y = min(_ds, max(1, _vol3d.shape[1] // 2))
                _ds_x = min(_ds, max(1, _vol3d.shape[2] // 2))
                _v = _vol3d[::_ds_z, ::_ds_y, ::_ds_x].astype(np.float32)
                _ds_spacing = (spacing[0] * _ds_z, spacing[1] * _ds_y, spacing[2] * _ds_x)

                if _use_ai_lung:
                    with st.spinner("Running pretrained lung segmentation model..."):
                        try:
                            _lung_mask = _run_lungmask(_v, _ds_spacing)
                            _v = np.where(_lung_mask > 0, _v, -2000)  # exclude everything outside the AI-predicted lung
                        except Exception as _seg_e:
                            st.error(f"AI lung segmentation failed, falling back to plain HU thresholding: {_seg_e}")

                if _organ_picks:
                    with st.spinner(f"Running pretrained multi-organ segmentation for {', '.join(_organ_picks)}..."):
                        try:
                            _organ_masks = _run_totalsegmentator(_v, _ds_spacing, _organ_picks, fast=_ts_fast)
                            if _organ_masks:
                                _combined = np.zeros(_v.shape, dtype=bool)
                                for _name, _m in _organ_masks.items():
                                    if _m.shape == _combined.shape:
                                        _combined |= _m
                                _v = np.where(_combined, _v, -2000)  # exclude everything outside the selected structures
                            else:
                                st.warning("No matching structures found in the output -- check the structure names or try without Fast mode.")
                        except Exception as _seg_e:
                            st.error(f"AI multi-organ segmentation failed, falling back to plain HU thresholding: {_seg_e}")

                _v = np.clip(_v, _lo, _hi)
                _nz, _ny, _nx = _v.shape
                _zz, _yy, _xx = np.mgrid[0:_nz, 0:_ny, 0:_nx]
                if "Volume" in _render_mode:
                    _fig = go.Figure(go.Volume(x=_xx.flatten(), y=_yy.flatten(), z=_zz.flatten(), value=_v.flatten(), isomin=_lo, isomax=_hi, opacity=0.15, opacityscale=[[0,0],[0.1,0.01],[0.3,0.05],[0.6,0.2],[1.0,0.6]], surface_count=25, colorscale="Hot"))
                    _fig.update_layout(scene=dict(bgcolor="black", xaxis=dict(showticklabels=False, showgrid=False), yaxis=dict(showticklabels=False, showgrid=False), zaxis=dict(showticklabels=False, showgrid=False)), paper_bgcolor="black", font_color="white", margin=dict(l=0, r=0, t=30, b=0), title=dict(text="Volume Cloud Render", font=dict(color="white")), height=600)
                else:
                    _fig = go.Figure(go.Isosurface(x=_xx.flatten(), y=_yy.flatten(), z=_zz.flatten(), value=_v.flatten(), isomin=_lo, isomax=_hi, surface_count=3, colorscale="Plasma", caps=dict(x_show=False, y_show=False, z_show=False)))
                    _fig.update_layout(scene=dict(bgcolor="black", xaxis=dict(showticklabels=False, showgrid=False), yaxis=dict(showticklabels=False, showgrid=False), zaxis=dict(showticklabels=False, showgrid=False)), paper_bgcolor="black", font_color="white", margin=dict(l=0, r=0, t=30, b=0), title=dict(text="Isosurface Render", font=dict(color="white")), height=600)
                st.plotly_chart(_fig, use_container_width=True)
            with st.expander("📖 HU Tissue Reference"):
                st.dataframe(pd.DataFrame({"Tissue": ["Air","Lung","Fat","Water/CSF","Soft Tissue","Blood","Contrast","Bone Trabecular","Bone Cortical"], "HU Min": [-1000,-900,-120,-10,20,30,100,200,700], "HU Max": [-950,-600,-50,15,80,45,300,400,1900]}), use_container_width=True, hide_index=True)

CODE_LAB_TOTAL = 8
def _render_code_lab(vol, nz, ny, nx, student_id):
    st.subheader("🐍 Code Lab — Python Array Slicing")
    st.markdown("Master CT-volume indexing by solving these exercises.")
    if "ct_exercises_done" not in st.session_state:
        st.session_state.ct_exercises_done = set()
    done = st.session_state.ct_exercises_done
    n_done = len(done)
    st.progress(n_done / CODE_LAB_TOTAL, text=f"Progress: {n_done}/{CODE_LAB_TOTAL} exercises completed")
    ex_num = st.selectbox("Choose an exercise", list(range(1, CODE_LAB_TOTAL + 1)), format_func=lambda n: f"Exercise {n} {'✅' if n in done else ''}", key="codelab_ex_select")
    mid_z, mid_y, mid_x = nz // 2, ny // 2, nx // 2
    WL_DEFAULT, WW_DEFAULT = 40, 400
    def _show_slice(arr2d, caption):
        st.image(_scale_up_img(_slice_to_img(arr2d, WL_DEFAULT, WW_DEFAULT)), caption=caption, width=300)
    if ex_num == 1:
        st.markdown("### Exercise 1 — Axial Slice")
        ans1 = st.radio("Select the correct expression:", ["A) `vol[30]`", "B) `vol[:, 30, :]`", "C) `vol[:, :, 30]`", "D) `vol[30, :, :]`"], key="ex1_radio")
        if st.button("✅ Submit Answer", key="ex1_submit"):
            if "A)" in ans1 or "D)" in ans1:
                st.success("✅ Correct!")
                done.add(1); st.session_state.ct_exercises_done = done
                _show_slice(vol[min(30, nz - 1)], "vol[30, :, :] — Axial slice 30")
            else: st.error("❌ Try again. Hint: The first axis (axis 0) is the slice direction.")
    elif ex_num == 2:
        st.markdown("### Exercise 2 — Coronal Slice")
        ans2 = st.radio("Select the correct expression:", ["A) `vol[y, :, :]`", "B) `vol[:, y, :]`", "C) `vol[:, :, y]`", "D) `vol[y]`"], key="ex2_radio")
        if st.button("✅ Submit Answer", key="ex2_submit"):
            if "B)" in ans2:
                st.success("✅ Correct!")
                done.add(2); st.session_state.ct_exercises_done = done
                _show_slice(vol[:, mid_y, :], f"vol[:, {mid_y}, :] — Coronal slice at midpoint y={mid_y}")
            else: st.error("❌ Try again. Hint: Axis 1 is the row direction.")
    elif ex_num == 3:
        st.markdown("### Exercise 3 — Sagittal Slice")
        ans3 = st.radio("Select the correct expression:", ["A) `vol[:, :, x]`", "B) `vol[x, :, :]`", "C) `vol[:, x, :]`", "D) `vol[x]`"], key="ex3_radio")
        if st.button("✅ Submit Answer", key="ex3_submit"):
            if "A)" in ans3:
                st.success("✅ Correct!")
                done.add(3); st.session_state.ct_exercises_done = done
                _show_slice(vol[:, :, mid_x], f"vol[:, :, {mid_x}] — Sagittal slice at midpoint x={mid_x}")
            else: st.error("❌ Try again. Hint: Axis 2 is the column direction.")
    elif ex_num == 4:
        st.markdown("### Exercise 4 — Step Slicing (Every Other Slice)")
        ans4 = st.radio("Select the correct expression:", ["A) `vol[::2, :, :]`", "B) `vol[2:, :, :]`", "C) `vol[:2, :, :]`", "D) `vol[2::2, :, :]`"], key="ex4_radio")
        if st.button("✅ Submit Answer", key="ex4_submit"):
            if "A)" in ans4:
                st.success("✅ Correct!")
                done.add(4); st.session_state.ct_exercises_done = done
                st.markdown(f"**Full stack:** `vol.shape = {vol.shape}`  \n**Every other slice:** `vol[::2].shape = {vol[::2].shape}`")
            else: st.error("❌ Try again. Hint: Use start:stop:step notation.")
    elif ex_num == 5:
        st.markdown("### Exercise 5 — Reverse Axial Order")
        ans5 = st.radio("Select the correct expression:", ["A) `vol[-1, :, :]`", "B) `vol[::-1, :, :]`", "C) `vol[0:-1, :, :]`", "D) `vol.flip()`"], key="ex5_radio")
        if st.button("✅ Submit Answer", key="ex5_submit"):
            if "B)" in ans5:
                st.success("✅ Correct!")
                done.add(5); st.session_state.ct_exercises_done = done
                _show_slice(vol[-1], "vol[-1] — first slice after reversal (original last slice)")
            else: st.error("❌ Try again. Hint: `[::-1]` reverses a Python list.")
    elif ex_num == 6:
        st.markdown("### Exercise 6 — Boolean Masking (Segmentation)")
        ans6 = st.radio("Select the correct expression:", ["A) `vol[40:80]`", "B) `(vol >= 40) & (vol <= 80)`", "C) `vol == [40, 80]`", "D) `vol.mask(40, 80)`"], key="ex6_radio")
        if st.button("✅ Submit Answer", key="ex6_submit"):
            if "B)" in ans6:
                st.success("✅ Correct!")
                done.add(6); st.session_state.ct_exercises_done = done
                sl_liver = vol[mid_z]
                liver_mask = (sl_liver >= 40) & (sl_liver <= 80)
                base_img = _slice_to_img(sl_liver, WL_DEFAULT, WW_DEFAULT).convert("RGB")
                overlay = np.array(base_img)
                overlay[liver_mask] = [255, 140, 0]
                st.image(_scale_up_img(Image.fromarray(overlay)), caption="Liver mask (HU 40–80) shown in orange", width=300)
            else: st.error("❌ Try again. Hint: Use comparison operators (`>=`, `<=`) with the bitwise `&` operator.")
    elif ex_num == 7:
        st.markdown("### Exercise 7 — Count Masked Voxels")
        ans7 = st.radio("Select the correct expression:", ["A) `len(mask)`", "B) `mask.count()`", "C) `mask.sum()`", "D) `np.count(mask)`"], key="ex7_radio")
        if st.button("✅ Submit Answer", key="ex7_submit"):
            if "C)" in ans7:
                st.success("✅ Correct!")
                done.add(7); st.session_state.ct_exercises_done = done
            else: st.error("❌ Try again. Hint: In NumPy, `True == 1` and `False == 0`.")
    elif ex_num == 8:
        st.markdown("### Exercise 8 — Volume Calculation")
        st.info("Each voxel is **0.5 mm × 0.5 mm × 1.0 mm**.  \nYou have **4800 liver voxels**.  \n**What is the liver volume in mL?** (1 mL = 1000 mm³)")
        user_answer = st.number_input("Your answer (mL):", min_value=0.0, max_value=100.0, value=0.0, step=0.01, key="ex8_number")
        CORRECT_ANSWER_ML = 4800 * 0.5 * 0.5 * 1.0 / 1000
        if st.button("✅ Submit Answer", key="ex8_submit"):
            if abs(user_answer - CORRECT_ANSWER_ML) <= 0.05:
                st.success(f"✅ Correct! The liver volume is **{CORRECT_ANSWER_ML:.2f} mL**.")
                done.add(8); st.session_state.ct_exercises_done = done
            else: st.error("❌ Try again. Hint: voxel volume = 0.5 × 0.5 × 1.0 = 0.25 mm³.")

CHALLENGE_TOTAL = 5
def _challenge_status_icon(cnum):
    unlocked = st.session_state.get("ct_challenge_unlocked", {})
    passed = st.session_state.get("ct_challenge_passed", {})
    if passed.get(cnum, False): return "✅"
    if unlocked.get(cnum, False) or cnum == 1: return "🎯"
    return "🔒"

def _init_challenge_state():
    if "ct_challenge_unlocked" not in st.session_state: st.session_state.ct_challenge_unlocked = {1: True}
    if "ct_challenge_passed" not in st.session_state: st.session_state.ct_challenge_passed = {}
    if "ct_challenge_scores" not in st.session_state: st.session_state.ct_challenge_scores = {}
    if "ct_challenge_tries" not in st.session_state: st.session_state.ct_challenge_tries = {}

def _render_challenges(vol, nz, ny, nx, spacing, voxel_ml, student_id):
    st.subheader("🎯 Guided Challenge Mode")
    _init_challenge_state()
    unlocked = st.session_state.ct_challenge_unlocked
    passed = st.session_state.ct_challenge_passed
    scores = st.session_state.ct_challenge_scores
    tries = st.session_state.ct_challenge_tries
    total_score = sum(scores.values())
    st.metric("Challenge Score", f"{total_score} / 500")
    st.progress(total_score / 500.0, text=f"{total_score}/500 points")

    c1_unlocked = unlocked.get(1, True)
    c1_passed = passed.get(1, False)
    with st.expander(f"{_challenge_status_icon(1)} Challenge 1 — Find the Organ (MPR Navigation)", expanded=(c1_unlocked and not c1_passed)):
        if not c1_unlocked: st.warning("🔒 Complete Challenge 0 first (no predecessor — always unlocked).")
        else:
            abdomen_vol = _make_abdomen()
            c1_nz = abdomen_vol.shape[0]
            liver_counts = np.array([int(np.sum((abdomen_vol[z] >= 40) & (abdomen_vol[z] <= 80))) for z in range(c1_nz)])
            true_max_slice = int(np.argmax(liver_counts))
            c1_z = st.slider("Axial slice (Abdomen case)", 0, c1_nz - 1, c1_nz // 2, key="c1_slider")
            sl_c1 = abdomen_vol[c1_z]
            liver_mask_c1 = (sl_c1 >= 40) & (sl_c1 <= 80)
            base_c1 = _slice_to_img(sl_c1, 60, 400).convert("RGB")
            ov_c1 = np.array(base_c1)
            ov_c1[liver_mask_c1] = [255, 140, 0]
            st.image(_scale_up_img(Image.fromarray(ov_c1)), caption=f"Axial z={c1_z}", width=300)
            if not c1_passed:
                if st.button("📤 Submit: this is the maximum-liver slice", key="c1_submit"):
                    diff = abs(c1_z - true_max_slice)
                    sc = 100 if diff <= 2 else (80 if diff <= 5 else 60)
                    scores[1] = sc; passed[1] = True; unlocked[2] = True
                    st.session_state.ct_challenge_scores = scores; st.session_state.ct_challenge_passed = passed; st.session_state.ct_challenge_unlocked = unlocked
                    st.success(f"✅ Challenge 1 complete! Score: **{sc}/100**  \nYour slice: {c1_z} | True max-liver slice: {true_max_slice}")
            else: st.success(f"✅ Already passed! Score: {scores.get(1, 0)}/100")

    c2_unlocked = unlocked.get(2, False)
    c2_passed = passed.get(2, False)
    with st.expander(f"{_challenge_status_icon(2)} Challenge 2 — Window the Brain", expanded=(c2_unlocked and not c2_passed)):
        if not c2_unlocked: st.warning("🔒 Complete Challenge 1 first.")
        else:
            brain_vol = _make_brain()
            b_mid = brain_vol.shape[0] // 2
            c2_preset = st.radio("Choose window preset:", ["Brain (W:80, L:40)", "Lung (W:1500, L:-600)", "Bone (W:2000, L:400)", "Soft Tissue (W:400, L:40)"], key="c2_preset")
            preset_map = {"Brain (W:80, L:40)": (40, 80), "Lung (W:1500, L:-600)": (-600, 1500), "Bone (W:2000, L:400)": (400, 2000), "Soft Tissue (W:400, L:40)": (40, 400)}
            wl_c2, ww_c2 = preset_map[c2_preset]
            st.image(_scale_up_img(_slice_to_img(brain_vol[b_mid], wl_c2, ww_c2)), caption=f"Brain CT with {c2_preset}", width=300)
            if not c2_passed:
                if st.button("📤 Submit window choice", key="c2_submit"):
                    if "Brain" in c2_preset:
                        sc2 = 100; scores[2] = sc2; passed[2] = True; unlocked[3] = True
                        st.session_state.ct_challenge_scores = scores; st.session_state.ct_challenge_passed = passed; st.session_state.ct_challenge_unlocked = unlocked
                        st.success("✅ Correct! Score: **100/100**")
                    else: st.error("❌ Incorrect. Hint: Which preset has the narrowest window?")
            else: st.success(f"✅ Already passed! Score: {scores.get(2, 0)}/100")

    c3_unlocked = unlocked.get(3, False)
    c3_passed = passed.get(3, False)
    with st.expander(f"{_challenge_status_icon(3)} Challenge 3 — Segment the Lung", expanded=(c3_unlocked and not c3_passed)):
        if not c3_unlocked: st.warning("🔒 Complete Challenge 2 first.")
        else:
            chest_vol = _make_chest()
            c3_mid = chest_vol.shape[0] // 2
            lo_c3 = st.slider("Low HU threshold", -1100, 0, -750, 10, key="c3_lo")
            hi_c3 = st.slider("High HU threshold", -1000, 200, -400, 10, key="c3_hi")
            sl_c3 = chest_vol[c3_mid]
            seg_mask_c3 = (sl_c3 >= lo_c3) & (sl_c3 <= hi_c3)
            base_c3 = _slice_to_img(sl_c3, -600, 1500).convert("RGB")
            ov_c3 = np.array(base_c3)
            ov_c3[seg_mask_c3] = [0, 180, 255]
            st.image(_scale_up_img(Image.fromarray(ov_c3)), caption=f"Lung segmentation overlay (HU {lo_c3} to {hi_c3})", width=300)
            if not c3_passed:
                if st.button("📤 Submit segmentation", key="c3_submit"):
                    sc3 = 100; scores[3] = sc3; passed[3] = True; unlocked[4] = True
                    st.session_state.ct_challenge_scores = scores; st.session_state.ct_challenge_passed = passed; st.session_state.ct_challenge_unlocked = unlocked
                    st.success(f"✅ Challenge 3 complete! Score: **{sc3}/100**")
            else: st.success(f"✅ Already passed! Score: {scores.get(3, 0)}/100")

    c4_unlocked = unlocked.get(4, False)
    c4_passed = passed.get(4, False)
    with st.expander(f"{_challenge_status_icon(4)} Challenge 4 — Identify the Tissue", expanded=(c4_unlocked and not c4_passed)):
        if not c4_unlocked: st.warning("🔒 Complete Challenge 3 first.")
        else:
            abd_vol = _make_abdomen()
            c4_z = abd_vol.shape[0] // 2
            sl_c4 = abd_vol[c4_z]
            fig_c4 = _render_annotated_histogram(sl_c4.flatten(), show_labels=True)
            st.pyplot(fig_c4); plt.close(fig_c4)
            ground_truth = set(_detect_tissues_in_slice(sl_c4.flatten()))
            all_tissue_names = [t[0] for t in TISSUE_BANDS]
            c4_selected = st.multiselect("Select all tissues present in this slice:", all_tissue_names, key="c4_multiselect")
            if not c4_passed:
                if st.button("📤 Submit tissue identification", key="c4_submit"):
                    selected_set = set(c4_selected)
                    wrong = len(selected_set.symmetric_difference(ground_truth))
                    sc4 = max(0, 100 - 10 * wrong); scores[4] = sc4
                    if sc4 >= 60: passed[4] = True; unlocked[5] = True
                    st.session_state.ct_challenge_scores = scores; st.session_state.ct_challenge_passed = passed; st.session_state.ct_challenge_unlocked = unlocked
                    if sc4 >= 60: st.success(f"✅ Challenge 4 complete! Score: **{sc4}/100**")
                    else: st.error(f"❌ Score: {sc4}/100 — too many wrong answers ({wrong}).")
            else: st.success(f"✅ Already passed! Score: {scores.get(4, 0)}/100")

    c5_unlocked = unlocked.get(5, False)
    c5_passed = passed.get(5, False)
    with st.expander(f"{_challenge_status_icon(5)} Challenge 5 — Write the Slice Expression", expanded=(c5_unlocked and not c5_passed)):
        if not c5_unlocked: st.warning("🔒 Complete Challenge 4 first.")
        else:
            ans_c5 = st.radio("Select the correct expression:", ["A) `vol[:, ::3, :]`", "B) `vol[::3, :, :]`", "C) `vol[:, :, ::3]`", "D) `vol[::3]`"], key="c5_radio")
            if not c5_passed:
                if st.button("📤 Submit expression", key="c5_submit"):
                    if "A)" in ans_c5:
                        sc5 = 100; scores[5] = sc5; passed[5] = True
                        st.session_state.ct_challenge_scores = scores; st.session_state.ct_challenge_passed = passed
                        st.success(f"✅ Challenge 5 complete! Score: **{sc5}/100**")
                    else: st.error("❌ Incorrect. Hint: Coronal slices vary along axis 1 (rows). Use `::3` on axis 1 only.")
            else: st.success(f"✅ Already passed! Score: {scores.get(5, 0)}/100")

    total_now = sum(st.session_state.ct_challenge_scores.values())
    if len(st.session_state.ct_challenge_passed) == CHALLENGE_TOTAL:
        st.balloons()
        st.success(f"🏆 All challenges complete! Final score: **{total_now} / 500**")
        if st.button("💾 Save challenge results", key="challenge_save"):
            record_session(student_id, "ct", total_now / 5.0, f"Challenges complete: {total_now}/500")
            st.success("Results saved!")
