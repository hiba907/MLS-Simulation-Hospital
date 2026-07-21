"""Generate a small synthetic DICOM zip for testing the CT Reading Room.

Run locally:
    pip install pydicom numpy
    python make_test_dicom.py

Produces test_ct.zip in the current directory — upload this to the CT module.
"""
import io
import zipfile
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def make_slice(arr2d: np.ndarray, instance_number: int, slice_location: float) -> FileDataset:
    """Wrap a 2D numpy array as a CT DICOM file dataset."""
    # File meta
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID         = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID    = generate_uid()

    ds = FileDataset("", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName  = "TEST^CT^PHANTOM"
    ds.PatientID    = "TEST001"
    ds.Modality     = "CT"
    ds.StudyDate    = "20260101"
    ds.StudyDescription = "Test CT for MLS Simulation Hospital"
    ds.SeriesDescription  = "Synthetic Chest"
    ds.SliceThickness     = 0.5
    ds.PixelSpacing       = [0.5, 0.5]
    ds.SliceLocation      = slice_location
    ds.InstanceNumber     = instance_number
    ds.Rows               = arr2d.shape[0]
    ds.Columns             = arr2d.shape[1]
    ds.BitsAllocated       = 16
    ds.BitsStored          = 16
    ds.HighBit             = 15
    ds.PixelRepresentation = 1   # signed
    ds.RescaleIntercept    = -1024
    ds.RescaleSlope        = 1.0
    ds.PixelData           = (arr2d.astype(np.int16)).tobytes()
    ds.is_little_endian    = True
    ds.is_implicit_VR      = False
    return ds


def main():
    nz, ny, nx = 60, 256, 256
    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.float32)
    cz = nz / 2

    # Body outline (ellipse)
    r_body = np.sqrt(((yy - 128) / 80) ** 2 + ((xx - 128) / 70) ** 2)
    vol = np.full((nz, ny, nx), -1000.0, dtype=np.float32)
    vol[r_body < 1.0] = -100.0                      # fat
    vol[r_body < 0.7] = 50.0                        # soft tissue

    # Spine (posterior)
    r_spine = np.sqrt(((yy - 200) / 8) ** 2 + ((xx - 128) / 8) ** 2)
    vol[r_spine < 1.0] = 800.0

    # Lungs
    for ox in (90, 166):
        r_lung = np.sqrt(((yy - 110) / 35) ** 2 + ((xx - ox) / 25) ** 2)
        vol[r_lung < 1.0] = -700.0

    # Heart (center)
    r_heart = np.sqrt(((yy - 110) / 18) ** 2 + ((xx - 128) / 14) ** 2)
    vol[r_heart < 1.0] = 55.0

    # 3D gaussian noise for realism (sigma 20)
    rng = np.random.default_rng(42)
    vol = vol + rng.normal(0, 20, vol.shape).astype(np.float32)

    out_path = "test_ct.zip"
    written = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(nz):
            # Offset rescale: pixel values stored as int16 after subtracting intercept
            slice_arr = vol[i] - (-1024)
            slice_arr = np.clip(slice_arr, -32768, 32767).astype(np.int16)
            ds = make_slice(slice_arr, instance_number=i + 1, slice_location=i * 0.5)
            buf = io.BytesIO()
            pydicom.dcmwrite(buf, ds, write_like_original=False)
            buf.seek(0)
            zf.writestr(f"slice_{i+1:03d}.dcm", buf.read())
            written += 1

    print(f"✅ Wrote {out_path} with {written} DICOM slices (size {ny}x{nx}, {nz} slices, 0.5mm isotropic)")


if __name__ == "__main__":
    main()
