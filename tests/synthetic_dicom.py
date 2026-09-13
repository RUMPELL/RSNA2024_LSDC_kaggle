"""Minimal synthetic DICOM writer for tests. No real imaging data is used."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid


def write_synthetic_dicom(
    path: Path,
    pixels: np.ndarray,
    photometric: str = "MONOCHROME2",
    image_position: Optional[Sequence[float]] = None,
) -> Path:
    """Write a tiny uncompressed 16-bit MR DICOM containing `pixels`."""
    pixels = np.asarray(pixels, dtype=np.uint16)
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    if image_position is not None:
        ds.ImagePositionPatient = [float(v) for v in image_position]
    ds.PixelData = pixels.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)
    return path


def read_dataset(path: Path) -> pydicom.Dataset:
    return pydicom.dcmread(str(path))
