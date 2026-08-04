"""Medical image input: DICOM, NIfTI and 3-D volume handling.

Why this exists
---------------
Everything upstream assumes a single 8-bit axial PNG/JPEG, which is what the
public training set ships. Nothing in a hospital looks like that. Real input is
a DICOM instance, a zipped DICOM series, or a NIfTI volume — 12–16 bit, with
rescale slope/intercept and window settings that must be applied before the
pixels mean anything, and often a whole volume rather than one slice.

Two jobs:

  1. Decode and window correctly. A DICOM read as raw integers and rescaled by
     min/max looks plausible and is wrong — the stored values are not display
     values until slope/intercept and the window are applied.

  2. Pick the right slice. The classifier was trained at one axial level (body
     of the lateral ventricles). Handing it an arbitrary slice from a volume
     produces a confident answer about anatomy the model has never seen, so
     `select_axial_slice` scores every slice for how well it matches that level
     and returns the best one, plus neighbours for multi-slice aggregation.

Both pydicom and nibabel are optional. If they are absent the module degrades
to "PNG/JPEG only" and says so rather than failing at import.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
    HAS_DICOM = True
except ImportError:  # pragma: no cover
    HAS_DICOM = False

    class InvalidDicomError(Exception):  # type: ignore[no-redef]
        pass

try:
    import nibabel as nib
    HAS_NIFTI = True
except ImportError:  # pragma: no cover
    HAS_NIFTI = False


DICOM_MAGIC = b"DICM"
NIFTI_MAGICS = (b"\x5c\x01\x00\x00", b"n+1\x00", b"ni1\x00")


@dataclass
class LoadedScan:
    """A decoded scan, always exposing a single 2-D slice to classify."""

    image: Image.Image
    modality: str                      # "MR", "CT", "unknown"
    source_format: str                 # "dicom" | "dicom-series" | "nifti" | "image"
    is_volume: bool
    n_slices: int
    selected_index: int | None
    slice_scores: list[float] | None
    neighbours: list[Image.Image]      # slices around the selected one
    warnings: list[str]
    meta: dict[str, Any]


# ---------------------------------------------------------------- detection
def sniff(data: bytes) -> str:
    if len(data) > 132 and data[128:132] == DICOM_MAGIC:
        return "dicom"
    if data[:4] in NIFTI_MAGICS or data[:2] == b"\x1f\x8b":
        # gzip may be a .nii.gz; confirmed later by the loader
        return "nifti"
    if data[:2] == b"PK":
        return "zip"
    return "image"


# ------------------------------------------------------------------ DICOM
def _apply_dicom_windowing(ds, arr: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Stored values -> display values. Order matters and is defined by the
    DICOM standard: modality LUT (slope/intercept) first, then the VOI LUT
    (window centre/width)."""
    warnings: list[str] = []
    arr = arr.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept

    centre = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)
    if centre is not None and width is not None:
        # These are MultiValue when several presets are stored; take the first.
        if hasattr(centre, "__iter__") and not isinstance(centre, (str, bytes)):
            centre = centre[0]
        if hasattr(width, "__iter__") and not isinstance(width, (str, bytes)):
            width = width[0]
        centre, width = float(centre), float(width)
        if width > 1:
            lo, hi = centre - width / 2, centre + width / 2
            arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        else:
            warnings.append("DICOM window width was degenerate; used min/max.")
            arr = _minmax(arr)
    else:
        warnings.append(
            "No WindowCenter/WindowWidth in the DICOM; fell back to min/max "
            "scaling, which can differ from how a workstation displays it.")
        arr = _minmax(arr)

    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = 1.0 - arr
    return arr, warnings


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(np.percentile(arr, 0.5)), float(np.percentile(arr, 99.5))
    if hi - lo < 1e-6:
        lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _dicom_meta(ds) -> dict[str, Any]:
    keys = ("Modality", "MagneticFieldStrength", "SliceThickness",
            "PixelSpacing", "SeriesDescription", "ProtocolName",
            "Manufacturer", "ImageOrientationPatient", "Rows", "Columns")
    out: dict[str, Any] = {}
    for k in keys:
        v = getattr(ds, k, None)
        if v is None:
            continue
        if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
            v = [float(x) for x in v]
        elif not isinstance(v, (str, int, float)):
            v = str(v)
        out[k] = v
    return out


def orientation_from_dicom(ds) -> str | None:
    """Derive the acquisition plane from ImageOrientationPatient.

    The tag holds the direction cosines of the row and column axes; their cross
    product is the slice normal, and whichever patient axis it aligns with names
    the plane. This is the only reliable way to know the plane — far better than
    inferring it from pixels.
    """
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None or len(iop) < 6:
        return None
    row = np.array([float(v) for v in iop[:3]])
    col = np.array([float(v) for v in iop[3:6]])
    normal = np.cross(row, col)
    axis = int(np.argmax(np.abs(normal)))
    return {0: "sagittal", 1: "coronal", 2: "axial"}[axis]


# ----------------------------------------------------------- slice scoring
def score_axial_slice(arr: np.ndarray) -> float:
    """How much does this slice look like the training level?

    The model was trained on axial slices at the body of the lateral
    ventricles. Such a slice has: a large centred brain cross-section, strong
    left-right mirror symmetry, and a dark central CSF structure. Slices near
    the vertex or the skull base fail one or more of these.
    """
    if arr.size == 0:
        return 0.0
    a = arr.astype(np.float32)
    if a.max() > a.min():
        a = (a - a.min()) / (a.max() - a.min())

    fg = a > max(0.10, float(a.mean()) * 0.5)
    area = float(fg.mean())
    if area < 0.04:
        return 0.0

    flipped = a[:, ::-1]
    denom = float(np.linalg.norm(a) * np.linalg.norm(flipped))
    symmetry = float((a * flipped).sum() / denom) if denom > 0 else 0.0

    h, w = a.shape
    cy0, cy1 = int(h * 0.35), int(h * 0.65)
    cx0, cx1 = int(w * 0.35), int(w * 0.65)
    core = a[cy0:cy1, cx0:cx1]
    core_fg = fg[cy0:cy1, cx0:cx1]
    # Ventricles: dark pixels inside the central brain region.
    dark_core = float((core_fg & (core < 0.35)).mean()) if core_fg.any() else 0.0

    # Prefer mid-range cross-sectional area: the vertex and base are small.
    area_term = float(np.exp(-((area - 0.40) ** 2) / (2 * 0.12 ** 2)))
    dark_term = float(np.exp(-((dark_core - 0.10) ** 2) / (2 * 0.07 ** 2)))
    return float(0.45 * area_term + 0.35 * symmetry + 0.20 * dark_term)


def select_axial_slice(volume: np.ndarray, axis: int = 2,
                       neighbours: int = 2) -> tuple[int, list[float], list[int]]:
    """Pick the slice best matching the training level, plus its neighbours."""
    n = volume.shape[axis]
    scores = []
    for i in range(n):
        sl = np.take(volume, i, axis=axis)
        scores.append(score_axial_slice(sl))
    best = int(np.argmax(scores))
    idx = [i for i in range(best - neighbours, best + neighbours + 1)
           if 0 <= i < n]
    return best, [float(s) for s in scores], idx


def guess_axial_axis(volume: np.ndarray, zooms: tuple[float, ...] | None = None
                     ) -> int:
    """Axis along which slices are axial.

    With voxel sizes available, the through-plane axis is normally the one with
    the coarsest spacing. Without them, fall back to the shortest axis, which is
    the usual layout for clinical brain series.
    """
    if zooms and len(zooms) >= 3:
        return int(np.argmax(zooms[:3]))
    return int(np.argmin(volume.shape[:3]))


def _to_image(arr: np.ndarray) -> Image.Image:
    a = _minmax(arr.astype(np.float32))
    return Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")


# ------------------------------------------------------------------ loaders
def _load_dicom_single(data: bytes) -> LoadedScan:
    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    arr = ds.pixel_array
    warnings: list[str] = []
    meta = _dicom_meta(ds)
    plane = orientation_from_dicom(ds)
    if plane:
        meta["derived_plane"] = plane
        if plane != "axial":
            warnings.append(
                f"ImageOrientationPatient indicates a {plane} slice. This model "
                f"was trained on axial slices only; the result will not be "
                f"meaningful.")

    modality = str(getattr(ds, "Modality", "unknown") or "unknown")
    if modality == "CT":
        warnings.append(
            "Modality is CT. This model was trained on T1 MRI; CT has entirely "
            "different tissue contrast and the output will be meaningless.")

    if arr.ndim == 3 and arr.shape[0] > 1:  # multi-frame instance
        vol = np.transpose(arr, (1, 2, 0))
        best, scores, idx = select_axial_slice(vol, axis=2)
        windowed = [_apply_dicom_windowing(ds, vol[:, :, i])[0] for i in idx]
        main, w = _apply_dicom_windowing(ds, vol[:, :, best])
        warnings += w
        return LoadedScan(_to_image(main), modality, "dicom", True,
                          vol.shape[2], best, scores,
                          [_to_image(x) for x in windowed], warnings, meta)

    if arr.ndim == 3:
        arr = arr[0]
    display, w = _apply_dicom_windowing(ds, arr)
    warnings += w
    return LoadedScan(_to_image(display), modality, "dicom", False, 1, None,
                      None, [], warnings, meta)


def _load_dicom_series(data: bytes) -> LoadedScan:
    slices = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            raw = zf.read(name)
            if len(raw) < 132:
                continue
            try:
                ds = pydicom.dcmread(io.BytesIO(raw), force=True)
                if not hasattr(ds, "pixel_array"):
                    continue
                pos = getattr(ds, "InstanceNumber", None)
                ipp = getattr(ds, "ImagePositionPatient", None)
                key = float(ipp[2]) if ipp is not None and len(ipp) > 2 else (
                    float(pos) if pos is not None else len(slices))
                slices.append((key, ds))
            except (InvalidDicomError, Exception):
                continue

    if not slices:
        raise ValueError("No readable DICOM instances found in the archive.")

    slices.sort(key=lambda t: t[0])
    ref = slices[len(slices) // 2][1]
    vol = np.stack([_apply_dicom_windowing(ds, ds.pixel_array)[0]
                    for _, ds in slices], axis=2)

    best, scores, idx = select_axial_slice(vol, axis=2)
    meta = _dicom_meta(ref)
    meta["n_instances"] = len(slices)
    plane = orientation_from_dicom(ref)
    warnings: list[str] = []
    if plane:
        meta["derived_plane"] = plane
        if plane != "axial":
            warnings.append(
                f"Series orientation is {plane}; reslicing is not implemented, "
                f"so the selected image is a {plane} slice the model was never "
                f"trained on.")
    modality = str(getattr(ref, "Modality", "unknown") or "unknown")
    if modality == "CT":
        warnings.append("Modality is CT; this model expects T1 MRI.")

    return LoadedScan(_to_image(vol[:, :, best]), modality, "dicom-series",
                      True, vol.shape[2], best, scores,
                      [_to_image(vol[:, :, i]) for i in idx], warnings, meta)


def _load_nifti(data: bytes) -> LoadedScan:
    holder = nib.FileHolder(fileobj=io.BytesIO(data))
    try:
        img = nib.Nifti1Image.from_file_map(
            {"header": holder, "image": holder})
    except Exception:
        img = nib.Nifti2Image.from_file_map(
            {"header": holder, "image": holder})

    arr = np.asanyarray(img.dataobj, dtype=np.float32)
    warnings: list[str] = []
    if arr.ndim == 4:
        arr = arr[..., 0]
        warnings.append("4-D NIfTI: used the first volume.")
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D NIfTI volume, got shape {arr.shape}.")

    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    # Reorient to canonical RAS so the axial axis is the third one.
    try:
        canonical = nib.as_closest_canonical(img)
        arr = np.asanyarray(canonical.dataobj, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        axis = 2
        meta_orient = "reoriented to canonical RAS"
    except Exception:
        axis = guess_axial_axis(arr, zooms)
        meta_orient = f"could not reorient; assumed axial axis {axis}"
        warnings.append(meta_orient)

    best, scores, idx = select_axial_slice(arr, axis=axis)
    main = np.take(arr, best, axis=axis)
    # NIfTI voxel order puts the anterior direction along +y; images want
    # anterior at the top, so rotate for display.
    main = np.rot90(main)
    neigh = [_to_image(np.rot90(np.take(arr, i, axis=axis))) for i in idx]

    return LoadedScan(_to_image(main), "MR", "nifti", True,
                      arr.shape[axis], best, scores, neigh, warnings,
                      {"zooms": list(zooms), "shape": list(arr.shape),
                       "orientation": meta_orient})


def load_scan(data: bytes) -> LoadedScan:
    """Decode any supported input into a single classifiable 2-D slice."""
    kind = sniff(data)

    if kind == "dicom":
        if not HAS_DICOM:
            raise ValueError(
                "This looks like a DICOM file but pydicom is not installed.")
        return _load_dicom_single(data)

    if kind == "zip":
        if not HAS_DICOM:
            raise ValueError(
                "This looks like a DICOM archive but pydicom is not installed.")
        return _load_dicom_series(data)

    if kind == "nifti":
        if not HAS_NIFTI:
            raise ValueError(
                "This looks like a NIfTI volume but nibabel is not installed.")
        return _load_nifti(data)

    img = Image.open(io.BytesIO(data))
    from PIL import ImageOps
    img = ImageOps.exif_transpose(img).convert("RGB")
    return LoadedScan(img, "unknown", "image", False, 1, None, None, [], [], {})


def capabilities() -> dict[str, bool]:
    return {"dicom": HAS_DICOM, "dicom_series": HAS_DICOM, "nifti": HAS_NIFTI,
            "image": True}
