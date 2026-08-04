"""Anatomical localisation and morphometry for axial brain MRI slices.

Design position
---------------
The classifier says *which stage*. This module says *where it looked* and
*what is measurably abnormal*, in named anatomy. Both halves are computed from
the pixels — nothing here is generated text.

That constraint is deliberate. A language model asked to "explain the MRI"
produces fluent anatomy that was never measured, which is precisely the failure
mode a paper cannot survive. So every statement this module emits is traceable
to either

  (a) a segmented region area / intensity measured on the input, or
  (b) the fraction of the model's class activation map falling inside a
      labelled atlas region after registration.

Scope and honest limits
-----------------------
* The training corpus is **axial T1 slices at the ventricular level only**.
  Coronal/sagittal classification is not supported: there is no multi-plane
  training data. `axial_view_check` verifies the input is consistent with an
  axial slice and estimates its level; it does not classify three planes.
* The atlas is a **coarse parametric lobar atlas** defined in template space,
  not a FreeSurfer-grade parcellation. Ventricles and CSF are data-driven
  (segmented per-scan); lobar boundaries are geometric priors. Region names
  therefore denote *approximate territories*, and the reports say so.
* Registration is a similarity transform (translation, rotation, isotropic
  scale) recovered from the brain mask. No non-linear warping.

Laterality follows **radiological convention**: image-left is the patient's
RIGHT hemisphere. `RADIOLOGICAL_CONVENTION` flips this if your source differs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
NORMS_PATH = HERE / "model" / "anatomy_norms.json"

RADIOLOGICAL_CONVENTION = True
TEMPLATE_SIZE = 192  # working resolution for registration / atlas rasterisation


# ---------------------------------------------------------------------------
# Atlas
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Region:
    """A labelled territory in normalised template space.

    Coordinates are normalised so the brain's fitted ellipse spans [-1, 1] in
    both axes: x = -1 image-left, y = -1 anterior (top of an axial slice).
    """

    key: str
    name: str
    side: str  # "left" | "right" | "midline"
    lobe: str
    description: str
    # Half-plane constraints: (ax, ay, b) meaning ax*x + ay*y <= b
    constraints: tuple[tuple[float, float, float], ...] = ()
    # Optional elliptical constraint: (cx, cy, rx, ry) — inside the ellipse
    ellipse: tuple[float, float, float, float] | None = None
    ring: tuple[float, float] | None = None  # (r_inner, r_outer) radial band


def _hemisphere(side: str) -> tuple[float, float, float]:
    """Constraint selecting an image half. Patient side depends on convention."""
    image_left = (side == "right") if RADIOLOGICAL_CONVENTION else (side == "left")
    return (1.0, 0.0, 0.0) if image_left else (-1.0, 0.0, 0.0)


def _build_atlas() -> tuple[Region, ...]:
    regions: list[Region] = []
    for side in ("left", "right"):
        h = _hemisphere(side)
        regions += [
            Region(
                key=f"frontal_{side}",
                name=f"{side.capitalize()} frontal lobe",
                side=side,
                lobe="frontal",
                description=(
                    "Anterior cortex and underlying white matter. Relatively "
                    "preserved until later stages in typical amnestic AD."
                ),
                constraints=(h, (0.0, 1.0, -0.30)),
            ),
            Region(
                key=f"temporal_{side}",
                name=f"{side.capitalize()} temporal region",
                side=side,
                lobe="temporal",
                description=(
                    "Lateral temporal territory. At this axial level it "
                    "approximates superior/middle temporal cortex; the "
                    "hippocampus itself lies inferior to this slice."
                ),
                constraints=(h, (0.0, -1.0, 0.30), (0.0, 1.0, 0.45)),
                ring=(0.62, 1.15),
            ),
            Region(
                key=f"parietal_{side}",
                name=f"{side.capitalize()} parietal lobe",
                side=side,
                lobe="parietal",
                description=(
                    "Posterior superior cortex. Precuneus and posterior "
                    "cingulate hypometabolism is an early AD signature."
                ),
                constraints=(h, (0.0, -1.0, -0.12), (0.0, 1.0, 0.62)),
            ),
            Region(
                key=f"occipital_{side}",
                name=f"{side.capitalize()} occipital lobe",
                side=side,
                lobe="occipital",
                description=(
                    "Posterior visual cortex. Usually spared in typical AD; "
                    "prominent involvement suggests posterior cortical atrophy."
                ),
                constraints=(h, (0.0, -1.0, -0.60)),
            ),
            Region(
                key=f"insular_{side}",
                name=f"{side.capitalize()} insular / peri-sylvian region",
                side=side,
                lobe="insular",
                description=(
                    "Deep cortex around the sylvian fissure, bordering the "
                    "temporal operculum."
                ),
                constraints=(h, (0.0, -1.0, 0.28), (0.0, 1.0, 0.20)),
                ring=(0.40, 0.66),
            ),
            Region(
                key=f"periventricular_{side}",
                name=f"{side.capitalize()} periventricular white matter",
                side=side,
                lobe="white matter",
                description=(
                    "Deep white matter bordering the lateral ventricle. "
                    "Signal change here indicates small-vessel disease, which "
                    "frequently coexists with AD pathology."
                ),
                constraints=(h, (0.0, -1.0, 0.42), (0.0, 1.0, 0.42)),
                ring=(0.16, 0.48),
            ),
            Region(
                key=f"basal_ganglia_{side}",
                name=f"{side.capitalize()} thalamus / basal ganglia",
                side=side,
                lobe="deep grey",
                description=(
                    "Deep grey nuclei flanking the third ventricle — caudate, "
                    "lentiform nucleus and thalamus."
                ),
                constraints=(h, (0.0, -1.0, 0.22), (0.0, 1.0, 0.30)),
                ring=(0.10, 0.42),
            ),
        ]

    regions += [
        Region(
            key="ventricles",
            name="Lateral ventricles",
            side="midline",
            lobe="ventricular system",
            description=(
                "CSF-filled lateral ventricles. Enlargement is the most "
                "reproducible structural correlate of global brain volume loss "
                "on a single axial slice."
            ),
            constraints=(),
            ellipse=(0.0, 0.0, 0.46, 0.62),
        ),
        Region(
            key="corpus_callosum_genu",
            name="Corpus callosum (genu)",
            side="midline",
            lobe="commissural",
            description=(
                "Anterior commissural fibres joining the frontal lobes."
            ),
            constraints=((0.0, 1.0, -0.22), (0.0, -1.0, 0.52)),
            ellipse=(0.0, -0.36, 0.26, 0.18),
        ),
        Region(
            key="corpus_callosum_splenium",
            name="Corpus callosum (splenium)",
            side="midline",
            lobe="commissural",
            description=(
                "Posterior commissural fibres; thinning correlates with "
                "posterior cortical and hippocampal degeneration."
            ),
            constraints=((0.0, -1.0, -0.28),),
            ellipse=(0.0, 0.40, 0.26, 0.18),
        ),
    ]
    return tuple(regions)


ATLAS: tuple[Region, ...] = _build_atlas()
ATLAS_BY_KEY: dict[str, Region] = {r.key: r for r in ATLAS}


# ---------------------------------------------------------------------------
# Segmentation primitives (numpy only — no scipy, to keep the bundle small)
# ---------------------------------------------------------------------------
def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=64, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    p = hist / total
    centres = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return float(centres[int(np.argmax(sigma_b))])


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    labels, n = ndi.label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def brain_mask(gray: np.ndarray) -> np.ndarray:
    """Tissue mask: everything above background, largest blob, holes filled.

    Enclosed dark structures (the ventricles) become part of the mask via
    fill_holes; sulcal CSF that still connects to the background does not.
    Use `intracranial_mask` for the denominator of any area fraction.
    """
    thr = max(0.06, _otsu(gray.ravel()) * 0.55)
    m = gray > thr
    m = ndi.binary_opening(m, structure=_disk(2))
    m = _largest_component(m)
    return ndi.binary_fill_holes(m)


def intracranial_mask(tissue: np.ndarray, close_radius: int = 7) -> np.ndarray:
    """Approximate the intracranial boundary, sulcal CSF included.

    Widened sulci reach the brain surface, so they stay connected to the
    background and survive fill_holes as exterior. Measuring CSF against a
    tissue-only mask therefore *excludes* the very atrophy signal we are after
    — and makes CSF fraction fall as disease worsens, which is backwards. A
    morphological closing bridges the sulcal openings so the denominator is
    intracranial area rather than tissue area.
    """
    m = ndi.binary_closing(tissue, structure=_disk(close_radius))
    return ndi.binary_fill_holes(m)


# ---------------------------------------------------------------------------
# Pose normalisation (similarity registration to template space)
# ---------------------------------------------------------------------------
@dataclass
class Pose:
    centroid: tuple[float, float]      # (cy, cx) in input pixels
    angle: float                       # radians, principal axis vs vertical
    scale_y: float                     # pixels per normalised unit
    scale_x: float
    area_px: int
    shape: tuple[int, int] = field(default=(0, 0))


def estimate_pose(mask: np.ndarray) -> Pose:
    ys, xs = np.nonzero(mask)
    if len(ys) < 10:
        h, w = mask.shape
        return Pose((h / 2, w / 2), 0.0, h / 2, w / 2, 0, mask.shape)

    cy, cx = ys.mean(), xs.mean()
    y0, x0 = ys - cy, xs - cx
    cov = np.cov(np.stack([y0, x0]))
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    # Angle of the long (anterior-posterior) axis away from image vertical.
    angle = float(np.arctan2(major[1], major[0]))
    if angle > np.pi / 2:
        angle -= np.pi
    elif angle < -np.pi / 2:
        angle += np.pi

    c, s = np.cos(-angle), np.sin(-angle)
    ry = c * y0 - s * x0
    rx = s * y0 + c * x0
    # 2.2 sigma approximates the visible extent of a filled ellipse.
    scale_y = max(1.0, float(ry.std()) * 2.2)
    scale_x = max(1.0, float(rx.std()) * 2.2)
    return Pose((float(cy), float(cx)), angle, scale_y, scale_x, int(mask.sum()),
                mask.shape)


def to_template(img: np.ndarray, pose: Pose, size: int = TEMPLATE_SIZE,
                order_nearest: bool = False) -> np.ndarray:
    """Resample an input-space array into normalised template space."""
    lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    tx, ty = np.meshgrid(lin, lin)  # ty = anterior->posterior, tx = left->right

    ry = ty * pose.scale_y
    rx = tx * pose.scale_x
    c, s = np.cos(pose.angle), np.sin(pose.angle)
    sy = c * ry - s * rx + pose.centroid[0]
    sx = s * ry + c * rx + pose.centroid[1]

    h, w = img.shape
    if order_nearest:
        yi = np.clip(np.round(sy).astype(np.int32), 0, h - 1)
        xi = np.clip(np.round(sx).astype(np.int32), 0, w - 1)
        return img[yi, xi]

    y0 = np.clip(np.floor(sy).astype(np.int32), 0, h - 1)
    x0 = np.clip(np.floor(sx).astype(np.int32), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = np.clip(sy - y0, 0, 1)
    wx = np.clip(sx - x0, 0, 1)
    top = img[y0, x0] * (1 - wx) + img[y0, x1] * wx
    bot = img[y1, x0] * (1 - wx) + img[y1, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


# ---------------------------------------------------------------------------
# Atlas rasterisation
# ---------------------------------------------------------------------------
_ATLAS_CACHE: dict[int, dict[str, np.ndarray]] = {}


def atlas_masks(size: int = TEMPLATE_SIZE) -> dict[str, np.ndarray]:
    if size in _ATLAS_CACHE:
        return _ATLAS_CACHE[size]

    lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(lin, lin)
    r = np.sqrt(x ** 2 + y ** 2)

    out: dict[str, np.ndarray] = {}
    for region in ATLAS:
        m = np.ones((size, size), dtype=bool)
        for ax, ay, b in region.constraints:
            m &= (ax * x + ay * y) <= b
        if region.ellipse is not None:
            ex, ey, rx, ry = region.ellipse
            m &= (((x - ex) / rx) ** 2 + ((y - ey) / ry) ** 2) <= 1.0
        if region.ring is not None:
            lo, hi = region.ring
            m &= (r >= lo) & (r <= hi)
        m &= r <= 1.10
        out[region.key] = m

    _ATLAS_CACHE[size] = out
    return out


# ---------------------------------------------------------------------------
# Tissue segmentation in template space
# ---------------------------------------------------------------------------
def segment_tissue(t_img: np.ndarray, t_icv: np.ndarray) -> dict[str, np.ndarray]:
    """Split intracranial contents into CSF / grey / white compartments.

    T1 intensity ordering is CSF < grey matter < white matter. Both thresholds
    come from Otsu applied to the intracranial pixels only, so they adapt to
    each scan's windowing instead of assuming a fixed intensity scale.

    `t_icv` must be the intracranial mask (see `intracranial_mask`), not the
    tissue mask, or sulcal CSF is silently excluded.
    """
    inside = t_img[t_icv]
    if inside.size < 32:
        empty = np.zeros_like(t_icv)
        return {"csf": empty, "grey": empty, "white": empty,
                "ventricle": empty, "sulcal_csf": empty}

    # Recursive Otsu, splitting downward. The first threshold separates the
    # dark classes (CSF + grey) from white matter, because WM/everything-else
    # is the dominant contrast in T1. The CSF/grey boundary must then come from
    # a second Otsu on the DARK side — re-thresholding the bright side instead
    # leaves grey matter pooled with CSF and inflates CSF fraction to ~40%.
    wm_thr = _otsu(inside)
    dark = inside[inside < wm_thr]
    csf_thr = _otsu(dark) if dark.size > 32 else wm_thr * 0.5

    csf = t_icv & (t_img < csf_thr)
    grey = t_icv & (t_img >= csf_thr) & (t_img < wm_thr)
    white = t_icv & (t_img >= wm_thr)

    # Ventricles: CSF components whose centre of mass sits centrally. Taking
    # only the single largest component loses the contralateral horn whenever
    # the two ventricles are not connected in-plane.
    size = t_img.shape[0]
    lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(lin, lin)
    r = np.sqrt(x ** 2 + y ** 2)

    ventricle = np.zeros_like(csf)
    candidate = csf & (r < 0.62)
    if candidate.any():
        labels, n = ndi.label(candidate)
        if n:
            areas = ndi.sum(candidate, labels, index=range(1, n + 1))
            biggest = float(areas.max())
            centres = ndi.center_of_mass(candidate, labels, range(1, n + 1))
            for i, (cy, cx) in enumerate(centres, start=1):
                ny = (cy / (size - 1)) * 2 - 1
                nx = (cx / (size - 1)) * 2 - 1
                central = np.hypot(nx, ny) < 0.45
                substantial = areas[i - 1] >= max(20.0, 0.10 * biggest)
                if central and substantial:
                    ventricle |= labels == i

    sulcal = csf & ~ventricle
    return {"csf": csf, "grey": grey, "white": white,
            "ventricle": ventricle, "sulcal_csf": sulcal}


# ---------------------------------------------------------------------------
# Morphometry
# ---------------------------------------------------------------------------
METRIC_INFO: dict[str, dict[str, str]] = {
    "ventricle_brain_ratio": {
        "label": "Ventricle-to-brain ratio",
        "unit": "fraction",
        "direction": "high",
        "meaning": "Lateral ventricle area divided by intracranial area on this "
                   "slice. Rises as surrounding tissue is lost.",
    },
    "csf_fraction": {
        "label": "Total CSF fraction",
        "unit": "fraction",
        "direction": "high",
        "meaning": "All CSF-intensity area inside the brain: ventricles plus "
                   "widened sulci.",
    },
    "sulcal_csf_fraction": {
        "label": "Sulcal CSF fraction",
        "unit": "fraction",
        "direction": "high",
        "meaning": "Peripheral CSF only — a proxy for cortical sulcal widening.",
    },
    "parenchymal_fraction": {
        "label": "Brain parenchymal fraction",
        "unit": "fraction",
        "direction": "low",
        "meaning": "Proportion of the intracranial area that is still tissue.",
    },
    "ventricle_asymmetry": {
        "label": "Ventricular asymmetry",
        "unit": "index",
        "direction": "high",
        "meaning": "|left - right| / (left + right) ventricular area. Marked "
                   "asymmetry is atypical for symmetric degenerative disease.",
    },
    "cortical_rim_fraction": {
        "label": "Cortical rim fraction",
        "unit": "fraction",
        "direction": "low",
        "meaning": "Grey-matter area in the outer shell of the slice — a coarse "
                   "surrogate for cortical thickness.",
    },
    "grey_white_ratio": {
        "label": "Grey/white ratio",
        "unit": "ratio",
        "direction": "low",
        "meaning": "Ratio of grey- to white-intensity area within the brain.",
    },
}


def compute_morphometry(t_img: np.ndarray, t_icv: np.ndarray,
                        tissue: dict[str, np.ndarray]) -> dict[str, float]:
    """All fractions are normalised by intracranial area, which cancels
    head-size differences between subjects."""
    icv_area = float(t_icv.sum())
    if icv_area < 100:
        return {k: float("nan") for k in METRIC_INFO}

    size = t_img.shape[0]
    lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(lin, lin)
    r = np.sqrt(x ** 2 + y ** 2)

    vent = tissue["ventricle"]
    vent_area = float(vent.sum())
    a, b = float((vent & (x < 0)).sum()), float((vent & (x > 0)).sum())
    asym = abs(a - b) / (a + b) if (a + b) > 0 else 0.0

    rim = t_icv & (r > 0.72)
    rim_grey = tissue["grey"] & rim

    grey_area = float(tissue["grey"].sum())
    white_area = float(tissue["white"].sum())

    return {
        "ventricle_brain_ratio": vent_area / icv_area,
        "csf_fraction": float(tissue["csf"].sum()) / icv_area,
        "sulcal_csf_fraction": float(tissue["sulcal_csf"].sum()) / icv_area,
        "parenchymal_fraction": (grey_area + white_area) / icv_area,
        "ventricle_asymmetry": asym,
        "cortical_rim_fraction": float(rim_grey.sum()) / max(1.0, float(rim.sum())),
        "grey_white_ratio": grey_area / max(1.0, white_area),
    }


# ---------------------------------------------------------------------------
# View / level checks
# ---------------------------------------------------------------------------
def axial_view_check(t_img: np.ndarray, t_brain: np.ndarray,
                     tissue: dict[str, np.ndarray]) -> dict[str, Any]:
    """Is this consistent with an axial slice, and at what level?

    NOT a three-plane classifier — see the module docstring. It tests the
    properties an axial slice at the ventricular level must have, and reports
    which ones fail.
    """
    flipped = t_img[:, ::-1]
    denom = float(np.linalg.norm(t_img) * np.linalg.norm(flipped))
    symmetry = float((t_img * flipped).sum() / denom) if denom > 0 else 0.0

    ys, xs = np.nonzero(t_brain)
    if len(ys) < 50:
        elongation = 1.0
    else:
        elongation = float((ys.max() - ys.min() + 1) / max(1, xs.max() - xs.min() + 1))

    vbr = float(tissue["ventricle"].sum()) / max(1.0, float(t_brain.sum()))

    checks = {
        "bilaterally_symmetric": symmetry > 0.90,
        "anteroposterior_elongation": 0.95 <= elongation <= 1.55,
        "central_ventricles_visible": vbr > 0.004,
    }

    if vbr < 0.008:
        level = "high (supraventricular — above or at the roof of the ventricles)"
    elif vbr < 0.055:
        level = "mid (body of the lateral ventricles / thalamic level)"
    else:
        level = "mid-low (ventricular level with prominent CSF spaces)"

    return {
        "plane": "axial",
        "plane_confidence": round(float(np.mean(list(checks.values()))), 3),
        "consistent_with_axial": sum(checks.values()) >= 2,
        "estimated_level": level,
        "checks": checks,
        "signals": {
            "mirror_symmetry": round(symmetry, 4),
            "ap_elongation": round(elongation, 4),
            "ventricle_fraction": round(vbr, 5),
        },
        "limitation": (
            "Plane is asserted, not classified: the training corpus contains "
            "axial slices only, so no coronal/sagittal discrimination is "
            "possible. These checks verify consistency with an axial slice."
        ),
    }


# ---------------------------------------------------------------------------
# Cohort norms
# ---------------------------------------------------------------------------
_NORMS: dict[str, Any] | None = None


def get_norms() -> dict[str, Any] | None:
    global _NORMS
    if _NORMS is None and NORMS_PATH.exists():
        _NORMS = json.loads(NORMS_PATH.read_text(encoding="utf-8"))
    return _NORMS


def z_scores(metrics: dict[str, float],
             reference: str = "NonDemented") -> dict[str, dict[str, float]]:
    """Z-score each metric against the healthy reference cohort."""
    norms = get_norms()
    if not norms:
        return {}
    ref = norms.get("by_class", {}).get(reference)
    if not ref:
        return {}

    out: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        stats = ref.get(key)
        if not stats or not np.isfinite(value):
            continue
        sd = float(stats.get("std", 0.0)) or 1e-6
        z = (value - float(stats["mean"])) / sd
        out[key] = {
            "value": float(value),
            "reference_mean": float(stats["mean"]),
            "reference_sd": float(stats["std"]),
            "z": float(z),
            "percentile": float(_normal_cdf(z) * 100.0),
        }
    return out


def _normal_cdf(z: float) -> float:
    # Abramowitz & Stegun 7.1.26 applied to erf.
    t = 1.0 / (1.0 + 0.3275911 * abs(z) / np.sqrt(2))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
                t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-(z ** 2) / 2)
    erf = erf if z >= 0 else -erf
    return float(0.5 * (1.0 + erf))


# ---------------------------------------------------------------------------
# CAM attribution
# ---------------------------------------------------------------------------
def attribute_cam(t_cam: np.ndarray, t_brain: np.ndarray) -> list[dict[str, Any]]:
    """Fraction of total class-activation mass falling inside each region.

    Regions overlap by construction (a ventricle voxel is also periventricular
    territory), so shares are reported per region against the total and do not
    sum to 100%.
    """
    cam = np.maximum(t_cam, 0.0) * t_brain
    total = float(cam.sum())
    if total <= 1e-9:
        return []

    masks = atlas_masks(t_cam.shape[0])
    rows: list[dict[str, Any]] = []
    for region in ATLAS:
        m = masks[region.key] & t_brain
        area = float(m.sum())
        if area < 20:
            continue
        mass = float(cam[m].sum())
        rows.append({
            "key": region.key,
            "name": region.name,
            "side": region.side,
            "lobe": region.lobe,
            "description": region.description,
            "attention_share": mass / total,
            # Mass per unit area: corrects for big regions winning by size alone.
            "attention_density": (mass / area) / (total / float(t_brain.sum())),
        })

    rows.sort(key=lambda r: r["attention_density"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def analyse(image: Image.Image, cam: np.ndarray | None = None,
            size: int = TEMPLATE_SIZE) -> dict[str, Any]:
    """Full anatomical analysis of one axial slice.

    `cam` is the raw (7x7-ish) class activation map for the predicted class.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    tissue_mask = brain_mask(gray)
    icv = intracranial_mask(tissue_mask)
    pose = estimate_pose(icv)

    t_img = to_template(gray, pose, size)
    t_icv = to_template(icv.astype(np.float32), pose, size) > 0.5
    tissue = segment_tissue(t_img, t_icv)

    metrics = compute_morphometry(t_img, t_icv, tissue)
    view = axial_view_check(t_img, t_icv, tissue)
    zs = z_scores(metrics)

    attribution: list[dict[str, Any]] = []
    if cam is not None and cam.size > 1:
        cam_input = _upsample(cam, gray.shape)
        t_cam = to_template(cam_input, pose, size)
        attribution = attribute_cam(t_cam, t_icv)

    return {
        "view": view,
        "metrics": metrics,
        "metric_info": METRIC_INFO,
        "z_scores": zs,
        "attribution": attribution,
        "pose": {
            "rotation_deg": round(float(np.degrees(pose.angle)), 2),
            "brain_area_px": pose.area_px,
        },
        "convention": ("radiological (image-left = patient right)"
                       if RADIOLOGICAL_CONVENTION
                       else "neurological (image-left = patient left)"),
        "atlas_note": (
            "Coarse parametric lobar atlas registered by similarity transform. "
            "Ventricular and CSF compartments are segmented per-scan; lobar "
            "boundaries are geometric priors, so region names denote "
            "approximate territories rather than exact parcellations."
        ),
    }


def _upsample(m: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    in_h, in_w = m.shape
    out_h, out_w = shape
    if in_h < 2 or in_w < 2:
        return np.full(shape, float(m.mean()), dtype=np.float32)
    ys = np.linspace(0, in_h - 1, out_h, dtype=np.float32)
    xs = np.linspace(0, in_w - 1, out_w, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32); y1 = np.minimum(y0 + 1, in_h - 1)
    x0 = np.floor(xs).astype(np.int32); x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    top = m[y0][:, x0] * (1 - wx) + m[y0][:, x1] * wx
    bot = m[y1][:, x0] * (1 - wx) + m[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)
