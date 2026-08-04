"""Rule-based radiological findings, keyed on measured values.

Every finding this module emits is triggered by a number computed in
`_anatomy.py` — a z-scored morphometric measurement, or the share of class
activation mass inside a labelled region. Nothing is generated from a language
model, because a fluent explanation that was never measured is exactly the
artefact that makes an imaging paper unfalsifiable.

Each rule carries:
  measurement    the observed number and how far it sits from the healthy cohort
  interpretation what that pattern is consistent with
  rationale      the mechanism, i.e. *why* this region and this change
  references     citations supporting the rationale, listed in REFERENCES

Language is deliberately hedged ("consistent with", "reported to"). These are
imaging correlates measured on a single 2-D slice, not diagnoses.
"""

from __future__ import annotations

from typing import Any

REFERENCES: dict[str, dict[str, str]] = {
    "braak1991": {
        "citation": "Braak H, Braak E. Neuropathological stageing of "
                    "Alzheimer-related changes. Acta Neuropathol. "
                    "1991;82(4):239-59.",
        "note": "Establishes the transentorhinal -> limbic -> neocortical "
                "progression of neurofibrillary pathology.",
    },
    "scheltens1992": {
        "citation": "Scheltens P, Leys D, Barkhof F, et al. Atrophy of medial "
                    "temporal lobes on MRI in 'probable' Alzheimer's disease "
                    "and normal ageing. J Neurol Neurosurg Psychiatry. "
                    "1992;55(10):967-72.",
        "note": "The MTA visual rating scale still used in clinical practice.",
    },
    "frisoni2010": {
        "citation": "Frisoni GB, Fox NC, Jack CR Jr, Scheltens P, Thompson PM. "
                    "The clinical use of structural MRI in Alzheimer disease. "
                    "Nat Rev Neurol. 2010;6(2):67-77.",
        "note": "Review of structural MRI markers and their diagnostic role.",
    },
    "jack2018": {
        "citation": "Jack CR Jr, Bennett DA, Blennow K, et al. NIA-AA Research "
                    "Framework: toward a biological definition of Alzheimer's "
                    "disease. Alzheimers Dement. 2018;14(4):535-62.",
        "note": "Places structural MRI as the 'N' (neurodegeneration) marker "
                "in the A/T/N scheme.",
    },
    "nestor2008": {
        "citation": "Nestor SM, Rupsingh R, Borrie M, et al. Ventricular "
                    "enlargement as a possible measure of Alzheimer's disease "
                    "progression validated using the ADNI database. Brain. "
                    "2008;131(Pt 9):2443-54.",
        "note": "Validates ventricular expansion as a progression marker.",
    },
    "fox2004": {
        "citation": "Fox NC, Schott JM. Imaging cerebral atrophy: normal "
                    "ageing to Alzheimer's disease. Lancet. "
                    "2004;363(9406):392-4.",
        "note": "Contrasts age-expected atrophy rates with those in AD.",
    },
    "wardlaw2013": {
        "citation": "Wardlaw JM, Smith EE, Biessels GJ, et al. Neuroimaging "
                    "standards for research into small vessel disease "
                    "(STRIVE). Lancet Neurol. 2013;12(8):822-38.",
        "note": "Reporting standards for white-matter and small-vessel change.",
    },
    "whitwell2007": {
        "citation": "Whitwell JL, Przybelski SA, Weigand SD, et al. 3D maps "
                    "from multiple MRI illustrate changing atrophy patterns as "
                    "subjects progress from MCI to AD. Brain. "
                    "2007;130(Pt 7):1777-86.",
        "note": "Maps the spatial progression of atrophy across clinical "
                "stages.",
    },
    "marcus2007": {
        "citation": "Marcus DS, Wang TH, Parker J, et al. Open Access Series "
                    "of Imaging Studies (OASIS): cross-sectional MRI data. "
                    "J Cogn Neurosci. 2007;19(9):1498-507.",
        "note": "Source cohort underlying the public dataset used here.",
    },
    "zhou2016": {
        "citation": "Zhou B, Khosla A, Lapedriza A, Oliva A, Torralba A. "
                    "Learning Deep Features for Discriminative Localization. "
                    "CVPR 2016:2921-9.",
        "note": "The class activation mapping formulation used for "
                "localisation here.",
    },
    "guo2017": {
        "citation": "Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of "
                    "Modern Neural Networks. ICML 2017:1321-30.",
        "note": "Temperature scaling, used to calibrate reported confidence.",
    },
    "liu2020": {
        "citation": "Liu W, Wang X, Owens J, Li Y. Energy-based "
                    "Out-of-distribution Detection. NeurIPS 2020.",
        "note": "The free-energy score used to flag out-of-distribution input.",
    },
    "tan2019": {
        "citation": "Tan M, Le QV. EfficientNet: Rethinking Model Scaling for "
                    "Convolutional Neural Networks. ICML 2019:6105-14.",
        "note": "Backbone architecture.",
    },
    "arun2021": {
        "citation": "Arun N, Gaw N, Singh P, et al. Assessing the "
                    "Trustworthiness of Saliency Maps for Localizing "
                    "Abnormalities in Medical Imaging. Radiol Artif Intell. "
                    "2021;3(6):e200267.",
        "note": "Cautions that saliency maps localise model attention, not "
                "pathology — the basis for this report's hedging.",
    },
}

# z-score bands for a metric whose abnormal direction is "high"
_BANDS = ((3.0, "marked"), (2.0, "notable"), (1.5, "borderline"))


def _severity(z: float, direction: str) -> str | None:
    signed = z if direction == "high" else -z
    for cut, label in _BANDS:
        if signed >= cut:
            return label
    return None


def _fmt(value: float, unit: str) -> str:
    if unit == "fraction":
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


# --------------------------------------------------------------------------
# Metric -> finding rules
# --------------------------------------------------------------------------
_METRIC_RULES: dict[str, dict[str, Any]] = {
    "ventricle_brain_ratio": {
        "region": "Lateral ventricles",
        "title": "Ventricular enlargement",
        "interpretation": (
            "Ventricular area is expanded relative to the healthy reference "
            "cohort, consistent with loss of surrounding periventricular "
            "tissue rather than a primary ventricular process."
        ),
        "rationale": (
            "The ventricles are a CSF-filled space bounded by brain tissue, so "
            "they enlarge passively as that tissue is lost (ex-vacuo "
            "dilatation). Because the boundary is high-contrast on T1, "
            "ventricular area is one of the most reproducible atrophy "
            "surrogates measurable from a single slice, and it tracks "
            "longitudinal progression in AD cohorts."
        ),
        "references": ["nestor2008", "frisoni2010", "fox2004"],
    },
    "sulcal_csf_fraction": {
        "region": "Cortical surface / sulci",
        "title": "Sulcal widening",
        "interpretation": (
            "CSF occupies more of the peripheral brain area than expected, "
            "consistent with cortical volume loss widening the sulci."
        ),
        "rationale": (
            "Cortical atrophy separates adjacent gyri, so the CSF-filled "
            "sulcal spaces between them broaden. On T1 that appears as an "
            "increase in dark, CSF-intensity area near the brain surface."
        ),
        "references": ["fox2004", "frisoni2010"],
    },
    "parenchymal_fraction": {
        "region": "Whole slice",
        "title": "Reduced brain parenchymal fraction",
        "interpretation": (
            "Less of the intracranial area is occupied by tissue than in the "
            "reference cohort — a global rather than focal pattern."
        ),
        "rationale": (
            "Parenchymal fraction normalises tissue area by intracranial area, "
            "which cancels head-size differences between subjects. It is the "
            "single-slice analogue of the whole-brain volume measures used as "
            "the 'N' neurodegeneration marker in the A/T/N framework."
        ),
        "references": ["jack2018", "frisoni2010"],
    },
    "cortical_rim_fraction": {
        "region": "Cortical ribbon",
        "title": "Thinned cortical ribbon",
        "interpretation": (
            "Grey-matter area in the outer shell of the slice is below the "
            "reference range, a coarse surrogate for cortical thinning."
        ),
        "rationale": (
            "Neuronal and synaptic loss thins the cortical ribbon. On a 2-D "
            "slice this shows as less grey-intensity tissue in the peripheral "
            "band, though partial-volume effects at this resolution make it a "
            "weaker measurement than surface-based cortical thickness."
        ),
        "references": ["whitwell2007", "frisoni2010"],
    },
    "ventricle_asymmetry": {
        "region": "Lateral ventricles",
        "title": "Asymmetric ventricular size",
        "interpretation": (
            "One lateral ventricle is substantially larger than the other. "
            "Degenerative disease is usually approximately symmetric, so "
            "marked asymmetry warrants considering a focal cause."
        ),
        "rationale": (
            "Alzheimer pathology spreads through anatomically connected "
            "networks bilaterally, producing broadly symmetric atrophy. "
            "Pronounced asymmetry more often reflects a focal lesion, prior "
            "infarct, developmental variation, or slice obliquity."
        ),
        "references": ["braak1991", "whitwell2007"],
    },
    "csf_fraction": {
        "region": "Intracranial CSF",
        "title": "Increased total CSF fraction",
        "interpretation": (
            "Combined ventricular and sulcal CSF is elevated, indicating "
            "generalised volume loss rather than an isolated compartment."
        ),
        "rationale": (
            "Total CSF fraction sums both compartments that expand ex vacuo. "
            "Elevation in both simultaneously argues for diffuse parenchymal "
            "loss instead of, say, obstructive hydrocephalus, which enlarges "
            "ventricles while effacing sulci."
        ),
        "references": ["fox2004", "frisoni2010"],
    },
}

# --------------------------------------------------------------------------
# Region -> mechanism, for CAM attribution
# --------------------------------------------------------------------------
_REGION_RATIONALE: dict[str, dict[str, Any]] = {
    "ventricles": {
        "rationale": (
            "Ventricular margins carry the strongest single-slice signal of "
            "global tissue loss, so a discriminative model concentrating here "
            "is behaving consistently with the established imaging marker."
        ),
        "references": ["nestor2008", "frisoni2010"],
    },
    "temporal": {
        "rationale": (
            "Medial temporal structures are affected earliest in the Braak "
            "sequence, and medial temporal atrophy is the best-validated "
            "single MRI marker for AD. Note that the hippocampus proper lies "
            "inferior to this axial level, so attention here reflects lateral "
            "temporal cortex and the temporal horn rather than hippocampus."
        ),
        "references": ["braak1991", "scheltens1992", "frisoni2010"],
    },
    "parietal": {
        "rationale": (
            "Posterior parietal cortex, especially precuneus and posterior "
            "cingulate, shows early metabolic and later structural change in "
            "AD, and is a region where atrophy separates AD from normal "
            "ageing."
        ),
        "references": ["whitwell2007", "jack2018"],
    },
    "frontal": {
        "rationale": (
            "Frontal cortex is relatively spared until later stages of typical "
            "amnestic AD, so prominent frontal weighting is more consistent "
            "with advanced disease or a non-amnestic presentation."
        ),
        "references": ["braak1991", "whitwell2007"],
    },
    "occipital": {
        "rationale": (
            "Occipital cortex is typically spared in amnestic AD. Prominent "
            "occipital involvement raises the possibility of posterior "
            "cortical atrophy or dementia with Lewy bodies."
        ),
        "references": ["whitwell2007"],
    },
    "white matter": {
        "rationale": (
            "Periventricular white-matter change usually reflects cerebral "
            "small-vessel disease, which commonly coexists with and "
            "compounds AD pathology rather than substituting for it."
        ),
        "references": ["wardlaw2013"],
    },
    "deep grey": {
        "rationale": (
            "Thalamic and striatal volume loss occurs in AD but later and less "
            "specifically than medial temporal change; it also accompanies "
            "vascular and mixed pathology."
        ),
        "references": ["whitwell2007", "wardlaw2013"],
    },
    "insular": {
        "rationale": (
            "Insular and peri-sylvian cortex is involved in the limbic phase "
            "of the Braak sequence, intermediate between medial temporal onset "
            "and neocortical spread."
        ),
        "references": ["braak1991"],
    },
    "commissural": {
        "rationale": (
            "Callosal thinning is secondary — it follows loss of the cortical "
            "neurons whose axons cross there, with the splenium reflecting "
            "posterior and temporal degeneration."
        ),
        "references": ["whitwell2007"],
    },
}


def _region_rationale(region: dict[str, Any]) -> dict[str, Any] | None:
    if region["key"] in _REGION_RATIONALE:
        return _REGION_RATIONALE[region["key"]]
    return _REGION_RATIONALE.get(region["lobe"])


# --------------------------------------------------------------------------
def generate(analysis: dict[str, Any], prediction: dict[str, Any] | None = None,
             max_regions: int = 4) -> dict[str, Any]:
    """Build the structured findings report."""
    findings: list[dict[str, Any]] = []
    used_refs: set[str] = set()

    # ---- morphometric findings -------------------------------------------
    info = analysis.get("metric_info", {})
    for key, stats in (analysis.get("z_scores") or {}).items():
        rule = _METRIC_RULES.get(key)
        meta = info.get(key)
        if not rule or not meta:
            continue
        severity = _severity(stats["z"], meta["direction"])
        if severity is None:
            continue

        direction = "above" if stats["z"] > 0 else "below"
        findings.append({
            "kind": "morphometry",
            "metric": key,
            "severity": severity,
            "region": rule["region"],
            "title": rule["title"],
            "measurement": (
                f"{meta['label']} measured at "
                f"{_fmt(stats['value'], meta['unit'])} "
                f"({abs(stats['z']):.1f} SD {direction} the non-demented "
                f"reference mean of "
                f"{_fmt(stats['reference_mean'], meta['unit'])})."
            ),
            "interpretation": rule["interpretation"],
            "rationale": rule["rationale"],
            "references": rule["references"],
            "z": stats["z"],
        })
        used_refs.update(rule["references"])

    findings.sort(key=lambda f: abs(f.get("z", 0.0)), reverse=True)

    # ---- attention findings ----------------------------------------------
    attention: list[dict[str, Any]] = []
    for region in (analysis.get("attribution") or [])[:max_regions]:
        if region["attention_density"] < 1.05:
            continue
        rationale = _region_rationale(region)
        if not rationale:
            continue
        attention.append({
            "kind": "attention",
            "region": region["name"],
            "side": region["side"],
            "lobe": region["lobe"],
            "attention_share": region["attention_share"],
            "attention_density": region["attention_density"],
            "measurement": (
                f"{region['attention_share'] * 100:.1f}% of the model's class "
                f"activation mass falls in this territory "
                f"({region['attention_density']:.1f}x the density expected if "
                f"attention were spread uniformly across the brain)."
            ),
            "anatomy": region["description"],
            "rationale": rationale["rationale"],
            "references": rationale["references"],
        })
        used_refs.update(rationale["references"])

    # ---- narrative summary ------------------------------------------------
    stage = (prediction or {}).get("label")
    if findings:
        lead = findings[0]
        summary = (
            f"Morphometry on this slice is abnormal in {len(findings)} of the "
            f"measured indices; the strongest deviation is "
            f"{lead['title'].lower()} ({lead['region']})."
        )
    else:
        summary = (
            "No measured morphometric index deviates more than 1.5 SD from the "
            "non-demented reference cohort on this slice."
        )
    if attention:
        summary += (
            f" The classifier weighted the {attention[0]['region'].lower()} "
            f"most densely."
        )
    if stage:
        summary += f" Predicted stage: {stage}."

    used_refs.update(["zhou2016", "arun2021", "marcus2007"])

    return {
        "summary": summary,
        "findings": findings,
        "attention": attention,
        "references": {k: REFERENCES[k] for k in sorted(used_refs)
                       if k in REFERENCES},
        "disclaimer": (
            "These are quantitative imaging correlates measured on a single "
            "2-D axial slice, compared against a small public reference "
            "cohort. They are not a radiology report and not a diagnosis. "
            "Class activation maps localise where a model attended, which is "
            "not evidence that the highlighted tissue is abnormal."
        ),
    }
