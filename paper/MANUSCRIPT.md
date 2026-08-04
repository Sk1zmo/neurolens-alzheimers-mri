# Deep Learning for Alzheimer's Stage Classification from Brain MRI: An Accessible Web-Deployed System and a Morphometric Control That Exposes Benchmark Leakage

**[Author One]¹, [Author Two]¹**

¹[Department], [Institution], [City]

*(Received: [Month Year] Revised: [Month Year] Accepted: [Month Year])*

**Corresponding Author**
[Name]. E-mail: [email]

---

## ABSTRACT

**Introduction and Aim:** Dementia affects 57 million people worldwide, with
Alzheimer's disease accounting for 60–70% of cases; an estimated 8.8 million
Indians aged over 60 are affected. Up to 90% of cases in low- and middle-income
countries remain undiagnosed, and India has approximately one radiologist per
100,000 people, 70–80% of whom practise in tier-1 cities. We aimed to build an
accessible, explainable system that classifies Alzheimer's stage from a single
axial brain MRI slice, and to test rigorously whether the very high accuracies
reported on the standard public benchmark reflect genuine disease staging.

**Materials and Methods:** An EfficientNet-B0 with a global-average-pool and
single-linear head was trained on the OASIS-derived Kaggle augmented
Alzheimer's MRI dataset (6,400 original and 33,984 augmented axial T1 slices;
four stages). Augmented images were assigned to their nearest original in
embedding space and discarded when that source fell in validation or test.
Confidence was calibrated by temperature scaling. Class activation maps were
folded into the exported ONNX graph as a 1×1 convolution, registered to a
template, and integrated over a lobar atlas. Seven morphometric indices were
segmented per scan. As a control, classical models were trained on those
indices alone using the identical split.

**Results:** On 1,280 held-out original slices the network reached 99.84%
accuracy (95% CI 99.61–100), macro-F1 0.9991, and quadratic weighted kappa
0.9986; calibration improved expected calibration error from 0.0503 to 0.0011.
All seven morphometric indices separated the four stages (one-way ANOVA, p from
6×10⁻⁵ to 3×10⁻⁹¹; η² up to 0.237). However, the best classical model built on
those same indices reached only macro-F1 0.5807 on the identical split, a gap
of 0.418. Silhouette coefficient was 0.94 in the network's logit space but
−0.05 in morphometric space, and validation macro-F1 saturated at exactly
1.000. Inference required 17 ms on CPU (490 full reports/min; US$0.0039 per
1,000 scans).

**Conclusion:** The system is accurate, calibrated, anatomically explainable
and deployable at negligible cost on commodity hardware. The morphometric
control, however, demonstrates that headline accuracy on this benchmark is
substantially inflated by subject-level leakage rather than reflecting disease
staging. We propose the morphometric baseline as a routine control for
imaging-classification studies.

**Keywords:** Alzheimer's disease, magnetic resonance imaging, deep learning,
class activation mapping, dataset leakage, morphometry.

---

## INTRODUCTION

Alzheimer's disease is a progressive neurodegenerative disorder and the leading
cause of dementia, characterised pathologically by amyloid plaques,
neurofibrillary tangles and progressive cortical and hippocampal atrophy.
Neurofibrillary pathology follows a stereotyped anatomical sequence, beginning
in the transentorhinal region before spreading to limbic and then neocortical
areas (1). Because this progression produces measurable structural change,
magnetic resonance imaging (MRI) has become the principal marker of
neurodegeneration in the current research framework for the disease (2).

The global burden is substantial and growing. The World Health Organization
estimates that 57 million people were living with dementia in 2021, with nearly
10 million new cases annually and Alzheimer's disease contributing 60–70% of
them; over 60% of those affected live in low- and middle-income countries (3).
In India, the first nationally representative estimate, derived from the
Longitudinal Aging Study in India, places dementia prevalence at 7.4% among
adults aged 60 and over — approximately 8.8 million people — with higher rates
among women and in rural areas, and marked variation between states (4).

Detection remains the bottleneck. Current diagnosis rests on clinical
assessment, neuropsychological testing, structural MRI, and where available
cerebrospinal fluid or positron emission tomography biomarkers. Each of the
confirmatory modalities is expensive, invasive or scarce. The World Alzheimer
Report 2021 found that approximately 75% of people with dementia worldwide have
never received a diagnosis, rising to as high as 90% in low- and middle-income
settings — over 41 million undiagnosed cases (5).

Three gaps sustain this. First, **interpretation is slow and expert-bound**: a
structural MRI study comprises hundreds of slices, and visual atrophy rating
scales such as the medial temporal atrophy score (6) require trained
radiological judgement and carry known inter-rater variability. Second,
**expertise is unevenly distributed**: India has roughly 20,000–22,000
radiologists for 1.4 billion people, about one per 100,000, of whom 70–80%
practise in tier-1 metropolitan centres; scanner access is similarly skewed,
with more MRI units within a five-kilometre radius of central Delhi than in the
entire state of Bihar (7). Third, **knowledge transfer between clinicians is
informal and lossy**: the reasoning behind a radiological impression is rarely
captured in a form that another clinician, or a trainee, can inspect and reuse.

Deep learning offers a route around the first two constraints, and a large
literature reports accuracies above 99% on public Alzheimer's MRI benchmarks.
Such figures warrant scepticism. The most widely used public dataset ships
original and heavily augmented images together, and the common protocol —
training on the augmented set and testing on the originals — evaluates the
model on data it has effectively already seen. Saliency maps used to justify
these models are frequently presented without any faithfulness test, despite
evidence that they may not localise the features a model actually uses (8).

We therefore set out with two aims: to build a calibrated, anatomically
explainable, low-cost and openly accessible classifier that could realistically
extend screening reach; and to subject the resulting accuracy to controls
strong enough to establish whether it reflects disease staging or dataset
artefact.

---

## MATERIALS AND METHODS

### Dataset

The Kaggle "augmented-alzheimer-mri-dataset" was used, comprising 6,400
original and 33,984 augmented axial T1-weighted brain MRI slices at the level
of the lateral ventricles, in four classes: NonDemented (3,200),
VeryMildDemented (2,240), MildDemented (896) and ModerateDemented (64). The
images derive from the Open Access Series of Imaging Studies (OASIS)
cross-sectional cohort (9), with class assignment corresponding to Clinical
Dementia Rating scores of 0, 0.5, 1 and 2 respectively.

### Leakage control

Because the augmented images are derived from the originals but carry
randomised filenames, no provenance mapping is available. A stratified
train/validation/test split (65/15/20) was drawn from the **original images
only**. Every original and augmented image was then embedded using a frozen
ImageNet-pretrained EfficientNet-B0, and each augmented image was assigned to
its single most similar original by cosine similarity. Augmented images whose
assigned source fell in validation or test were removed from training.

This procedure discarded 11,878 of 33,984 augmented images (35.0%), closely
matching the 35% of originals held out — the agreement between these two
independent quantities is the internal check that source assignment was
behaving correctly. The final training set contained 26,266 images (4,160
original, 22,106 augmented).

We additionally tested whether subject identity could be recovered. The class
counts factorise exactly as 100, 70, 28 and 2 subjects × 32 slices, matching
the OASIS CDR distribution, suggesting the export might consist of contiguous
per-subject blocks. This hypothesis was rejected: within-block and
across-boundary embedding similarity were indistinguishable (0.893 vs 0.889),
and the best-fitting block length differed by class (20, 49, 56, 16). Subject
identity is therefore irrecoverable from the redistributed data.

### Model and training

EfficientNet-B0 pretrained on ImageNet (10) was used with the classifier head
kept deliberately as global average pooling followed by a single linear layer.
Training proceeded in two stages: three epochs with the backbone frozen, then
22 epochs of full fine-tuning with discriminative learning rates (backbone
3×10⁻⁴, head 9×10⁻⁴) under a one-cycle schedule, AdamW, weight decay 10⁻⁴,
label smoothing 0.05, batch size 32. A class-weighted cross-entropy loss and a
balanced sampler were used because ModerateDemented constitutes approximately
1% of original images. Model selection used validation macro-F1 rather than
accuracy, for the same reason. Augmentation at training time was limited to
horizontal flips, mild affine transformation, and brightness/contrast jitter;
vertical flips and large rotations were excluded to preserve the canonical
orientation of the slice.

### Calibration

A single temperature parameter was fitted on the validation split by L-BFGS
minimisation of negative log-likelihood (11) and applied at inference.

### Explainability and anatomical localisation

Because the head is global-average-pool followed by one linear layer, the
classic class activation map is exact and requires no backward pass (12). It
was folded into the exported ONNX graph as a 1×1 convolution over the final
feature map.

Each slice was segmented into intracranial, cerebrospinal fluid (CSF),
grey-matter and white-matter compartments using recursive Otsu thresholding on
intracranial pixels, and registered to a normalised template by a similarity
transform recovered from the intracranial mask's centroid and principal axes. A
coarse parametric lobar atlas (frontal, temporal, parietal, occipital, insular,
periventricular white matter, deep grey nuclei, lateral ventricles, callosal
genu and splenium) was intersected with the registered activation map to
quantify attention per territory, reported as both share and area-normalised
density.

Seven morphometric indices were computed per slice: ventricle-to-brain ratio,
total CSF fraction, sulcal CSF fraction, brain parenchymal fraction,
ventricular asymmetry, cortical rim fraction and grey/white ratio. All
fractions were normalised by intracranial area to cancel head-size differences.

### Morphometric control experiment

To test whether the network's accuracy reflects measurable atrophy, four
classical classifiers — logistic regression, linear discriminant analysis,
random forest and histogram gradient boosting — were trained on the seven
morphometric indices alone, using the **identical** training and test rows as
the network.

### Validation experiments

Faithfulness of the activation maps was assessed by deletion and insertion
curves against random-order controls. Selective prediction was assessed by
risk-coverage curves under four uncertainty scores. Robustness was measured
under rotation, additive Gaussian noise, contrast scaling, Gaussian blur and
downsampling. Class separability was quantified by silhouette coefficient in
both logit and morphometric spaces.

### Deployment

The trained model was exported to ONNX (opset 17) after verifying agreement
with the PyTorch checkpoint on held-out slices (maximum probability deviation
1×10⁻⁵; no change in predicted class). Serving uses ONNX Runtime, NumPy, SciPy
and Pillow in a serverless function, without PyTorch. DICOM instances, zipped
DICOM series and NIfTI volumes are accepted; DICOM windowing applies the
modality lookup table before the value-of-interest lookup table, and
acquisition plane is derived from the ImageOrientationPatient direction
cosines. For volumes, each slice is scored for correspondence to the training
level and the prediction is averaged across the selected slice and its
neighbours.

### Statistical analysis

Confidence intervals were obtained by percentile bootstrap over test images
(2,000 resamples); Wilson intervals were used for small-sample recall.
Between-stage differences in morphometry were tested by one-way ANOVA with η²
effect size and Spearman correlation against ordinal severity. Ordinal
performance was summarised by quadratic weighted kappa and mean absolute stage
error.

---

## RESULTS

### Classification performance

On 1,280 held-out original slices, the network achieved the results in Table 1.

**Table 1. Held-out classification performance (n = 1,280; 2,000 bootstrap resamples).**

| Metric | Value | 95% CI |
| --- | --- | --- |
| Accuracy | 0.9984 | 0.9961–1.0000 |
| Balanced accuracy | 0.9989 | 0.9971–1.0000 |
| Macro F1 | 0.9991 | 0.9976–1.0000 |
| Cohen's kappa | 0.9974 | 0.9935–1.0000 |
| Quadratic weighted kappa | 0.9986 | — |
| Mean absolute stage error | 0.0016 | — |

Only two of 1,280 slices were misclassified, both onto an adjacent severity
stage; no error spanned more than one stage.

### Calibration

Temperature scaling (T = 0.426) reduced expected calibration error from 0.0503
to 0.0011, with a Brier score of 0.0028 (Figure 5).

### Morphometry separates the stages

All seven indices differed significantly across the four stages, in the
directions predicted by the pathophysiology — ventricular and CSF fractions
rising with severity, parenchymal and cortical-rim fractions falling (Table 2,
Figure 6).

**Table 2. Morphometry by stage (n = 1,564 original slices).**

| Index | Non | VeryMild | Mild | Moderate | η² | ρ | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ventricle-to-brain ratio | 0.062 | 0.075 | 0.088 | 0.090 | 0.093 | +0.324 | 1×10⁻³² |
| Total CSF fraction | 0.169 | 0.186 | 0.201 | 0.211 | 0.201 | +0.443 | 1×10⁻⁷⁵ |
| Sulcal CSF fraction | 0.108 | 0.111 | 0.113 | 0.121 | 0.014 | +0.102 | 6×10⁻⁵ |
| Parenchymal fraction | 0.831 | 0.815 | 0.799 | 0.789 | 0.201 | −0.443 | 1×10⁻⁷⁵ |
| Ventricular asymmetry | 0.090 | 0.092 | 0.114 | 0.114 | 0.017 | +0.128 | 7×10⁻⁶ |
| Cortical rim fraction | 0.383 | 0.344 | 0.319 | 0.320 | 0.212 | −0.437 | 3×10⁻⁸⁰ |
| Grey/white ratio | 0.477 | 0.408 | 0.379 | 0.384 | 0.237 | −0.450 | 3×10⁻⁹¹ |

### The morphometric control

Despite those highly significant differences, classical models trained on the
same indices and evaluated on the identical split performed far below the
network (Table 3).

**Table 3. Morphometry-only control versus the network, identical split
(n_train = 1,015; n_test = 313).**

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Logistic regression | 0.3674 | 0.3416 |
| Linear discriminant analysis | 0.4569 | 0.3365 |
| Random forest | 0.5815 | 0.5574 |
| Histogram gradient boosting | 0.5687 | 0.5807 |
| **EfficientNet-B0 (this work)** | **0.9984** | **0.9991** |

The gap is 0.418 macro-F1. Silhouette coefficient was 0.94 in the network's
logit space but −0.05 in morphometric space, indicating that the morphometric
features form essentially no cluster structure by stage even though their
group means differ. A learning curve on the morphometric features was flat from
n = 58 to n = 1,173 (macro-F1 0.360 to 0.374), showing the ceiling is a
property of the features, not of sample size. Validation macro-F1 during
training saturated at exactly 1.000 from epoch 16 onwards (Figure 2).

### Explainability

Activation maps concentrated on the lateral ventricles and periventricular
territories. Insertion curves confirmed faithfulness (area under curve 0.841
for CAM-ordered versus 0.482 for random-ordered revelation). Deletion curves
did not (0.632 versus 0.476); we attribute this to the CAM concentrating on the
ventricles, which are intensity-homogeneous, so that blurring them removes
little information irrespective of importance.

### Robustness and deployment

Accuracy was preserved under rotation to ±15° and across contrast scaling, but
degraded under additive noise (0.620 at σ = 0.10; 0.360 at σ = 0.25) and blur
(0.183 at 5 px radius). Inference required a median of 17 ms for
classification and 119 ms for the full anatomical report on CPU, corresponding
to 490 reports per minute and approximately US$0.0039 per 1,000 scans on
commodity serverless infrastructure, with no GPU required.

---

## DISCUSSION

Two findings matter, and they point in different directions.

**The system works as an engineering artefact.** It is calibrated, so a stated
confidence means what it says; it is explainable in named anatomy rather than
by a heatmap alone; it runs on CPU in 17 ms at a cost of fractions of a cent
per thousand scans; and it is reachable from any browser without installation
or specialist hardware. Against a backdrop of one radiologist per 100,000
people concentrated in metropolitan centres (7), and up to 90% of dementia
undiagnosed in low- and middle-income settings (5), the cost and access profile
is the substantive contribution. A tool that runs anywhere with a network
connection and returns a structured, referenced report addresses the
distribution problem directly — not by replacing radiological judgement, but by
making a first-pass structured read available where no radiologist practises.
The same structured report also makes reasoning transferable: the measurements,
their reference ranges and the literature behind them are explicit, which is
what informal clinician-to-clinician knowledge transfer currently is not.

**The headline accuracy, however, should not be believed as a measure of
disease staging.** Our morphometric control is the reason. The seven indices we
measure are the established structural correlates of the disease, and they do
differ across stages with very large effect sizes. Yet every classical model
built on them plateaus near macro-F1 0.58 on the identical split, while the
network reports 0.9991. A network genuinely reading atrophy should not
outperform the combined measurable atrophy signal by 42 points. Three further
observations agree: silhouette in morphometric space is approximately zero, so
the stages do not form separable clusters in the space of measurable anatomy;
the morphometric learning curve is flat, so the shortfall is not a sample-size
artefact; and validation accuracy saturates at exactly 1.000, which a genuinely
held-out set should not.

The most parsimonious explanation is that the network is substantially
recognising individual brains rather than disease stage. The dataset provides
approximately 32 slices per subject, and subject identity is destroyed and — as
we showed — irrecoverable, so no partition of these files can separate
subjects. Adjacent slices of one brain therefore appear on both sides of any
split, and a network with sufficient capacity can exploit that.

This has a constructive consequence. **We propose the morphometry-only baseline
as a routine control** for medical-imaging classification studies. It is cheap,
uses no additional data, and provides an interpretable lower bound: when a deep
model vastly exceeds every domain-relevant measurement computed from the same
images, that gap is a signal to investigate leakage rather than a result to
report. Applied to the published literature on this benchmark, where accuracies
above 99% are routine, the control would be informative.

We note limitations beyond leakage. Analysis is confined to a single axial
level, so the hippocampus — the earliest and best-validated site of atrophy in
this disease (6) — lies below the imaged plane and is never measured directly.
The atlas is a coarse parametric parcellation registered affinely, not a
subject-specific segmentation, so region names denote approximate territories.
The ModerateDemented class contains only 64 original slices in the entire
dataset and 13 in our test split, so its per-class figures carry wide
uncertainty however favourable the point estimate. Deletion-based saliency
testing failed, and although we give a structural explanation, the maps should
be read as indicating where the model attended, not as evidence that the
highlighted tissue is abnormal (8). Finally, the system has never been
evaluated on an external cohort or on clinical data, and it is not a medical
device.

---

## CONCLUSION

We present a calibrated, anatomically explainable Alzheimer's stage classifier
that runs at 17 ms on CPU for fractions of a cent per thousand scans and is
accessible from any browser, together with a structured morphometric report
grounded in citable literature. On the standard public benchmark it reaches
99.84% accuracy. Using a morphometry-only control on the identical split, we
show that this figure is substantially inflated by subject-level leakage that
the dataset makes unavoidable, and we propose that control as routine practice.
The engineering contribution — accessibility, calibration, anatomical
explanation, negligible cost — stands independently of the benchmark number,
and is where the clinical value of such systems in resource-limited settings
actually lies.

---

## FUTURE DIRECTIONS

1. **Subject-level evaluation.** Rebuilding from the OASIS-1 source, which
   retains subject identifiers, would permit subject-wise grouped
   cross-validation and yield the first uninflated estimate for this
   architecture. Our loader for this is implemented and awaiting data access.
2. **External validation.** Evaluation on an independent cohort (ADNI, AIBL,
   MIRIAD) is the decisive test of generalisation.
3. **Volumetric analysis.** Extending from a single slice to the full volume
   would bring the hippocampus and medial temporal structures into range,
   enabling automated medial temporal atrophy scoring.
4. **Reader study.** A blinded interface has been implemented, reporting
   Cohen's kappa between reader, model and reference along with per-case
   reading time; a clinician study would quantify the time saving this work
   motivates but does not yet demonstrate.
5. **Prospective and equity evaluation.** Subgroup analysis by age, sex,
   education and scanner, and prospective evaluation in a
   resource-limited setting, are required before any deployment claim.

---

## ACKNOWLEDGEMENT

The authors thank the Open Access Series of Imaging Studies for making the
source imaging data publicly available.

## CONFLICT OF INTEREST

The authors declare no conflict of interest.

## DATA AND CODE AVAILABILITY

All code, trained weights, figures, tables and raw measurements are openly
available at https://github.com/Sk1zmo/neurolens-alzheimers-mri and the running
system at https://neurolens-opal.vercel.app.

---

## REFERENCES

1. Braak H, Braak E. Neuropathological stageing of Alzheimer-related changes.
   *Acta Neuropathol.* 1991;82(4):239–59.
2. Jack CR Jr, Bennett DA, Blennow K, et al. NIA-AA Research Framework: toward
   a biological definition of Alzheimer's disease. *Alzheimers Dement.*
   2018;14(4):535–62.
3. World Health Organization. Dementia — Fact sheet. Geneva: WHO; updated 31
   March 2025. Available from: https://www.who.int/news-room/fact-sheets/detail/dementia
4. Lee J, Meijer E, Langa KM, et al. Prevalence of dementia in India: national
   and state estimates from a nationwide study. *Alzheimers Dement.*
   2023;19(7):2898–912.
5. Gauthier S, Rosa-Neto P, Morais JA, Webster C. World Alzheimer Report 2021:
   Journey through the diagnosis of dementia. London: Alzheimer's Disease
   International; 2021.
6. Scheltens P, Leys D, Barkhof F, et al. Atrophy of medial temporal lobes on
   MRI in "probable" Alzheimer's disease and normal ageing: diagnostic value
   and neuropsychological correlates. *J Neurol Neurosurg Psychiatry.*
   1992;55(10):967–72.
7. Arora R. The training and practice of radiology in India: current trends.
   *Quant Imaging Med Surg.* 2014;4(6):449–50. [See also: radiologist density
   and distribution figures reported in the Indian radiology trade and
   professional literature, cited in the text.]
8. Arun N, Gaw N, Singh P, et al. Assessing the trustworthiness of saliency
   maps for localizing abnormalities in medical imaging. *Radiol Artif Intell.*
   2021;3(6):e200267.
9. Marcus DS, Wang TH, Parker J, Csernansky JG, Morris JC, Buckner RL. Open
   Access Series of Imaging Studies (OASIS): cross-sectional MRI data in young,
   middle aged, nondemented, and demented older adults. *J Cogn Neurosci.*
   2007;19(9):1498–507.
10. Tan M, Le QV. EfficientNet: rethinking model scaling for convolutional
    neural networks. *Proc ICML.* 2019;97:6105–14.
11. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural
    networks. *Proc ICML.* 2017;70:1321–30.
12. Zhou B, Khosla A, Lapedriza A, Oliva A, Torralba A. Learning deep features
    for discriminative localization. *Proc CVPR.* 2016:2921–9.
13. Frisoni GB, Fox NC, Jack CR Jr, Scheltens P, Thompson PM. The clinical use
    of structural MRI in Alzheimer disease. *Nat Rev Neurol.* 2010;6(2):67–77.
14. Nestor SM, Rupsingh R, Borrie M, et al. Ventricular enlargement as a
    possible measure of Alzheimer's disease progression validated using the
    Alzheimer's disease neuroimaging initiative database. *Brain.*
    2008;131(Pt 9):2443–54.
15. Wardlaw JM, Smith EE, Biessels GJ, et al. Neuroimaging standards for
    research into small vessel disease and its contribution to ageing and
    neurodegeneration. *Lancet Neurol.* 2013;12(8):822–38.

---

## FIGURE LEGENDS

**Figure 1.** Dataset composition and leakage control. Class imbalance in the
source data; nearest-source filtering removing 35.0% of augmented images,
matching the 35% of originals held out.

**Figure 2.** Training dynamics. Two-stage schedule. Validation macro-F1
saturates at exactly 1.000 from epoch 16 — read as a warning rather than a
result.

**Figure 3.** Confusion matrix on 1,280 held-out original slices, as counts and
row-normalised recall.

**Figure 4.** One-vs-rest ROC and precision-recall curves by stage.

**Figure 5.** Reliability diagrams before and after temperature scaling.

**Figure 6.** Morphometry across stages; one-way ANOVA with η² and Spearman ρ.

**Figure 7.** Class activation density by atlas region and true stage.

**Figure 8.** Per-class performance and headline metrics with bootstrap
confidence intervals.

**Figure 9.** Atlas-based localisation: registered slice, lobar parcellation,
and mean attention density in template space.

**Figure 10.** Qualitative examples across the severity range with activation
maps.

**Figure 11.** Selective prediction: accuracy against coverage as uncertain
cases are deferred.

**Figure 12.** Saliency faithfulness: deletion and insertion curves against
random controls.

**Figure 13.** Robustness to rotation, noise, contrast, blur and downsampling.

**Figure 14.** Ablation across training configurations on the shared test
split.

**Figure 15.** Subject recovery: a negative result. Block-length analysis
showing subject identity is not encoded in the export order.

**Supplementary Figure S1.** Segmentation quality control per stage.
