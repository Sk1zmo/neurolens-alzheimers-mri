# NeuroLens

**Live: [neurolens-opal.vercel.app](https://neurolens-opal.vercel.app)**

Four-stage Alzheimer's classification from a brain MRI slice, with a class
activation map, calibrated confidence, an anonymous scan history, and a
human-in-the-loop retraining pipeline. Deployed as a single Vercel project:
Next.js frontend plus a Python function serving an ONNX model.

Held-out test performance (1,280 original MRI slices, leak-filtered split):

| Metric | Value |
| --- | --- |
| Accuracy | 0.9984 |
| Balanced accuracy | 0.9989 |
| Macro F1 | 0.9991 |
| Macro AUC (OvR) | 1.0000 |
| Cohen's kappa | 0.9974 |
| ECE (raw → calibrated) | 0.0503 → 0.0011 |
| Median inference latency | 11 ms CPU |

Read [the caveat below](#what-makes-this-different-from-the-usual-notebook)
before quoting those numbers — this dataset has no subject identifiers, so they
overstate performance on a genuinely new patient.

> **Not a medical device.** Research and teaching use only. It has no regulatory
> clearance, has never been validated clinically, and must not inform a medical
> decision.

```text
alzheimer-app/
├── training/        PyTorch: split, train, evaluate, export, retrain
├── artifacts/       splits, checkpoints, metrics, plots  (git-ignored)
└── web/             Next.js app + Vercel Python inference function
    ├── api/         predict.py, _inference.py, model/model.onnx
    ├── src/         app router pages, components, lib
    └── supabase/    schema.sql
```

---

## What makes this different from the usual notebook

**The accuracy number is honest.** The standard recipe for this Kaggle dataset —
train on the 33,984 augmented images, test on the 6,400 originals — reports
~99% and means nothing, because the augmented images are *derived from* the
originals. The model is tested on what it already memorised.

`prepare_split.py` fixes this with nearest-source assignment:

1. Carve a stratified train/val/test split out of `OriginalDataset` only.
2. Embed every original and every augmented image with a frozen ImageNet
   EfficientNet-B0.
3. Assign each augmented image to its single most similar original.
4. Drop it from training if that source landed in val or test.

That removed **11,878 of 33,984** augmented images — 35.0%, almost exactly the
35% of originals held out. The match between those two numbers is the check that
the assignment worked.

**The caveat that survives.** The dataset has no subject identifiers. Slices from
one brain appear throughout and cannot be kept on one side of the split.
Real-world accuracy on a new patient is lower than the reported figure. That
warning is carried into the model card and the app's About page rather than
quietly dropped.

**Explanations without autograd.** The classifier head is
`global-average-pool → single Linear`, which makes the classic Class Activation
Map exact and gradient-free — it is a 1×1 convolution over the final feature
map. `export_onnx.py` folds it into the graph, so the serverless function gets a
real saliency map from one forward pass with no PyTorch anywhere.

**Calibrated confidence.** A fine-tuned CNN's raw softmax is over-confident, and
this app puts a percentage in front of a user. `evaluate.py` fits a temperature
on the validation split; the deployed model applies it at inference.

**Serving fits in a serverless function.** PyTorch alone blows past Vercel's
bundle limit. ONNX Runtime + NumPy + Pillow lands around 40 MB, and the CAM
colormap and bilinear upsample are written in NumPy rather than pulling in
60 MB of OpenCV for two calls.

---

## 1. Train

Requires Python 3.12 and, ideally, an NVIDIA GPU.

```bash
cd alzheimer-app
py -3.12 -m venv .venv
.venv\Scripts\activate                    # macOS/Linux: source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r training/requirements.txt
```

Point `DATA_ROOT` in `training/config.py` at your copy of the dataset, then:

```bash
cd training
python prepare_split.py     # leak-filtered split          (~10 min on a 4060)
python train.py             # two-stage fine-tune          (~45-90 min)
python evaluate.py          # held-out metrics + plots + temperature
python export_onnx.py       # ONNX -> web/api/model/, metrics -> web/public/model/
python make_samples.py      # demo images from the test split
```

On a laptop GPU that is also driving the desktop, lower the footprint:

```bash
python train.py --batch-size 32 --workers 2
```

`train.py` prints a per-batch tqdm bar. If you launched it with its output
piped somewhere (a log file, a supervising process), Python block-buffers
stdout and you will see nothing for a long time — attach a live view from
another terminal instead:

```bash
python watch_training.py
# [=========================---------] ~73.9%  ep ~18.5/25  elapsed 1h41m  eta ~36m  GPU 97%
```

It reconstructs progress from the process start time and checkpoint saves, so
it works against a run already in flight, and prints the final validation
numbers when the job exits.

`train.py` selects on validation **macro-F1**, not accuracy: `ModerateDemented`
is ~1% of the original data and plain accuracy would happily ignore it.

## 2. Supabase

1. Create a project at [supabase.com](https://supabase.com).
2. SQL Editor → paste `web/supabase/schema.sql` → Run. This creates the two
   tables, the private `scans` bucket, and the stats views.
3. Project Settings → API → copy the URL and the **service role** key.

Access model: there are no end-user accounts. RLS is enabled with *no*
permissive policies, so the anon key can read nothing. Every read and write goes
through a server-side route handler using the service-role key, scoped by the
caller's session id.

## 3. Run locally

```bash
cd web
cp .env.example .env.local        # fill in SUPABASE_* and REVIEW_TOKEN
npm install
```

`next dev` does not execute Vercel's Python runtime, so run the same handler
directly in a second terminal:

```bash
python ../training/serve_local.py     # http://127.0.0.1:8000/api/predict
```

and set in `.env.local`:

```text
NEXT_PUBLIC_PREDICT_URL=http://127.0.0.1:8000/api/predict
```

Then `npm run dev`.

## 4. Deploy to Vercel

```bash
cd web
vercel login
vercel link
vercel env add SUPABASE_URL production
vercel env add SUPABASE_SERVICE_ROLE_KEY production
vercel env add REVIEW_TOKEN production
vercel --prod
```

Leave `NEXT_PUBLIC_PREDICT_URL` unset in production — the relative
`/api/predict` path resolves to the Python function. `web/api/model/model.onnx`
must be committed for the function to have a model to load.

---

## The two-pass database

Every saved scan is written **twice**, to two tables with genuinely different
lifecycles:

| | `scans` | `training_queue` |
|---|---|---|
| Read by | the user, in `/history` | `training/retrain.py` |
| Written | on save | on save, if consent is given |
| Label | the model's prediction, frozen | `verified_label`, assigned by a human |
| Deleting from history | removes it | keeps it once reviewed |

They are separate rows rather than one row with flags because a user deleting
their history must not silently shrink the training corpus, and a reviewer
re-labelling a scan must not rewrite what the user was originally shown.

**A queue row never becomes training data on the model's own say-so.** It needs a
`verified_label` assigned by a human in `/review` (unlocked by `REVIEW_TOKEN`).
Training on your own predictions only amplifies what the model already believes.

### Retraining

```bash
cd training
python retrain.py --min-new 25
```

Downloads approved+labelled scans, merges them into the training manifest,
warm-starts from the current checkpoint, and re-evaluates **on the unchanged
held-out test split** so before/after is directly comparable. A new checkpoint is
promoted only if it beats the incumbent on macro-F1 — otherwise the incumbent
metrics are restored and nothing ships. Promotion re-exports the ONNX model and
marks the queue rows as used.

```bash
cd ../web && vercel --prod    # ship the new model
```

---

## Routes

| Route | What it does |
|---|---|
| `/` | Upload, classify, CAM overlay, calibrated confidence, PDF report |
| `/history` | This browser's saved scans; expand for probabilities; delete |
| `/model` | Model card: held-out metrics, confusion matrix, ROC, calibration |
| `/review` | Reviewer console — assign verified labels, approve/reject |
| `/about` | How it works, data handling, limitations |

| API | |
|---|---|
| `POST /api/predict` | Python function. Raw image bytes or `{"image": "<data-url>"}` |
| `GET /api/predict` | Model metadata; doubles as a warm-up probe |
| `GET/POST /api/scans` | List / save, scoped by `x-session-id` |
| `DELETE /api/scans/:id` | Delete, scoped by `x-session-id` |
| `GET/PATCH /api/review` | Queue and labelling, gated by `x-review-token` |
| `GET /api/stats` | Aggregate counts only — no scan contents, no session ids |

---

## Environment variables

| Name | Where | Purpose |
|---|---|---|
| `SUPABASE_URL` | server | Project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | server | Bypasses RLS. **Never** prefix `NEXT_PUBLIC_` |
| `REVIEW_TOKEN` | server | Unlocks `/review`. Unset ⇒ the console is disabled |
| `NEXT_PUBLIC_PREDICT_URL` | client | Local dev only; unset in production |

## Dataset

[uraninjo/augmented-alzheimer-mri-dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)
— axial T1 brain **MRI** (not CT). Classes: `NonDemented`, `VeryMildDemented`,
`MildDemented`, `ModerateDemented`. `ModerateDemented` has only 64 original
slices in the entire dataset, which is why its held-out metrics carry wide error
bars.
