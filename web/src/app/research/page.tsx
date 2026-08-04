import fs from "node:fs/promises";
import path from "node:path";
import Link from "next/link";

export const metadata = { title: "Research" };
export const revalidate = 3600;

interface Stats {
  n_test: number;
  n_bootstrap: number;
  headline: Record<string, { value: number; ci95_low: number; ci95_high: number }>;
  per_class: Record<
    string,
    {
      precision: number;
      recall: number;
      f1: number;
      support: number;
      auc: number;
      average_precision: number;
    }
  >;
  error_analysis: {
    total_errors: number;
    adjacent_stage_errors: number;
    adjacent_fraction: number;
  };
  morphometry: Record<
    string,
    {
      label: string;
      by_class: Record<string, { mean: number; sd: number; n: number }>;
      anova_F: number;
      anova_p: number;
      eta_squared: number;
      spearman_rho_vs_severity: number;
    }
  >;
}

const CLASS_DIRS = [
  "NonDemented",
  "VeryMildDemented",
  "MildDemented",
  "ModerateDemented",
];

const FIGURES = [
  {
    file: "fig01_dataset.png",
    title: "Dataset composition and leakage control",
    caption:
      "The source data is severely imbalanced — ModerateDemented has only 64 original slices. " +
      "Nearest-source filtering removed 35.0% of augmented images as descendants of held-out " +
      "originals, matching the 35% of originals held out. That agreement is the evidence the " +
      "source assignment worked.",
  },
  {
    file: "fig02_training.png",
    title: "Training dynamics",
    caption:
      "Two-stage schedule: the linear head is warmed up against a frozen backbone before the " +
      "whole network is fine-tuned with discriminative learning rates under a one-cycle " +
      "schedule. Selection is on validation macro-F1, not accuracy, because ModerateDemented " +
      "is about 1% of the original data. Read panel (c) as a warning rather than a result: " +
      "validation macro-F1 saturates at exactly 1.000 by epoch 16 and stays there. A held-out " +
      "set that is solved perfectly is not measuring generalisation — it is the subject-level " +
      "leakage of this dataset showing up in the training dynamics.",
  },
  {
    file: "fig03_confusion.png",
    title: "Confusion on the held-out test set",
    caption:
      "Counts and row-normalised recall over 1,280 original slices whose augmented derivatives " +
      "were excluded from training.",
  },
  {
    file: "fig04_roc_pr.png",
    title: "Discrimination by stage",
    caption:
      "One-vs-rest ROC and precision-recall. Precision-recall is the more honest view under " +
      "this much class imbalance; dotted lines mark each class's prevalence, the no-skill floor.",
  },
  {
    file: "fig05_calibration.png",
    title: "Confidence calibration",
    caption:
      "Reliability diagrams before and after temperature scaling fitted on the validation " +
      "split. Marker area is proportional to the number of test images in each confidence bin.",
  },
  {
    file: "fig06_morphometry.png",
    title: "Quantitative morphometry across stages",
    caption:
      "Seven indices measured by segmentation, with no involvement of the classifier. All " +
      "separate the stages under one-way ANOVA, and all move in the direction predicted by " +
      "the pathophysiology of Alzheimer's disease. This is independent evidence that the " +
      "labels correspond to measurable anatomy rather than a dataset artefact.",
  },
  {
    file: "fig07_attention.png",
    title: "Attention by atlas region and stage",
    caption:
      "Mean class-activation density per registered atlas region. A value of 1.0 is what " +
      "uniform attention across the brain would give, so values above 1 mark territories the " +
      "model weights more densely than chance.",
  },
  {
    file: "fig08_perclass.png",
    title: "Performance with uncertainty",
    caption:
      "Per-class precision, recall and F1, alongside headline metrics with percentile " +
      "bootstrap 95% confidence intervals over test images.",
  },
  {
    file: "fig09_atlas.png",
    title: "Atlas-based localisation",
    caption:
      "Each slice is registered to a normalised template by a similarity transform recovered " +
      "from the intracranial mask, then intersected with a coarse parametric lobar atlas. " +
      "Ventricular and CSF compartments are segmented per-scan; lobar boundaries are " +
      "geometric priors.",
  },
  {
    file: "fig10_qualitative.png",
    title: "Qualitative examples",
    caption:
      "One held-out example per stage with its class activation map, predicted label, " +
      "ventricle-to-brain ratio and most densely weighted region.",
  },
  {
    file: "fig11_uncertainty.png",
    title: "Selective prediction",
    caption:
      "Accuracy on the cases the system answers, as the least certain are deferred to a " +
      "clinician. On the clean split this is degenerate — the model makes too few errors " +
      "to rank, and that is reported rather than hidden. Under noise the mechanism works, " +
      "and the free-energy score is the best deferral signal.",
  },
  {
    file: "fig12_saliency.png",
    title: "Are the activation maps faithful?",
    caption:
      "Insertion passes decisively: revealing CAM-ranked pixels restores the prediction far " +
      "faster than revealing random ones. Deletion fails, and the likely reason is " +
      "structural — the CAM concentrates on the ventricles, which are homogeneous, so " +
      "blurring them removes little information however important they are. Reported as a " +
      "split verdict rather than resolved in the method's favour.",
  },
  {
    file: "fig13_robustness.png",
    title: "Robustness to acquisition variation",
    caption:
      "The model tolerates rotation and contrast shifts but degrades sharply under noise " +
      "and blur — accuracy falls below 40% at σ = 0.25 and below 20% at a 5 px blur. Any " +
      "deployment across scanners or protocols has to account for that.",
  },
  {
    file: "fig14_ablation.png",
    title: "Ablation",
    caption:
      "All variants share one shortened schedule and one held-out test split; only the " +
      "training manifest or the sampler changes. The gap between the shipped configuration " +
      "and the no-leak-filter row is the accuracy inflation the standard protocol for this " +
      "dataset buys for free.",
  },
  {
    file: "fig15_convergence.png",
    title: "Where the accuracy actually comes from",
    caption:
      "The paper's central argument in one panel. A CNN trained on real images only, and " +
      "classical models on segmented morphometry, agree at macro-F1 0.50–0.58 despite sharing " +
      "no machinery. Every point above that band arrives with augmented copies of the test " +
      "subjects. Removing the leak filter reaches exactly 1.000 on 1,280 unseen slices, which " +
      "no genuine evaluation of a hard clinical task should produce.",
  },
  {
    file: "fig11_subject_recovery.png",
    title: "Subject recovery — a negative result",
    caption:
      "The class counts are exactly 100/70/28/2 subjects × 32 slices, matching the OASIS-1 " +
      "CDR breakdown, which suggested the export might be contiguous per-subject blocks. It " +
      "is not: within-block and across-boundary similarity are identical, and the " +
      "best-fitting block length differs per class. Subject identity is genuinely " +
      "destroyed, so subject-level splitting requires the original OASIS-1 download.",
  },
  {
    file: "qc_segmentation.png",
    title: "Segmentation quality control",
    caption:
      "Per-stage QC of the pipeline every morphometric index depends on: registered slice, " +
      "intracranial versus tissue mask, ventricle and sulcal CSF compartments, grey/white " +
      "split, and atlas outlines. Area-based measurements fail silently when a mask is wrong, " +
      "so this panel is checked whenever the segmentation changes.",
  },
];

interface Experiments {
  throughput?: {
    classification_only: { median_ms: number; throughput_per_min: number };
    full_pipeline: { median_ms: number; throughput_per_min: number };
    cost_model: { usd_per_1000_scans: number; assumption: string };
    hardware: string;
  };
  saliency?: {
    auc_deletion_cam: number;
    auc_deletion_random: number;
    auc_insertion_cam: number;
    auc_insertion_random: number;
    deletion_passes: boolean;
    insertion_passes: boolean;
    interpretation: string;
  };
  uncertainty?: Record<
    string,
    { base_accuracy: number; n_errors: number; degenerate: boolean }
  >;
}

async function readExperiments(): Promise<Experiments | null> {
  try {
    const file = path.join(process.cwd(), "public", "paper", "experiments.json");
    return JSON.parse(await fs.readFile(file, "utf-8")) as Experiments;
  } catch {
    return null;
  }
}

async function readStats(): Promise<Stats | null> {
  try {
    const file = path.join(process.cwd(), "public", "paper", "statistics.json");
    return JSON.parse(await fs.readFile(file, "utf-8")) as Stats;
  } catch {
    return null;
  }
}

async function figureExists(name: string): Promise<boolean> {
  try {
    await fs.access(path.join(process.cwd(), "public", "paper", "figures", name));
    return true;
  } catch {
    try {
      await fs.access(path.join(process.cwd(), "public", "paper", name));
      return true;
    } catch {
      return false;
    }
  }
}

function fmtP(p: number): string {
  if (!Number.isFinite(p)) return "—";
  if (p < 1e-16) return "< 1e-16";
  return p.toExponential(1);
}

export default async function ResearchPage() {
  const stats = await readStats();
  const exp = await readExperiments();
  const present = await Promise.all(
    FIGURES.map(async (f) => ({ ...f, ok: await figureExists(f.file) })),
  );

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 sm:py-10">
      <header className="mb-8 max-w-2xl">
        <h1 className="text-[1.75rem] font-semibold tracking-tight">
          Research artifacts
        </h1>
        <p className="mt-3 text-[0.9375rem] leading-relaxed text-[var(--text-secondary)]">
          Every figure, table and statistic behind this system, generated
          reproducibly from the training pipeline. Figures are available as
          300&nbsp;dpi PNG and vector PDF; tables as CSV and LaTeX.
        </p>
        <div className="mt-4 flex flex-wrap gap-1.5">
          <a href="/paper/tables/table1_headline.csv" className="chip" download>
            Table 1 · headline (CSV)
          </a>
          <a href="/paper/tables/table1_headline.tex" className="chip" download>
            Table 1 (LaTeX)
          </a>
          <a href="/paper/tables/table2_per_class.csv" className="chip" download>
            Table 2 · per class
          </a>
          <a href="/paper/tables/table3_morphometry.csv" className="chip" download>
            Table 3 · morphometry
          </a>
          <a href="/paper/morphometry.csv" className="chip" download>
            Raw morphometry
          </a>
          <a href="/paper/statistics.json" className="chip" download>
            statistics.json
          </a>
        </div>
      </header>

      {/* ---- methods ------------------------------------------------------ */}
      <section className="card mb-6 p-5 sm:p-6">
        <h2 className="text-base font-semibold tracking-tight">Method</h2>
        <ol className="mt-4 space-y-3.5 text-[0.875rem] leading-relaxed text-[var(--text-secondary)]">
          <li>
            <strong className="font-medium text-[var(--text-primary)]">
              Leakage control.
            </strong>{" "}
            The test set is drawn from original slices only. Every augmented
            image is embedded with a frozen ImageNet EfficientNet-B0, assigned
            to its single nearest original, and discarded if that source fell in
            validation or test. The standard protocol for this dataset — train
            on augmented, test on original — evaluates on data the model has
            effectively memorised.
          </li>
          <li>
            <strong className="font-medium text-[var(--text-primary)]">
              Classification.
            </strong>{" "}
            EfficientNet-B0 with a global-average-pool → single-linear head.
            Two-stage fine-tune, class-weighted cross-entropy with a balanced
            sampler, one-cycle schedule, selection on validation macro-F1.
          </li>
          <li>
            <strong className="font-medium text-[var(--text-primary)]">
              Calibration.
            </strong>{" "}
            A single temperature fitted on validation by L-BFGS, applied at
            inference.
          </li>
          <li>
            <strong className="font-medium text-[var(--text-primary)]">
              Localisation.
            </strong>{" "}
            Because the head is GAP → Linear, the classic class activation map
            is exact and gradient-free — a 1×1 convolution folded into the ONNX
            graph at export. The map is warped to template space and integrated
            over atlas regions.
          </li>
          <li>
            <strong className="font-medium text-[var(--text-primary)]">
              Morphometry.
            </strong>{" "}
            Intracranial mask by threshold, largest component, closing and hole
            filling; CSF/grey/white by recursive Otsu on intracranial pixels;
            ventricles as central CSF components. All fractions normalised by
            intracranial area to cancel head size.
          </li>
        </ol>
      </section>

      {/* ---- headline table ----------------------------------------------- */}
      {stats && (
        <section className="card mb-6 p-5 sm:p-6">
          <h2 className="text-base font-semibold tracking-tight">
            Table 1 — Held-out performance
          </h2>
          <p className="mt-1.5 text-[0.8125rem] text-[var(--text-secondary)]">
            {stats.n_test} original slices · {stats.n_bootstrap} bootstrap
            resamples · scored through the deployed ONNX endpoint
          </p>
          <div className="mt-4 overflow-x-auto scroll-thin">
            <table className="w-full min-w-[420px] text-[0.8125rem]">
              <thead>
                <tr className="text-left text-[var(--text-muted)]">
                  <th className="pb-2 font-medium">Metric</th>
                  <th className="pb-2 text-right font-medium">Value</th>
                  <th className="pb-2 text-right font-medium">95% CI</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(stats.headline).map(([k, v]) => (
                  <tr key={k} className="border-t border-[var(--border)]">
                    <td className="py-2 capitalize">{k.replace(/_/g, " ")}</td>
                    <td className="tnum py-2 text-right font-medium">
                      {v.value.toFixed(4)}
                    </td>
                    <td className="tnum py-2 text-right text-[var(--text-secondary)]">
                      [{v.ci95_low.toFixed(4)}, {v.ci95_high.toFixed(4)}]
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[0.75rem] leading-relaxed text-[var(--text-muted)]">
            {stats.error_analysis.total_errors} errors in total,{" "}
            {stats.error_analysis.adjacent_stage_errors} of them onto an
            adjacent severity stage.
          </p>
        </section>
      )}

      {/* ---- morphometry table -------------------------------------------- */}
      {stats && (
        <section className="card mb-6 p-5 sm:p-6">
          <h2 className="text-base font-semibold tracking-tight">
            Table 3 — Morphometry by stage
          </h2>
          <p className="mt-1.5 max-w-2xl text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
            Measured by segmentation with no involvement of the classifier, so
            these are an independent check that the labels track real anatomy.
            ρ is Spearman correlation with ordinal severity.
          </p>
          <div className="mt-4 overflow-x-auto scroll-thin">
            <table className="w-full min-w-[680px] text-[0.75rem]">
              <thead>
                <tr className="text-left text-[var(--text-muted)]">
                  <th className="pb-2 font-medium">Index</th>
                  {CLASS_DIRS.map((c) => (
                    <th key={c} className="pb-2 text-right font-medium">
                      {c.replace("Demented", "")}
                    </th>
                  ))}
                  <th className="pb-2 text-right font-medium">η²</th>
                  <th className="pb-2 text-right font-medium">ρ</th>
                  <th className="pb-2 text-right font-medium">p</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(stats.morphometry).map(([key, m]) => (
                  <tr key={key} className="border-t border-[var(--border)]">
                    <td className="py-2">{m.label}</td>
                    {CLASS_DIRS.map((c) => (
                      <td
                        key={c}
                        className="tnum py-2 text-right text-[var(--text-secondary)]"
                      >
                        {m.by_class[c]?.mean.toFixed(3) ?? "—"}
                      </td>
                    ))}
                    <td className="tnum py-2 text-right">
                      {m.eta_squared.toFixed(3)}
                    </td>
                    <td className="tnum py-2 text-right">
                      {m.spearman_rho_vs_severity >= 0 ? "+" : ""}
                      {m.spearman_rho_vs_severity.toFixed(3)}
                    </td>
                    <td className="tnum py-2 text-right text-[var(--text-secondary)]">
                      {fmtP(m.anova_p)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* ---- headline finding --------------------------------------------- */}
      <section
        className="mb-6 rounded-md border p-5"
        style={{
          borderColor: "color-mix(in srgb, var(--danger) 35%, transparent)",
          background: "color-mix(in srgb, var(--danger) 7%, transparent)",
        }}
      >
        <span className="lozenge lozenge-danger">Primary finding</span>
        <h2 className="mt-2.5 text-base font-semibold tracking-tight">
          The 99.8% is largely leakage, and two controls prove it
        </h2>
        <p className="mt-2 max-w-3xl text-[0.875rem] leading-relaxed text-[var(--text-secondary)]">
          Retrain the identical architecture on <strong>original images only</strong>{" "}
          and it scores macro-F1 <strong>0.504</strong>. Classical models on seven
          segmented morphometric indices, same split, reach <strong>0.581</strong>.
          Two methods sharing no architecture, features or optimiser land in the
          same band. Everything above it is bought with augmented copies of the
          test subjects — and because subject identity is irrecoverable from this
          dataset, no split can repair it. Our own slice-level filter recovered
          just 0.47 accuracy points.
        </p>
        <div className="mt-4 overflow-x-auto scroll-thin">
          <table className="data-table min-w-[520px]">
            <thead>
              <tr>
                <th>Configuration</th>
                <th className="text-right">Train n</th>
                <th className="text-right">Accuracy</th>
                <th className="text-right">Macro F1</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Morphometry only (7 indices)", "1,015", "0.5687", "0.5807", true],
                ["CNN, original images only", "4,160", "0.5930", "0.5042", true],
                ["CNN + leak-filtered augmented", "26,266", "0.9953", "0.9972", false],
                ["CNN + all augmented (standard)", "38,144", "1.0000", "1.0000", false],
              ].map((r) => (
                <tr key={r[0] as string}>
                  <td>
                    <span className="flex items-center gap-2">
                      <span
                        className="h-2 w-2 shrink-0 rounded-sm"
                        style={{
                          background: r[4] ? "var(--series-3)" : "var(--series-2)",
                        }}
                        aria-hidden="true"
                      />
                      {r[0]}
                    </span>
                  </td>
                  <td className="tnum text-right">{r[1]}</td>
                  <td className="tnum text-right">{r[2]}</td>
                  <td className="tnum text-right font-semibold">{r[3]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-[0.75rem] leading-relaxed text-[var(--text-muted)]">
          Green rows use no augmented data and are the honest estimate. The
          leak filter works at slice level, but each subject contributes ~32
          slices — an augmented derivative of any one of them discloses that
          subject whichever slice is tested. No slice-level procedure can fix
          subject-level leakage.
        </p>
      </section>

      {/* ---- clinical utility --------------------------------------------- */}
      {exp?.throughput && (
        <section className="card mb-6 p-5 sm:p-6">
          <h2 className="text-base font-semibold tracking-tight">
            Table 4 — Measured throughput and cost
          </h2>
          <p className="mt-1.5 max-w-2xl text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
            Any claim about saving clinician time or money needs numbers behind
            it. These are measured end-to-end through the deployed path, on CPU.
          </p>
          <dl className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
            {[
              {
                k: "Classify",
                v: `${exp.throughput.classification_only.median_ms.toFixed(0)} ms`,
                h: `${exp.throughput.classification_only.throughput_per_min.toFixed(0)}/min`,
              },
              {
                k: "Full report",
                v: `${exp.throughput.full_pipeline.median_ms.toFixed(0)} ms`,
                h: `${exp.throughput.full_pipeline.throughput_per_min.toFixed(0)}/min`,
              },
              {
                k: "Cost / 1000",
                v: `$${exp.throughput.cost_model.usd_per_1000_scans.toFixed(4)}`,
                h: "serverless, warm",
              },
              { k: "Hardware", v: "CPU", h: "no GPU required" },
            ].map((m) => (
              <div key={m.k}>
                <dt className="text-[0.6875rem] uppercase tracking-wide text-[var(--text-muted)]">
                  {m.k}
                </dt>
                <dd className="tnum mt-0.5 text-xl font-semibold leading-none">
                  {m.v}
                </dd>
                <p className="mt-1 text-[0.6875rem] text-[var(--text-muted)]">
                  {m.h}
                </p>
              </div>
            ))}
          </dl>
          <p className="mt-4 text-[0.75rem] leading-relaxed text-[var(--text-muted)]">
            {exp.throughput.cost_model.assumption} These figures describe
            inference cost only — they are not a claim that the system saves
            clinician time, which requires the reader study.
          </p>
        </section>
      )}

      {/* ---- saliency verdict --------------------------------------------- */}
      {exp?.saliency && (
        <section className="card mb-6 p-5 sm:p-6">
          <h2 className="text-base font-semibold tracking-tight">
            Saliency faithfulness — a split verdict
          </h2>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {[
              {
                name: "Deletion",
                pass: exp.saliency.deletion_passes,
                cam: exp.saliency.auc_deletion_cam,
                rand: exp.saliency.auc_deletion_random,
                better: "lower",
              },
              {
                name: "Insertion",
                pass: exp.saliency.insertion_passes,
                cam: exp.saliency.auc_insertion_cam,
                rand: exp.saliency.auc_insertion_random,
                better: "higher",
              },
            ].map((t) => (
              <div
                key={t.name}
                className="rounded-xl border p-4"
                style={{
                  borderColor: `color-mix(in srgb, ${
                    t.pass ? "var(--good)" : "var(--serious)"
                  } 40%, transparent)`,
                  background: `color-mix(in srgb, ${
                    t.pass ? "var(--good)" : "var(--serious)"
                  } 8%, transparent)`,
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[0.875rem] font-medium">{t.name}</span>
                  <span
                    className="text-[0.75rem] font-medium"
                    style={{
                      color: t.pass ? "var(--good)" : "var(--serious)",
                    }}
                  >
                    {t.pass ? "passes" : "fails"}
                  </span>
                </div>
                <p className="tnum mt-2 text-[0.8125rem] text-[var(--text-secondary)]">
                  CAM {t.cam.toFixed(3)} · random {t.rand.toFixed(3)} ({t.better}{" "}
                  is better)
                </p>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
            {exp.saliency.interpretation}
          </p>
        </section>
      )}

      {/* ---- figures ------------------------------------------------------ */}
      <section className="space-y-6">
        <h2 className="text-base font-semibold tracking-tight">Figures</h2>
        {present.map((fig, i) =>
          fig.ok ? (
            <figure key={fig.file} className="card overflow-hidden">
              <div className="bg-white p-3">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={
                    fig.file.startsWith("qc_")
                      ? `/paper/${fig.file}`
                      : `/paper/figures/${fig.file}`
                  }
                  alt={fig.title}
                  className="w-full"
                  loading="lazy"
                />
              </div>
              <figcaption className="border-t border-[var(--border)] p-4">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <p className="text-[0.875rem] font-medium">
                    {fig.file.startsWith("fig")
                      ? `Figure ${i + 1} — ${fig.title}`
                      : fig.title}
                  </p>
                  {fig.file.startsWith("fig") && (
                    <a
                      href={`/paper/figures/${fig.file.replace(".png", ".pdf")}`}
                      className="text-[0.75rem] text-[var(--accent)] hover:underline"
                      download
                    >
                      vector PDF ↓
                    </a>
                  )}
                </div>
                <p className="mt-1.5 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
                  {fig.caption}
                </p>
              </figcaption>
            </figure>
          ) : null,
        )}
        {present.every((f) => !f.ok) && (
          <div className="card p-6">
            <p className="text-sm">No figures are bundled with this deployment.</p>
            <pre className="mt-3 overflow-x-auto rounded-lg bg-[var(--surface-2)] p-3 text-[0.75rem] leading-relaxed">
{`cd training
python build_anatomy_norms.py
python paper_stats.py
python paper_figures.py`}
            </pre>
          </div>
        )}
      </section>

      {/* ---- limitations --------------------------------------------------- */}
      <section
        className="mt-6 rounded-xl border p-5"
        style={{
          borderColor: "color-mix(in srgb, var(--warning) 40%, transparent)",
          background: "color-mix(in srgb, var(--warning) 8%, transparent)",
        }}
      >
        <h2 className="text-base font-semibold tracking-tight">
          Limitations to state in any write-up
        </h2>
        <ul className="mt-3 space-y-2 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
          <li className="flex gap-2">
            <span aria-hidden="true">·</span>
            <span>
              <strong className="font-medium text-[var(--text-primary)]">
                No subject identifiers.
              </strong>{" "}
              Slices from one brain appear across splits and cannot be
              separated. Accuracy therefore overstates performance on an unseen
              patient — the headline number is a within-cohort ceiling, not a
              generalisation estimate.
            </span>
          </li>
          <li className="flex gap-2">
            <span aria-hidden="true">·</span>
            <span>
              <strong className="font-medium text-[var(--text-primary)]">
                Single plane, single level.
              </strong>{" "}
              Axial slices near the ventricular body only. The plane is verified,
              not classified; coronal and sagittal discrimination would need a
              multi-plane training set.
            </span>
          </li>
          <li className="flex gap-2">
            <span aria-hidden="true">·</span>
            <span>
              <strong className="font-medium text-[var(--text-primary)]">
                Coarse atlas.
              </strong>{" "}
              Lobar boundaries are geometric priors in template space, not a
              FreeSurfer-grade parcellation, and registration is affine rather
              than non-linear. Region names denote approximate territories. The
              hippocampus lies below this slice and is never measured directly.
            </span>
          </li>
          <li className="flex gap-2">
            <span aria-hidden="true">·</span>
            <span>
              <strong className="font-medium text-[var(--text-primary)]">
                Saliency ≠ pathology.
              </strong>{" "}
              A class activation map shows where a model attended. Treating it
              as a lesion detector is the most common over-reading of these
              methods.
            </span>
          </li>
          <li className="flex gap-2">
            <span aria-hidden="true">·</span>
            <span>
              <strong className="font-medium text-[var(--text-primary)]">
                ModerateDemented is tiny.
              </strong>{" "}
              64 original slices in the whole dataset, 13 in the test split.
              Its interval is wide however good the point estimate looks.
            </span>
          </li>
        </ul>
        <p className="mt-3 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
          Full method and data handling in{" "}
          <Link href="/about" className="text-[var(--accent)] hover:underline">
            How it works
          </Link>
          .
        </p>
      </section>
    </div>
  );
}
