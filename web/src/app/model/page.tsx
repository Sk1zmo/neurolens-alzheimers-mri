import fs from "node:fs/promises";
import path from "node:path";
import Link from "next/link";
import ConfusionMatrix from "@/components/ConfusionMatrix";
import RocChart from "@/components/RocChart";
import StatTile from "@/components/StatTile";
import CorpusPanel from "@/components/CorpusPanel";
import type { ModelMetrics } from "@/lib/types";

export const metadata = { title: "Model card" };
export const revalidate = 3600;

async function readMetrics(): Promise<ModelMetrics | null> {
  try {
    const file = path.join(process.cwd(), "public", "model", "metrics.json");
    return JSON.parse(await fs.readFile(file, "utf-8")) as ModelMetrics;
  } catch {
    return null;
  }
}

function Section({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="card p-5 sm:p-6">
      <h2 className="text-base font-semibold tracking-tight">{title}</h2>
      {description && (
        <p className="mt-1.5 max-w-2xl text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
          {description}
        </p>
      )}
      <div className="mt-5">{children}</div>
    </section>
  );
}

export default async function ModelPage() {
  const metrics = await readMetrics();

  if (!metrics) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-16 sm:px-6">
        <h1 className="text-2xl font-semibold tracking-tight">Model card</h1>
        <div className="card mt-6 p-6">
          <p className="text-sm">
            No evaluation report is bundled with this deployment.
          </p>
          <p className="mt-2 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
            Run the pipeline to generate it:
          </p>
          <pre className="mt-3 overflow-x-auto rounded-lg bg-[var(--surface-2)] p-3 text-[0.75rem] leading-relaxed">
{`cd training
python prepare_split.py
python train.py
python evaluate.py     # writes artifacts/reports/metrics.json
python export_onnx.py  # copies it into web/public/model/`}
          </pre>
        </div>
      </div>
    );
  }

  const h = metrics.headline;

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 sm:py-10">
      <header className="mb-8 max-w-2xl">
        <h1 className="text-[1.75rem] font-semibold tracking-tight">Model card</h1>
        <p className="mt-3 text-[0.9375rem] leading-relaxed text-[var(--text-secondary)]">
          Every figure here comes from {metrics.test_set.n} held-out original MRI
          slices that the model never saw — and whose augmented derivatives were
          removed from the training set. Nothing on this page is a training-set
          number.
        </p>
      </header>

      <div className="space-y-5">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatTile
            label="Accuracy"
            value={`${(h.accuracy * 100).toFixed(1)}%`}
            hint="held-out test set"
            tone="accent"
          />
          <StatTile
            label="Balanced acc."
            value={`${(h.balanced_accuracy * 100).toFixed(1)}%`}
            hint="mean per-class recall"
          />
          <StatTile
            label="Macro F1"
            value={h.macro_f1.toFixed(3)}
            hint="all classes weighted equally"
          />
          <StatTile
            label="Macro AUC"
            value={h.macro_auc_ovr.toFixed(3)}
            hint="one-vs-rest"
          />
        </div>

        <Section
          title="Where it gets things wrong"
          description="Rows are the true stage, columns what the model predicted, coloured by share of the true class. Adjacent-stage confusions sit immediately off the diagonal."
        >
          <ConfusionMatrix matrix={metrics.confusion_matrix} />
        </Section>

        <Section
          title="Separability by stage"
          description="One-vs-rest ROC. The curve answers how well this stage can be ranked apart from the other three at any operating threshold."
        >
          <RocChart metrics={metrics} />
        </Section>

        <Section
          title="Is the confidence honest?"
          description="A fine-tuned CNN's raw softmax is systematically over-confident. Fitting a single temperature on the validation split rescales it so a stated 80% actually means roughly 80%."
        >
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatTile
              label="ECE before"
              value={h.ece_uncalibrated.toFixed(4)}
              hint="raw softmax"
            />
            <StatTile
              label="ECE after"
              value={h.ece_calibrated.toFixed(4)}
              hint={`T = ${metrics.temperature.toFixed(3)}`}
              tone="accent"
            />
            <StatTile
              label="Brier score"
              value={h.brier_score.toFixed(4)}
              hint="lower is better"
            />
            <StatTile
              label="Cohen's kappa"
              value={h.cohen_kappa.toFixed(3)}
              hint="agreement beyond chance"
            />
          </div>
          <p className="mt-4 text-[0.75rem] leading-relaxed text-[var(--text-muted)]">
            Expected calibration error is the average gap between stated
            confidence and observed accuracy across 15 confidence bins. The
            deployed model applies this temperature at inference, so the
            percentage shown on the analyze page is the calibrated one.
          </p>
        </Section>

        <Section
          title="Per-class breakdown"
          description="ModerateDemented is the class to watch: the original dataset contains only 64 such slices in total, so its held-out support is small and its metrics are correspondingly noisy."
        >
          <div className="overflow-x-auto scroll-thin">
            <table className="w-full min-w-[460px] text-[0.8125rem]">
              <thead>
                <tr className="text-left text-[var(--text-muted)]">
                  <th className="pb-2 font-medium">Stage</th>
                  <th className="pb-2 text-right font-medium">Precision</th>
                  <th className="pb-2 text-right font-medium">Recall</th>
                  <th className="pb-2 text-right font-medium">F1</th>
                  <th className="pb-2 text-right font-medium">AUC</th>
                  <th className="pb-2 text-right font-medium">Support</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics.per_class).map(([name, m]) => (
                  <tr key={name} className="border-t border-[var(--border)]">
                    <td className="py-2 font-medium">{name}</td>
                    <td className="tnum py-2 text-right">
                      {(m.precision * 100).toFixed(1)}%
                    </td>
                    <td className="tnum py-2 text-right">
                      {(m.recall * 100).toFixed(1)}%
                    </td>
                    <td className="tnum py-2 text-right">{m.f1.toFixed(3)}</td>
                    <td className="tnum py-2 text-right">{m.auc.toFixed(3)}</td>
                    <td className="tnum py-2 text-right text-[var(--text-secondary)]">
                      {m.support}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section
          title="Live contribution corpus"
          description="Scans uploaded through this app, and how many have been reviewed into training-ready data."
        >
          <CorpusPanel />
        </Section>

        <Section title="Specification">
          <dl className="grid gap-x-6 gap-y-3 text-[0.8125rem] sm:grid-cols-2">
            {[
              ["Architecture", metrics.architecture],
              ["Input", `${metrics.img_size}×${metrics.img_size} RGB, ImageNet normalisation`],
              ["Serving format", "ONNX (opset 17), onnxruntime CPU"],
              ["Explanation", "Class activation map folded into the graph as a 1×1 conv"],
              ["Calibration", `temperature scaling, T = ${metrics.temperature.toFixed(3)}`],
              ["Test-time augmentation", metrics.tta_horizontal_flip ? "horizontal flip" : "none"],
              ["Test set", metrics.test_set.source],
              ["Class support", metrics.test_set.per_class_n.join(" / ")],
            ].map(([k, v]) => (
              <div key={k as string}>
                <dt className="text-[0.6875rem] uppercase tracking-wide text-[var(--text-muted)]">
                  {k}
                </dt>
                <dd className="mt-0.5 leading-relaxed">{v}</dd>
              </div>
            ))}
          </dl>
        </Section>

        <section
          className="rounded-xl border p-5"
          style={{
            borderColor: "color-mix(in srgb, var(--warning) 40%, transparent)",
            background: "color-mix(in srgb, var(--warning) 8%, transparent)",
          }}
        >
          <h2 className="flex items-center gap-2 text-base font-semibold tracking-tight">
            <svg
              width="17"
              height="17"
              viewBox="0 0 24 24"
              fill="none"
              style={{ color: "var(--warning)" }}
              aria-hidden="true"
            >
              <path
                d="M12 3.5 21 19H3l9-15.5Z"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinejoin="round"
              />
              <path
                d="M12 10v3.6M12 16.4v.2"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
            </svg>
            Limitations you should read before quoting these numbers
          </h2>
          <ul className="mt-3 space-y-2 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
            {metrics.caveats.map((c) => (
              <li key={c} className="flex gap-2">
                <span aria-hidden="true">·</span>
                <span>{c}</span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
            More detail in{" "}
            <Link href="/about" className="text-[var(--accent)] hover:underline">
              How it works
            </Link>
            .
          </p>
        </section>
      </div>
    </div>
  );
}
