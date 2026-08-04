"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import AnatomyReport from "@/components/AnatomyReport";
import Dropzone from "@/components/Dropzone";
import ProbabilityBars from "@/components/ProbabilityBars";
import ScanViewer from "@/components/ScanViewer";
import { classById } from "@/lib/classes";
import { fileToDataUrl, loadMetrics, runPrediction } from "@/lib/predict";
import { downloadReport } from "@/lib/pdf";
import { getSessionId } from "@/lib/session";
import type { ModelMetrics, PredictResponse } from "@/lib/types";

type Phase = "idle" | "running" | "done" | "error";
type SaveState = "idle" | "saving" | "saved" | "error";

const SAMPLES = [
  { file: "non-demented.jpg", label: "Non demented" },
  { file: "very-mild-demented.jpg", label: "Very mild" },
  { file: "mild-demented.jpg", label: "Mild" },
  { file: "moderate-demented.jpg", label: "Moderate" },
];

function Alert({
  tone,
  title,
  children,
}: {
  tone: "critical" | "warning";
  title: string;
  children: React.ReactNode;
}) {
  const color = tone === "critical" ? "var(--critical)" : "var(--warning)";
  return (
    <div
      role="alert"
      className="flex gap-2.5 rounded-xl border p-3"
      style={{
        borderColor: `color-mix(in srgb, ${color} 40%, transparent)`,
        background: `color-mix(in srgb, ${color} 9%, transparent)`,
      }}
    >
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        className="mt-0.5 shrink-0"
        style={{ color }}
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
      <div>
        <p className="text-[0.8125rem] font-medium">{title}</p>
        <p className="mt-0.5 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
          {children}
        </p>
      </div>
    </div>
  );
}

export default function Analyzer() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [saveError, setSaveError] = useState<string | null>(null);
  const [contribute, setContribute] = useState(true);
  const [note, setNote] = useState("");

  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadMetrics().then(setMetrics);
  }, []);

  const analyze = useCallback(async (target: File) => {
    setPhase("running");
    setError(null);
    setResult(null);
    setSaveState("idle");
    setSaveError(null);
    setElapsed(null);

    const started = performance.now();
    try {
      const [dataUrl, prediction] = await Promise.all([
        fileToDataUrl(target),
        runPrediction(target),
      ]);
      setPreview(dataUrl);
      setResult(prediction);
      setElapsed(Math.round(performance.now() - started));
      setPhase("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
      setPhase("error");
    }
  }, []);

  const onFile = useCallback(
    (f: File) => {
      setFile(f);
      void analyze(f);
    },
    [analyze],
  );

  async function useSample(name: string) {
    try {
      setPhase("running");
      const res = await fetch(`/samples/${name}`);
      if (!res.ok) throw new Error("Sample image is not available.");
      const blob = await res.blob();
      const f = new File([blob], name, { type: blob.type || "image/jpeg" });
      setFile(f);
      await analyze(f);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load the sample.");
      setPhase("error");
    }
  }

  async function save() {
    if (!result || !preview) return;
    setSaveState("saving");
    setSaveError(null);
    try {
      const res = await fetch("/api/scans", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-session-id": getSessionId(),
        },
        body: JSON.stringify({
          prediction: result,
          image: preview,
          filename: file?.name ?? null,
          note: note.trim() || null,
          contributeToTraining: contribute,
        }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error ?? "Save failed.");
      setSaveState("saved");
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Save failed.");
      setSaveState("error");
    }
  }

  function reset() {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setPhase("idle");
    setSaveState("idle");
    setNote("");
  }

  const predicted = result ? classById(result.class_id) : null;
  const perClass =
    result && metrics ? metrics.per_class[result.label] ?? null : null;
  const lowMargin = result ? result.margin < 0.15 : false;
  const badInput = result ? !result.input_check.looks_like_brain_mri : false;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] lg:gap-8">
      {/* ------------------------------------------------------------ input */}
      <section className="lg:sticky lg:top-20 lg:self-start">
        <div className="card p-5">
          {!preview || phase === "idle" ? (
            <>
              <Dropzone onFile={onFile} disabled={phase === "running"} />
              <div className="mt-5">
                <p className="text-[0.75rem] font-medium uppercase tracking-wide text-[var(--text-muted)]">
                  Or try a held-out sample
                </p>
                <div className="mt-2.5 flex flex-wrap gap-1.5">
                  {SAMPLES.map((s) => (
                    <button
                      key={s.file}
                      className="btn !px-2.5 !py-1.5 text-[0.75rem]"
                      onClick={() => useSample(s.file)}
                      disabled={phase === "running"}
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
                <p className="mt-2.5 text-[0.75rem] leading-relaxed text-[var(--text-muted)]">
                  Samples come from the held-out test split — the model never saw
                  them or any augmentation of them during training.
                </p>
              </div>
            </>
          ) : (
            <>
              <ScanViewer
                original={preview}
                overlay={result?.overlay_png ?? null}
                heatmap={result?.cam_png ?? null}
              />
              <div className="mt-4 flex flex-wrap gap-2">
                <button className="btn flex-1" onClick={reset}>
                  Analyze another scan
                </button>
                {result && (
                  <button
                    className="btn"
                    onClick={() =>
                      void downloadReport(result, preview, metrics)
                    }
                  >
                    <svg
                      width="15"
                      height="15"
                      viewBox="0 0 24 24"
                      fill="none"
                      aria-hidden="true"
                    >
                      <path
                        d="M12 4v10m0 0 4-4m-4 4-4-4"
                        stroke="currentColor"
                        strokeWidth="1.6"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <path
                        d="M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16"
                        stroke="currentColor"
                        strokeWidth="1.6"
                        strokeLinecap="round"
                      />
                    </svg>
                    Report
                  </button>
                )}
              </div>
              {file && (
                <p className="mt-3 truncate text-[0.75rem] text-[var(--text-muted)]">
                  {file.name} · {(file.size / 1024).toFixed(0)} KB
                  {result &&
                    ` · ${result.image_size[0]}×${result.image_size[1]} px`}
                </p>
              )}
            </>
          )}
        </div>
      </section>

      {/* ----------------------------------------------------------- output */}
      <section ref={resultRef} className="min-w-0">
        {phase === "idle" && (
          <div className="card flex h-full min-h-[320px] flex-col items-center justify-center p-8 text-center">
            <div
              className="mb-4 flex h-12 w-12 items-center justify-center rounded-full"
              style={{ background: "var(--surface-2)" }}
              aria-hidden="true"
            >
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                <circle
                  cx="11"
                  cy="11"
                  r="6.5"
                  stroke="var(--text-muted)"
                  strokeWidth="1.6"
                />
                <path
                  d="m16 16 4 4"
                  stroke="var(--text-muted)"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                />
              </svg>
            </div>
            <p className="text-sm font-medium">No scan analyzed yet</p>
            <p className="mt-1.5 max-w-sm text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
              Upload an axial brain MRI slice and the classifier will return a
              four-stage prediction, a calibrated confidence, and a map of the
              regions it weighted.
            </p>
          </div>
        )}

        {phase === "running" && (
          <div className="card p-6">
            <div className="flex items-center gap-2.5">
              <span
                className="h-2 w-2 rounded-full pulse-soft"
                style={{ background: "var(--accent)" }}
                aria-hidden="true"
              />
              <p className="text-sm font-medium">Running inference…</p>
            </div>
            <div
              className="sweep relative mt-4 h-1 overflow-hidden rounded-full"
              style={{ background: "var(--surface-3)" }}
              role="progressbar"
              aria-label="Analyzing"
            />
            <p className="mt-3 text-[0.8125rem] text-[var(--text-secondary)]">
              The first request after a quiet period also pays a cold start
              while the ONNX session loads.
            </p>
            <div className="mt-5 space-y-2.5">
              {[0, 1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="h-2 rounded-full pulse-soft"
                  style={{
                    background: "var(--surface-3)",
                    width: `${88 - i * 14}%`,
                    animationDelay: `${i * 110}ms`,
                  }}
                />
              ))}
            </div>
          </div>
        )}

        {phase === "error" && (
          <div className="card p-6">
            <Alert tone="critical" title="Prediction failed">
              {error}
            </Alert>
            <button className="btn mt-4" onClick={reset}>
              Start over
            </button>
          </div>
        )}

        {phase === "done" && result && predicted && (
          <div className="space-y-4 rise">
            {/* headline */}
            <div className="card p-5 sm:p-6">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-[0.6875rem] font-medium uppercase tracking-[0.08em] text-[var(--text-muted)]">
                    Predicted stage
                  </p>
                  <h2 className="mt-1.5 text-[1.75rem] font-semibold leading-tight tracking-tight text-[var(--accent)]">
                    {result.label}
                  </h2>
                  <p className="mt-1.5 max-w-sm text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
                    {predicted.blurb}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-[0.6875rem] font-medium uppercase tracking-[0.08em] text-[var(--text-muted)]">
                    Confidence
                  </p>
                  <p className="mt-1.5 text-[1.75rem] font-semibold leading-tight tracking-tight">
                    {(result.confidence * 100).toFixed(1)}%
                  </p>
                  <p className="mt-0.5 text-[0.6875rem] text-[var(--text-muted)]">
                    temperature-calibrated
                  </p>
                </div>
              </div>

              <div className="mt-5 border-t border-[var(--border)] pt-4">
                <ProbabilityBars
                  probabilities={result.probabilities}
                  predictedId={result.class_id}
                />
              </div>
            </div>

            {/* reliability */}
            {(badInput || result.out_of_distribution || lowMargin) && (
              <div className="space-y-2.5">
                {badInput && (
                  <Alert tone="critical" title="This may not be a brain MRI slice">
                    The structure check failed{" "}
                    {Object.entries(result.input_check.checks)
                      .filter(([, ok]) => !ok)
                      .map(([k]) => k.replace(/_/g, " "))
                      .join(", ")}
                    . The model will still return a stage for any image you give
                    it — that output means nothing here.
                  </Alert>
                )}
                {result.out_of_distribution && (
                  <Alert tone="warning" title="Outside the training distribution">
                    This scan&apos;s features sit beyond the 99th percentile of the
                    held-out test set, so the confidence above is not supported by
                    anything the model has been measured on.
                  </Alert>
                )}
                {lowMargin && !badInput && (
                  <Alert tone="warning" title="Close call between two stages">
                    Only {(result.margin * 100).toFixed(1)} percentage points
                    separate the top two stages. Adjacent Alzheimer&apos;s stages
                    are genuinely hard to separate from a single slice.
                  </Alert>
                )}
              </div>
            )}

            {/* anatomical localisation, morphometry and findings */}
            <AnatomyReport
              anatomy={result.anatomy}
              report={result.report}
            />

            {/* model context */}
            <div className="card p-5">
              <div className="flex items-baseline justify-between gap-3">
                <h3 className="text-sm font-semibold">What this number is worth</h3>
                <Link
                  href="/model"
                  className="text-[0.75rem] text-[var(--accent)] hover:underline"
                >
                  Full model card →
                </Link>
              </div>

              {metrics && perClass ? (
                <>
                  <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
                    {[
                      {
                        k: "Recall",
                        v: `${(perClass.recall * 100).toFixed(1)}%`,
                        h: `of true "${predicted.short}" scans found`,
                      },
                      {
                        k: "Precision",
                        v: `${(perClass.precision * 100).toFixed(1)}%`,
                        h: "of this prediction is correct",
                      },
                      {
                        k: "Class AUC",
                        v: perClass.auc.toFixed(3),
                        h: "one-vs-rest",
                      },
                      {
                        k: "Test images",
                        v: String(perClass.support),
                        h: "held out for this class",
                      },
                    ].map((m) => (
                      <div key={m.k}>
                        <dt className="text-[0.6875rem] uppercase tracking-wide text-[var(--text-muted)]">
                          {m.k}
                        </dt>
                        <dd className="tnum mt-0.5 text-lg font-semibold leading-none">
                          {m.v}
                        </dd>
                        <p className="mt-1 text-[0.6875rem] leading-snug text-[var(--text-muted)]">
                          {m.h}
                        </p>
                      </div>
                    ))}
                  </dl>
                  <p className="mt-4 text-[0.75rem] leading-relaxed text-[var(--text-muted)]">
                    Measured on {metrics.test_set.n} held-out original MRI slices
                    whose augmented derivatives were removed from training. The
                    dataset has no subject identifiers, so real-world accuracy on
                    a new patient is lower than this.
                  </p>
                </>
              ) : (
                <p className="mt-3 text-[0.8125rem] text-[var(--text-secondary)]">
                  Held-out metrics are not bundled in this deployment. Run{" "}
                  <code className="rounded bg-[var(--surface-2)] px-1 py-0.5 text-[0.75rem]">
                    training/evaluate.py
                  </code>{" "}
                  to generate them.
                </p>
              )}

              <div className="mt-4 flex flex-wrap gap-1.5 border-t border-[var(--border)] pt-3.5">
                <span className="chip">
                  {result.model.name} v{result.model.version}
                </span>
                {elapsed !== null && <span className="chip">{elapsed} ms round trip</span>}
                <span className="chip">T = {result.model.temperature.toFixed(2)}</span>
                <span className="chip">
                  margin {(result.margin * 100).toFixed(1)} pts
                </span>
              </div>
            </div>

            {/* save */}
            <div className="card p-5">
              <h3 className="text-sm font-semibold">Save this scan</h3>
              <p className="mt-1.5 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
                Saved scans appear in your{" "}
                <Link href="/history" className="text-[var(--accent)] hover:underline">
                  history
                </Link>
                , which is tied to this browser only — there are no accounts.
              </p>

              {saveState === "saved" ? (
                <div className="mt-4 flex items-center gap-2 text-[0.8125rem]">
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    style={{ color: "var(--good)" }}
                    aria-hidden="true"
                  >
                    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" />
                    <path
                      d="m8.2 12.3 2.5 2.5 5-5.2"
                      stroke="currentColor"
                      strokeWidth="1.8"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <span>Saved.</span>
                  <Link href="/history" className="text-[var(--accent)] hover:underline">
                    View history
                  </Link>
                </div>
              ) : (
                <>
                  <input
                    className="field mt-4"
                    placeholder="Optional note (e.g. slice index, source)"
                    value={note}
                    maxLength={200}
                    onChange={(e) => setNote(e.target.value)}
                  />
                  <label className="mt-3 flex cursor-pointer items-start gap-2.5">
                    <input
                      type="checkbox"
                      checked={contribute}
                      onChange={(e) => setContribute(e.target.checked)}
                      className="mt-0.5 accent-[var(--accent)]"
                    />
                    <span className="text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
                      Also add it to the retraining pool. It is only ever trained
                      on after a human assigns a verified label in the review
                      console — the model&apos;s own guess is never used as ground
                      truth.
                    </span>
                  </label>

                  <button
                    className="btn btn-primary mt-4 w-full"
                    onClick={() => void save()}
                    disabled={saveState === "saving"}
                  >
                    {saveState === "saving" ? "Saving…" : "Save to history"}
                  </button>

                  {saveError && (
                    <p
                      className="mt-2.5 text-[0.8125rem]"
                      style={{ color: "var(--critical)" }}
                      role="alert"
                    >
                      {saveError}
                    </p>
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
