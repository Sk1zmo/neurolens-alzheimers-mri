"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { CLASSES, classByDir } from "@/lib/classes";
import { runPrediction } from "@/lib/predict";

const STORE = "neurolens.readerStudy";

interface Case {
  file: string;
  truth: string;
}

interface Rating {
  file: string;
  truth: string;
  reader: string;
  confidence: number;
  model?: string;
  modelConfidence?: number;
  seconds: number;
  at: string;
}

/**
 * Cohen's kappa — agreement corrected for what chance alone would produce.
 * Raw agreement is misleading here because the class distribution is skewed;
 * a reader who always answered "NonDemented" would score well on percentage
 * agreement and zero on kappa.
 */
function kappa(a: string[], b: string[], labels: string[]): number {
  const n = a.length;
  if (n === 0) return NaN;
  let observed = 0;
  for (let i = 0; i < n; i++) if (a[i] === b[i]) observed++;
  const po = observed / n;
  let pe = 0;
  for (const l of labels) {
    const pa = a.filter((x) => x === l).length / n;
    const pb = b.filter((x) => x === l).length / n;
    pe += pa * pb;
  }
  return pe === 1 ? NaN : (po - pe) / (1 - pe);
}

function interpretKappa(k: number): string {
  if (!Number.isFinite(k)) return "not computable";
  if (k < 0.2) return "slight";
  if (k < 0.4) return "fair";
  if (k < 0.6) return "moderate";
  if (k < 0.8) return "substantial";
  return "almost perfect";
}

export default function RatingClient() {
  const [cases, setCases] = useState<Case[] | null>(null);
  const [index, setIndex] = useState(0);
  const [ratings, setRatings] = useState<Rating[]>([]);
  const [choice, setChoice] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(3);
  const [revealed, setRevealed] = useState<Rating | null>(null);
  const [busy, setBusy] = useState(false);
  const [startedAt, setStartedAt] = useState<number>(Date.now());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem(STORE);
    if (saved) {
      try {
        const parsed = JSON.parse(saved) as Rating[];
        setRatings(parsed);
        setIndex(parsed.length);
      } catch {
        /* ignore corrupt state */
      }
    }
    fetch("/reader-study/manifest.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("no manifest"))))
      .then((d) => setCases(d.cases as Case[]))
      .catch(() =>
        setError(
          "No reader-study set is bundled. Run `python training/make_reader_study.py` to generate one.",
        ),
      );
  }, []);

  useEffect(() => {
    setStartedAt(Date.now());
    setChoice(null);
    setConfidence(3);
    setRevealed(null);
  }, [index]);

  const current = cases?.[index] ?? null;

  const commit = useCallback(async () => {
    if (!current || !choice) return;
    setBusy(true);
    setError(null);
    const seconds = (Date.now() - startedAt) / 1000;
    const rating: Rating = {
      file: current.file,
      truth: current.truth,
      reader: choice,
      confidence,
      seconds,
      at: new Date().toISOString(),
    };
    try {
      const res = await fetch(`/reader-study/${current.file}`);
      const blob = await res.blob();
      const file = new File([blob], current.file, {
        type: blob.type || "image/jpeg",
      });
      const prediction = await runPrediction(file);
      rating.model = prediction.class_dir;
      rating.modelConfidence = prediction.confidence;
    } catch {
      /* the reader's rating still counts if the model call fails */
    }
    const next = [...ratings.filter((r) => r.file !== rating.file), rating];
    setRatings(next);
    localStorage.setItem(STORE, JSON.stringify(next));
    setRevealed(rating);
    setBusy(false);
  }, [current, choice, confidence, startedAt, ratings]);

  const stats = useMemo(() => {
    const done = ratings.filter((r) => r.model);
    const labels = CLASSES.map((c) => c.dir);
    const readerVsTruth = kappa(
      ratings.map((r) => r.reader),
      ratings.map((r) => r.truth),
      labels,
    );
    const modelVsTruth = kappa(
      done.map((r) => r.model!),
      done.map((r) => r.truth),
      labels,
    );
    const readerVsModel = kappa(
      done.map((r) => r.reader),
      done.map((r) => r.model!),
      labels,
    );
    const readerAcc =
      ratings.length === 0
        ? NaN
        : ratings.filter((r) => r.reader === r.truth).length / ratings.length;
    const modelAcc =
      done.length === 0
        ? NaN
        : done.filter((r) => r.model === r.truth).length / done.length;
    const medianSeconds =
      ratings.length === 0
        ? NaN
        : [...ratings.map((r) => r.seconds)].sort((a, b) => a - b)[
            Math.floor(ratings.length / 2)
          ];
    return {
      n: ratings.length,
      readerVsTruth,
      modelVsTruth,
      readerVsModel,
      readerAcc,
      modelAcc,
      medianSeconds,
    };
  }, [ratings]);

  function exportCsv() {
    const header =
      "file,truth,reader,reader_confidence,model,model_confidence,seconds,at\n";
    const body = ratings
      .map(
        (r) =>
          `${r.file},${r.truth},${r.reader},${r.confidence},${r.model ?? ""},${
            r.modelConfidence?.toFixed(4) ?? ""
          },${r.seconds.toFixed(1)},${r.at}`,
      )
      .join("\n");
    const url = URL.createObjectURL(
      new Blob([header + body], { type: "text/csv" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `reader-study-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  if (error && !cases) {
    return (
      <div className="card p-6">
        <p className="text-sm font-medium">Reader study unavailable</p>
        <p className="mt-2 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
          {error}
        </p>
      </div>
    );
  }

  if (!cases) {
    return <div className="card h-72 pulse-soft" aria-busy="true" />;
  }

  const finished = index >= cases.length;

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
      <section>
        {finished ? (
          <div className="card p-8 text-center">
            <p className="text-base font-semibold">Session complete</p>
            <p className="mt-2 text-[0.8125rem] text-[var(--text-secondary)]">
              {ratings.length} cases rated. Export the CSV for analysis.
            </p>
            <div className="mt-5 flex justify-center gap-2">
              <button className="btn btn-primary" onClick={exportCsv}>
                Export CSV
              </button>
              <button
                className="btn"
                onClick={() => {
                  localStorage.removeItem(STORE);
                  setRatings([]);
                  setIndex(0);
                }}
              >
                Start over
              </button>
            </div>
          </div>
        ) : (
          <div className="card p-5">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-[0.75rem] text-[var(--text-muted)]">
                Case {index + 1} of {cases.length}
              </span>
              <span
                className="h-1 w-32 overflow-hidden rounded-full"
                style={{ background: "var(--surface-3)" }}
              >
                <span
                  className="block h-full rounded-full"
                  style={{
                    width: `${((index + 1) / cases.length) * 100}%`,
                    background: "var(--accent)",
                  }}
                />
              </span>
            </div>

            <div
              className="relative aspect-square w-full overflow-hidden rounded-xl border border-[var(--border)]"
              style={{ background: "#0b0b0b" }}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`/reader-study/${current!.file}`}
                alt={`Case ${index + 1}`}
                className="absolute inset-0 h-full w-full object-contain"
              />
            </div>

            <p className="mt-4 text-[0.75rem] font-medium uppercase tracking-wide text-[var(--text-muted)]">
              Your rating
            </p>
            <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {CLASSES.map((c) => (
                <button
                  key={c.id}
                  disabled={Boolean(revealed)}
                  onClick={() => setChoice(c.dir)}
                  className={`btn !py-2 text-[0.8125rem] ${
                    choice === c.dir
                      ? "!border-[var(--accent)] !bg-[var(--accent-soft)]"
                      : ""
                  }`}
                >
                  {c.short}
                </button>
              ))}
            </div>

            <label className="mt-4 block">
              <span className="text-[0.75rem] text-[var(--text-secondary)]">
                Your confidence: {["very low", "low", "moderate", "high", "very high"][confidence - 1]}
              </span>
              <input
                type="range"
                min={1}
                max={5}
                value={confidence}
                disabled={Boolean(revealed)}
                onChange={(e) => setConfidence(Number(e.target.value))}
                className="mt-1.5 w-full accent-[var(--accent)]"
              />
            </label>

            {!revealed ? (
              <button
                className="btn btn-primary mt-4 w-full"
                disabled={!choice || busy}
                onClick={() => void commit()}
              >
                {busy ? "Scoring…" : "Commit rating and reveal"}
              </button>
            ) : (
              <div className="mt-4 rounded-xl border border-[var(--border)] bg-[var(--surface-2)] p-4">
                <div className="grid grid-cols-3 gap-3 text-center">
                  {[
                    { k: "You", v: classByDir(revealed.reader)?.short },
                    {
                      k: "Model",
                      v: revealed.model
                        ? classByDir(revealed.model)?.short
                        : "—",
                    },
                    { k: "Truth", v: classByDir(revealed.truth)?.short },
                  ].map((x) => (
                    <div key={x.k}>
                      <p className="text-[0.6875rem] uppercase tracking-wide text-[var(--text-muted)]">
                        {x.k}
                      </p>
                      <p className="mt-0.5 text-[0.875rem] font-medium">{x.v}</p>
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-center text-[0.75rem] text-[var(--text-secondary)]">
                  {revealed.reader === revealed.truth
                    ? "You matched the reference label."
                    : "You differed from the reference label."}
                  {revealed.model &&
                    (revealed.model === revealed.truth
                      ? " The model matched."
                      : " The model also differed.")}
                </p>
                <button
                  className="btn btn-primary mt-4 w-full"
                  onClick={() => setIndex((i) => i + 1)}
                >
                  Next case →
                </button>
              </div>
            )}
          </div>
        )}
      </section>

      <aside className="space-y-4 lg:sticky lg:top-20 lg:self-start">
        <div className="card p-5">
          <h2 className="text-sm font-semibold">Agreement so far</h2>
          <p className="mt-1.5 text-[0.75rem] leading-relaxed text-[var(--text-muted)]">
            Cohen&apos;s κ corrects for agreement expected by chance — necessary
            here because the classes are skewed.
          </p>
          <dl className="mt-4 space-y-3">
            {[
              { k: "Reader vs reference", v: stats.readerVsTruth },
              { k: "Model vs reference", v: stats.modelVsTruth },
              { k: "Reader vs model", v: stats.readerVsModel },
            ].map((row) => (
              <div key={row.k}>
                <dt className="text-[0.75rem] text-[var(--text-secondary)]">
                  {row.k}
                </dt>
                <dd className="tnum text-lg font-semibold leading-tight">
                  {Number.isFinite(row.v) ? row.v.toFixed(3) : "—"}
                  <span className="ml-2 text-[0.6875rem] font-normal text-[var(--text-muted)]">
                    {interpretKappa(row.v)}
                  </span>
                </dd>
              </div>
            ))}
          </dl>
          <div className="mt-4 grid grid-cols-3 gap-2 border-t border-[var(--border)] pt-3.5 text-center">
            {[
              {
                k: "Your acc",
                v: Number.isFinite(stats.readerAcc)
                  ? `${(stats.readerAcc * 100).toFixed(0)}%`
                  : "—",
              },
              {
                k: "Model acc",
                v: Number.isFinite(stats.modelAcc)
                  ? `${(stats.modelAcc * 100).toFixed(0)}%`
                  : "—",
              },
              {
                k: "Median s",
                v: Number.isFinite(stats.medianSeconds)
                  ? stats.medianSeconds.toFixed(1)
                  : "—",
              },
            ].map((x) => (
              <div key={x.k}>
                <p className="text-[0.6875rem] uppercase tracking-wide text-[var(--text-muted)]">
                  {x.k}
                </p>
                <p className="tnum mt-0.5 text-sm font-medium">{x.v}</p>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[0.6875rem] leading-relaxed text-[var(--text-muted)]">
            n = {stats.n}. Reading time is the gap between the case appearing and
            you committing — the number behind any time-saving claim.
          </p>
        </div>

        <button
          className="btn w-full"
          onClick={exportCsv}
          disabled={ratings.length === 0}
        >
          Export {ratings.length} ratings as CSV
        </button>
      </aside>
    </div>
  );
}
