"use client";

import { useState } from "react";
import type { AnatomyResult, FindingsReport } from "@/lib/types";

const SEVERITY: Record<string, { colour: string; label: string }> = {
  marked: { colour: "var(--critical)", label: "Marked" },
  notable: { colour: "var(--serious)", label: "Notable" },
  borderline: { colour: "var(--warning)", label: "Borderline" },
};

/**
 * A z-score is a signed deviation, which is a diverging quantity: the neutral
 * midpoint is the reference mean and the two arms mean opposite things. So the
 * bar grows left or right from a centre line rather than filling from zero.
 */
function ZBar({ z }: { z: number }) {
  const clamped = Math.max(-4, Math.min(4, z));
  const half = (Math.abs(clamped) / 4) * 50;
  const abnormal = Math.abs(z) >= 1.5;
  return (
    <div
      className="relative h-2 w-full overflow-hidden rounded-full"
      style={{ background: "var(--surface-3)" }}
      aria-hidden="true"
    >
      <span
        className="absolute inset-y-0 left-1/2 w-px"
        style={{ background: "var(--axis)" }}
      />
      <span
        className="absolute inset-y-0 rounded-full"
        style={{
          left: clamped < 0 ? `${50 - half}%` : "50%",
          width: `${half}%`,
          background: abnormal
            ? clamped > 0
              ? "var(--series-2)"
              : "var(--series-1)"
            : "var(--text-muted)",
          opacity: abnormal ? 0.95 : 0.45,
        }}
      />
    </div>
  );
}

export default function AnatomyReport({
  anatomy,
  report,
}: {
  anatomy: AnatomyResult | null;
  report: FindingsReport | null;
}) {
  const [showRefs, setShowRefs] = useState(false);
  const [showAll, setShowAll] = useState(false);

  if (!anatomy || anatomy.error) {
    return (
      <div className="card p-5">
        <h3 className="text-sm font-semibold">Anatomical analysis</h3>
        <p className="mt-2 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
          {anatomy?.error
            ? `Segmentation did not complete on this image (${anatomy.error}). The classification above is unaffected.`
            : "Not available for this scan."}
        </p>
      </div>
    );
  }

  const metricRows = Object.entries(anatomy.z_scores).sort(
    (a, b) => Math.abs(b[1].z) - Math.abs(a[1].z),
  );
  const visible = showAll ? metricRows : metricRows.slice(0, 4);

  return (
    <div className="space-y-4">
      {/* ---- view / plane ------------------------------------------------ */}
      <div className="card p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold">Cross-section</h3>
            <p className="mt-1.5 text-[0.9375rem] font-medium capitalize text-[var(--accent)]">
              {anatomy.view.plane} plane
            </p>
            <p className="mt-0.5 text-[0.8125rem] text-[var(--text-secondary)]">
              {anatomy.view.estimated_level}
            </p>
          </div>
          <span
            className="chip"
            style={
              anatomy.view.consistent_with_axial
                ? undefined
                : { borderColor: "var(--warning)" }
            }
          >
            {anatomy.view.consistent_with_axial ? "consistent" : "atypical"} ·{" "}
            {(anatomy.view.plane_confidence * 100).toFixed(0)}%
          </span>
        </div>

        <dl className="mt-4 grid grid-cols-3 gap-3 border-t border-[var(--border)] pt-3.5">
          {Object.entries(anatomy.view.signals).map(([k, v]) => (
            <div key={k}>
              <dt className="text-[0.6875rem] uppercase tracking-wide text-[var(--text-muted)]">
                {k.replace(/_/g, " ")}
              </dt>
              <dd className="tnum mt-0.5 text-sm font-medium">{v}</dd>
            </div>
          ))}
        </dl>

        <p className="mt-3 text-[0.6875rem] leading-relaxed text-[var(--text-muted)]">
          {anatomy.view.limitation}
        </p>
      </div>

      {/* ---- morphometry ------------------------------------------------- */}
      <div className="card p-5">
        <h3 className="text-sm font-semibold">Morphometry</h3>
        <p className="mt-1.5 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
          Measured from the segmented slice and z-scored against the
          non-demented reference cohort. Bars grow from the reference mean.
        </p>

        <ul className="mt-4 space-y-3">
          {visible.map(([key, s]) => {
            const info = anatomy.metric_info[key];
            const abnormal = Math.abs(s.z) >= 1.5;
            return (
              <li key={key}>
                <div className="flex items-baseline justify-between gap-3">
                  <span
                    className={`text-[0.8125rem] ${abnormal ? "font-medium" : "text-[var(--text-secondary)]"}`}
                  >
                    {info?.label ?? key}
                  </span>
                  <span className="tnum shrink-0 text-[0.8125rem]">
                    {info?.unit === "fraction"
                      ? `${(s.value * 100).toFixed(1)}%`
                      : s.value.toFixed(3)}
                    <span
                      className="ml-2 text-[0.75rem]"
                      style={{
                        color: abnormal
                          ? "var(--text-primary)"
                          : "var(--text-muted)",
                      }}
                    >
                      z {s.z >= 0 ? "+" : ""}
                      {s.z.toFixed(2)}
                    </span>
                  </span>
                </div>
                <div className="mt-1.5">
                  <ZBar z={s.z} />
                </div>
                <p className="mt-1 text-[0.6875rem] text-[var(--text-muted)]">
                  reference{" "}
                  {info?.unit === "fraction"
                    ? `${(s.reference_mean * 100).toFixed(1)}%`
                    : s.reference_mean.toFixed(3)}{" "}
                  ± {(s.reference_sd * (info?.unit === "fraction" ? 100 : 1)).toFixed(
                    info?.unit === "fraction" ? 1 : 3,
                  )}
                  {info?.unit === "fraction" ? "pp" : ""}
                </p>
              </li>
            );
          })}
        </ul>

        {metricRows.length > 4 && (
          <button
            className="btn btn-ghost mt-3 w-full !py-1.5 text-[0.75rem]"
            onClick={() => setShowAll((v) => !v)}
          >
            {showAll
              ? "Show fewer"
              : `Show all ${metricRows.length} indices`}
          </button>
        )}
      </div>

      {/* ---- findings ---------------------------------------------------- */}
      {report && (
        <div className="card p-5">
          <h3 className="text-sm font-semibold">Findings</h3>
          <p className="mt-1.5 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
            {report.summary}
          </p>

          {report.findings.length > 0 && (
            <ul className="mt-4 space-y-3">
              {report.findings.map((f) => {
                const sev = SEVERITY[f.severity] ?? SEVERITY.borderline;
                return (
                  <li
                    key={f.metric}
                    className="rounded-xl border border-[var(--border)] bg-[var(--surface-2)] p-3.5"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className="rounded-md px-1.5 py-0.5 text-[0.6875rem] font-medium"
                        style={{
                          background: `color-mix(in srgb, ${sev.colour} 18%, transparent)`,
                          color: sev.colour,
                        }}
                      >
                        {sev.label}
                      </span>
                      <span className="text-[0.875rem] font-medium">
                        {f.title}
                      </span>
                      <span className="text-[0.75rem] text-[var(--text-muted)]">
                        {f.region}
                      </span>
                    </div>
                    <p className="mt-2 text-[0.8125rem] leading-relaxed">
                      {f.measurement}
                    </p>
                    <p className="mt-1.5 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
                      {f.interpretation}
                    </p>
                    <details className="mt-2 group">
                      <summary className="cursor-pointer text-[0.75rem] text-[var(--accent)] marker:content-['']">
                        Why this happens →
                      </summary>
                      <p className="mt-1.5 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
                        {f.rationale}
                      </p>
                      <p className="mt-1.5 text-[0.6875rem] text-[var(--text-muted)]">
                        {f.references.join(", ")}
                      </p>
                    </details>
                  </li>
                );
              })}
            </ul>
          )}

          {report.attention.length > 0 && (
            <>
              <h4 className="mt-5 text-[0.75rem] font-medium uppercase tracking-wide text-[var(--text-muted)]">
                Where the model looked
              </h4>
              <ul className="mt-2.5 space-y-2.5">
                {report.attention.map((a) => (
                  <li
                    key={a.region}
                    className="rounded-xl border border-[var(--border)] p-3.5"
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="text-[0.875rem] font-medium">
                        {a.region}
                      </span>
                      <span className="tnum shrink-0 text-[0.75rem] text-[var(--text-secondary)]">
                        {(a.attention_share * 100).toFixed(1)}% ·{" "}
                        {a.attention_density.toFixed(1)}×
                      </span>
                    </div>
                    <div
                      className="mt-2 h-1.5 w-full overflow-hidden rounded-full"
                      style={{ background: "var(--surface-3)" }}
                      aria-hidden="true"
                    >
                      <span
                        className="block h-full rounded-full"
                        style={{
                          width: `${Math.min(100, a.attention_share * 100)}%`,
                          background: "var(--accent)",
                        }}
                      />
                    </div>
                    <p className="mt-2 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
                      {a.anatomy}
                    </p>
                    <details className="mt-2">
                      <summary className="cursor-pointer text-[0.75rem] text-[var(--accent)] marker:content-['']">
                        Why this region matters →
                      </summary>
                      <p className="mt-1.5 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
                        {a.rationale}
                      </p>
                    </details>
                  </li>
                ))}
              </ul>
            </>
          )}

          <button
            className="btn btn-ghost mt-4 w-full !py-1.5 text-[0.75rem]"
            onClick={() => setShowRefs((v) => !v)}
          >
            {showRefs ? "Hide" : "Show"} {Object.keys(report.references).length}{" "}
            references
          </button>
          {showRefs && (
            <ol className="mt-3 space-y-2.5 border-t border-[var(--border)] pt-3.5">
              {Object.entries(report.references).map(([key, ref]) => (
                <li key={key} className="text-[0.75rem] leading-relaxed">
                  <span className="font-medium">{ref.citation}</span>
                  <br />
                  <span className="text-[var(--text-muted)]">{ref.note}</span>
                </li>
              ))}
            </ol>
          )}

          <p className="mt-4 border-t border-[var(--border)] pt-3.5 text-[0.6875rem] leading-relaxed text-[var(--text-muted)]">
            {report.disclaimer}
          </p>
          <p className="mt-2 text-[0.6875rem] leading-relaxed text-[var(--text-muted)]">
            {anatomy.atlas_note} Laterality: {anatomy.convention}.
          </p>
        </div>
      )}
    </div>
  );
}
