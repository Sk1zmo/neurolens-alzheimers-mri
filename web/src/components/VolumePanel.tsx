"use client";

import { CLASSES } from "@/lib/classes";
import type { InputFormat, VolumeInfo } from "@/lib/types";

const FORMAT_LABEL: Record<string, string> = {
  dicom: "DICOM instance",
  "dicom-series": "DICOM series",
  nifti: "NIfTI volume",
  image: "Image file",
};

/**
 * Slice-selection profile across the volume. This is a magnitude-over-position
 * curve, so it gets one hue with the chosen slice marked — colouring slices
 * categorically would imply they are distinct series rather than one sweep.
 */
function SliceProfile({
  scores,
  selected,
}: {
  scores: number[];
  selected: number;
}) {
  const w = 320;
  const h = 54;
  const max = Math.max(...scores, 1e-6);
  const pts = scores
    .map((s, i) => {
      const x = (i / Math.max(1, scores.length - 1)) * w;
      const y = h - (s / max) * (h - 6) - 3;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const selX = (selected / Math.max(1, scores.length - 1)) * w;
  const selY = h - (scores[selected] / max) * (h - 6) - 3;

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="w-full"
      role="img"
      aria-label={`Slice suitability across ${scores.length} slices; slice ${selected} selected`}
    >
      <path d={pts} fill="none" stroke="var(--text-muted)" strokeWidth="1.4" />
      <line
        x1={selX}
        x2={selX}
        y1={0}
        y2={h}
        stroke="var(--accent)"
        strokeWidth="1"
        strokeDasharray="3 2"
      />
      <circle
        cx={selX}
        cy={selY}
        r="4"
        fill="var(--accent)"
        stroke="var(--surface)"
        strokeWidth="1.8"
      />
    </svg>
  );
}

export default function VolumePanel({
  volume,
  input,
}: {
  volume: VolumeInfo | null;
  input: InputFormat | undefined;
}) {
  if (!input) return null;
  const hasWarnings = input.warnings.length > 0;
  if (input.source === "image" && !hasWarnings) return null;

  return (
    <div className="card p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold">Input</h3>
          <p className="mt-1.5 text-[0.9375rem] font-medium text-[var(--accent)]">
            {FORMAT_LABEL[input.source] ?? input.source}
            {input.is_volume && ` · ${input.n_slices} slices`}
          </p>
        </div>
        {input.modality !== "unknown" && (
          <span className="chip">{input.modality}</span>
        )}
      </div>

      {hasWarnings && (
        <ul className="mt-3 space-y-2">
          {input.warnings.map((w) => (
            <li
              key={w}
              className="flex gap-2 rounded-lg p-2.5 text-[0.8125rem] leading-relaxed"
              style={{
                background: "color-mix(in srgb, var(--warning) 10%, transparent)",
                color: "var(--text-secondary)",
              }}
            >
              <span aria-hidden="true" style={{ color: "var(--warning)" }}>
                ⚠
              </span>
              <span>{w}</span>
            </li>
          ))}
        </ul>
      )}

      {volume && volume.slice_scores && (
        <div className="mt-4 border-t border-[var(--border)] pt-3.5">
          <div className="flex items-baseline justify-between gap-2">
            <p className="text-[0.75rem] font-medium uppercase tracking-wide text-[var(--text-muted)]">
              Slice selection
            </p>
            <p className="tnum text-[0.75rem] text-[var(--text-secondary)]">
              slice {volume.selected_index} of {volume.n_slices}
            </p>
          </div>
          <div className="mt-2">
            <SliceProfile
              scores={volume.slice_scores}
              selected={volume.selected_index}
            />
          </div>
          <p className="mt-1 text-[0.6875rem] leading-relaxed text-[var(--text-muted)]">
            Each slice is scored for how closely it matches the axial level the
            model was trained on — a large centred cross-section, strong mirror
            symmetry, and visible central CSF. The peak is used.
          </p>

          <dl className="mt-3.5 grid grid-cols-2 gap-3">
            <div>
              <dt className="text-[0.6875rem] uppercase tracking-wide text-[var(--text-muted)]">
                Averaged over
              </dt>
              <dd className="tnum mt-0.5 text-sm font-medium">
                {volume.aggregated_over} slices
              </dd>
            </div>
            <div>
              <dt className="text-[0.6875rem] uppercase tracking-wide text-[var(--text-muted)]">
                Slice agreement
              </dt>
              <dd className="tnum mt-0.5 text-sm font-medium">
                {(volume.slice_agreement * 100).toFixed(0)}%
              </dd>
            </div>
          </dl>

          {volume.slice_agreement < 0.6 && (
            <p
              className="mt-2.5 text-[0.75rem] leading-relaxed"
              style={{ color: "var(--serious)" }}
            >
              Neighbouring slices disagree about the stage. Treat the aggregate
              as unstable — a single slice from this volume would have given a
              materially different answer.
            </p>
          )}

          <div className="mt-3.5">
            <p className="text-[0.6875rem] uppercase tracking-wide text-[var(--text-muted)]">
              Per-slice probabilities
            </p>
            <div className="mt-1.5 space-y-1">
              {volume.per_slice_probabilities.map((row, i) => (
                <div key={i} className="flex items-center gap-1.5">
                  <span className="tnum w-6 shrink-0 text-[0.6875rem] text-[var(--text-muted)]">
                    {volume.selected_index -
                      Math.floor(volume.per_slice_probabilities.length / 2) +
                      i}
                  </span>
                  <span className="flex h-2 flex-1 overflow-hidden rounded-full">
                    {row.map((p, j) => (
                      <span
                        key={j}
                        style={{
                          width: `${p * 100}%`,
                          background: [
                            "var(--series-1)",
                            "var(--series-2)",
                            "var(--series-3)",
                            "var(--series-4)",
                          ][j],
                        }}
                      />
                    ))}
                  </span>
                </div>
              ))}
            </div>
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1">
              {CLASSES.map((c, j) => (
                <span
                  key={c.id}
                  className="flex items-center gap-1.5 text-[0.6875rem] text-[var(--text-secondary)]"
                >
                  <span
                    className="h-2 w-2 rounded-sm"
                    style={{
                      background: [
                        "var(--series-1)",
                        "var(--series-2)",
                        "var(--series-3)",
                        "var(--series-4)",
                      ][j],
                    }}
                  />
                  {c.short}
                </span>
              ))}
            </div>
          </div>

          <p className="mt-3 text-[0.6875rem] leading-relaxed text-[var(--text-muted)]">
            {volume.note}
          </p>
        </div>
      )}
    </div>
  );
}
