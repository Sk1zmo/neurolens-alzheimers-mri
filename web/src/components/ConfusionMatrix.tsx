"use client";

import { useState } from "react";
import { CLASSES } from "@/lib/classes";

/**
 * A grid of magnitudes: sequential encoding, one hue, light to dark. Cells are
 * normalised per row so the colour answers "what fraction of this true class
 * landed here" — raw counts would paint the whole matrix by class frequency
 * (NonDemented has 50x the support of ModerateDemented) and hide every error.
 */
const RAMP = [
  "var(--seq-100)",
  "var(--seq-250)",
  "var(--seq-400)",
  "var(--seq-550)",
  "var(--seq-700)",
];

function rampStep(v: number): string {
  if (v < 0.02) return "var(--surface-2)";
  const i = Math.min(RAMP.length - 1, Math.floor(v * RAMP.length));
  return RAMP[i];
}

export default function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);

  const rowTotals = matrix.map((row) => row.reduce((a, b) => a + b, 0));

  return (
    <div>
      <div className="overflow-x-auto scroll-thin">
        <table className="w-full min-w-[440px] border-separate border-spacing-0.5 text-[0.75rem]">
          <caption className="sr-only">
            Confusion matrix: rows are the true stage, columns the predicted
            stage, values are the share of that true class.
          </caption>
          <thead>
            <tr>
              <th className="p-1 text-left font-medium text-[var(--text-muted)]">
                True ╲ Predicted
              </th>
              {CLASSES.map((c) => (
                <th
                  key={c.id}
                  scope="col"
                  className="p-1 text-center font-medium text-[var(--text-muted)]"
                >
                  {c.short}
                </th>
              ))}
              <th className="p-1 text-right font-medium text-[var(--text-muted)]">n</th>
            </tr>
          </thead>
          <tbody>
            {CLASSES.map((rowClass) => {
              const total = rowTotals[rowClass.id] || 1;
              return (
                <tr key={rowClass.id}>
                  <th
                    scope="row"
                    className="whitespace-nowrap p-1 text-left font-medium text-[var(--text-secondary)]"
                  >
                    {rowClass.label}
                  </th>
                  {CLASSES.map((colClass) => {
                    const count = matrix[rowClass.id]?.[colClass.id] ?? 0;
                    const share = count / total;
                    const on = hover?.i === rowClass.id && hover?.j === colClass.id;
                    const diagonal = rowClass.id === colClass.id;
                    return (
                      <td key={colClass.id} className="p-0">
                        <div
                          className="relative flex h-12 cursor-default items-center justify-center rounded-md transition-transform"
                          style={{
                            background: rampStep(share),
                            /* 2px surface gap comes from border-spacing; the
                               ring marks the hovered cell without moving it */
                            boxShadow: on
                              ? "0 0 0 2px var(--surface), 0 0 0 3.5px var(--accent)"
                              : undefined,
                          }}
                          onMouseEnter={() =>
                            setHover({ i: rowClass.id, j: colClass.id })
                          }
                          onMouseLeave={() => setHover(null)}
                          title={`${count} of ${total} true ${rowClass.label} predicted as ${colClass.label}`}
                        >
                          <span
                            className="tnum text-[0.8125rem] font-medium"
                            style={{
                              color:
                                share >= 0.4
                                  ? "#fff"
                                  : share < 0.02
                                    ? "var(--text-muted)"
                                    : "var(--text-primary)",
                              fontWeight: diagonal ? 600 : 500,
                            }}
                          >
                            {share < 0.001 ? "—" : `${(share * 100).toFixed(0)}%`}
                          </span>
                        </div>
                      </td>
                    );
                  })}
                  <td className="tnum p-1 text-right text-[var(--text-muted)]">
                    {rowTotals[rowClass.id]}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="mt-3 flex items-center gap-3">
        <span className="text-[0.6875rem] text-[var(--text-muted)]">0%</span>
        <div className="flex h-1.5 flex-1 overflow-hidden rounded-full">
          {RAMP.map((c) => (
            <span key={c} className="flex-1" style={{ background: c }} />
          ))}
        </div>
        <span className="text-[0.6875rem] text-[var(--text-muted)]">100%</span>
        <span className="ml-1 text-[0.6875rem] text-[var(--text-muted)]">
          share of true class
        </span>
      </div>

      <p className="mt-3 min-h-[1.25rem] text-[0.75rem] text-[var(--text-secondary)]">
        {hover ? (
          <>
            <span className="tnum font-medium">
              {matrix[hover.i]?.[hover.j] ?? 0}
            </span>{" "}
            of {rowTotals[hover.i]} true{" "}
            <span className="font-medium">{CLASSES[hover.i].label}</span> scans were
            predicted{" "}
            <span className="font-medium">{CLASSES[hover.j].label}</span>.
          </>
        ) : (
          "Hover a cell for counts. The diagonal is correct; off-diagonal cells to the immediate left and right are adjacent-stage confusions."
        )}
      </p>
    </div>
  );
}
