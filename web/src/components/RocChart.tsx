"use client";

import { useMemo, useRef, useState } from "react";
import { CLASSES } from "@/lib/classes";
import type { ModelMetrics } from "@/lib/types";

const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];

const W = 560;
const H = 420;
const PAD = { top: 16, right: 18, bottom: 40, left: 46 };
const PW = W - PAD.left - PAD.right;
const PH = H - PAD.top - PAD.bottom;

const sx = (v: number) => PAD.left + v * PW;
const sy = (v: number) => PAD.top + (1 - v) * PH;

/** Step-interpolate a ROC curve at a given false-positive rate. */
function tprAt(fpr: number[], tpr: number[], x: number): number {
  if (fpr.length === 0) return 0;
  if (x <= fpr[0]) return tpr[0];
  for (let i = 1; i < fpr.length; i++) {
    if (fpr[i] >= x) {
      const span = fpr[i] - fpr[i - 1];
      const t = span === 0 ? 0 : (x - fpr[i - 1]) / span;
      return tpr[i - 1] + t * (tpr[i] - tpr[i - 1]);
    }
  }
  return tpr[tpr.length - 1];
}

export default function RocChart({ metrics }: { metrics: ModelMetrics }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);

  const curves = useMemo(
    () =>
      CLASSES.map((c) => {
        const r = metrics.roc[String(c.id)];
        const fpr = r?.fpr ?? [0, 1];
        const tpr = r?.tpr ?? [0, 1];
        return {
          ...c,
          color: SERIES[c.id],
          auc: r?.auc ?? NaN,
          fpr,
          tpr,
          path: fpr
            .map((f, i) => `${i === 0 ? "M" : "L"}${sx(f).toFixed(2)},${sy(tpr[i]).toFixed(2)}`)
            .join(" "),
        };
      }),
    [metrics],
  );

  function onMove(e: React.PointerEvent<SVGSVGElement>) {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const v = (px - PAD.left) / PW;
    setHoverX(v >= 0 && v <= 1 ? v : null);
  }

  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full touch-none"
        role="img"
        aria-label="One-vs-rest ROC curves for each Alzheimer's stage on the held-out test set"
        onPointerMove={onMove}
        onPointerLeave={() => setHoverX(null)}
      >
        {ticks.map((t) => (
          <g key={`g${t}`}>
            <line className="grid-line" x1={sx(0)} x2={sx(1)} y1={sy(t)} y2={sy(t)} />
            <line className="grid-line" x1={sx(t)} x2={sx(t)} y1={sy(0)} y2={sy(1)} />
          </g>
        ))}

        {/* chance reference */}
        <line
          x1={sx(0)}
          y1={sy(0)}
          x2={sx(1)}
          y2={sy(1)}
          stroke="var(--axis)"
          strokeWidth="1.5"
          strokeDasharray="4 4"
        />
        <text
          className="tick"
          x={sx(0.72)}
          y={sy(0.68)}
          transform={`rotate(-45 ${sx(0.72)} ${sy(0.68)})`}
        >
          chance
        </text>

        <line className="axis-line" x1={sx(0)} x2={sx(1)} y1={sy(0)} y2={sy(0)} />
        <line className="axis-line" x1={sx(0)} x2={sx(0)} y1={sy(0)} y2={sy(1)} />

        {ticks.map((t) => (
          <g key={`t${t}`}>
            <text className="tick" x={sx(t)} y={sy(0) + 16} textAnchor="middle">
              {t}
            </text>
            <text className="tick" x={sx(0) - 8} y={sy(t) + 3.5} textAnchor="end">
              {t}
            </text>
          </g>
        ))}

        <text
          className="tick"
          x={PAD.left + PW / 2}
          y={H - 6}
          textAnchor="middle"
          style={{ fontSize: 11.5 }}
        >
          False positive rate
        </text>
        <text
          className="tick"
          x={-(PAD.top + PH / 2)}
          y={13}
          textAnchor="middle"
          transform="rotate(-90)"
          style={{ fontSize: 11.5 }}
        >
          True positive rate
        </text>

        {curves.map((c) => (
          <path
            key={c.id}
            d={c.path}
            fill="none"
            stroke={c.color}
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity={hoverX === null ? 1 : 0.92}
          />
        ))}

        {hoverX !== null && (
          <g pointerEvents="none">
            <line
              x1={sx(hoverX)}
              x2={sx(hoverX)}
              y1={sy(0)}
              y2={sy(1)}
              stroke="var(--text-muted)"
              strokeWidth="1"
            />
            {curves.map((c) => {
              const y = tprAt(c.fpr, c.tpr, hoverX);
              return (
                <circle
                  key={c.id}
                  cx={sx(hoverX)}
                  cy={sy(y)}
                  r={4}
                  fill={c.color}
                  stroke="var(--surface)"
                  strokeWidth={2}
                />
              );
            })}
          </g>
        )}
      </svg>

      {/* Legend doubles as the table view — required relief for the low-contrast
          light-mode series, and more useful than crowding labels onto curves
          that bunch in the top-left corner. */}
      <div className="mt-3 overflow-x-auto scroll-thin">
        <table className="w-full min-w-[380px] text-[0.8125rem]">
          <thead>
            <tr className="text-left text-[var(--text-muted)]">
              <th className="pb-1.5 font-medium">Stage</th>
              <th className="pb-1.5 text-right font-medium">AUC</th>
              {hoverX !== null && (
                <th className="pb-1.5 text-right font-medium tnum">
                  TPR @ FPR {hoverX.toFixed(2)}
                </th>
              )}
              <th className="pb-1.5 text-right font-medium">Recall</th>
              <th className="pb-1.5 text-right font-medium">Precision</th>
            </tr>
          </thead>
          <tbody>
            {curves.map((c) => {
              const pc = metrics.per_class[c.label];
              return (
                <tr key={c.id} className="border-t border-[var(--border)]">
                  <td className="py-1.5">
                    <span className="flex items-center gap-2">
                      <span
                        className="h-2.5 w-2.5 shrink-0 rounded-sm"
                        style={{ background: c.color }}
                        aria-hidden="true"
                      />
                      {c.label}
                    </span>
                  </td>
                  <td className="tnum py-1.5 text-right font-medium">
                    {Number.isFinite(c.auc) ? c.auc.toFixed(3) : "—"}
                  </td>
                  {hoverX !== null && (
                    <td className="tnum py-1.5 text-right text-[var(--text-secondary)]">
                      {tprAt(c.fpr, c.tpr, hoverX).toFixed(3)}
                    </td>
                  )}
                  <td className="tnum py-1.5 text-right text-[var(--text-secondary)]">
                    {pc ? `${(pc.recall * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td className="tnum py-1.5 text-right text-[var(--text-secondary)]">
                    {pc ? `${(pc.precision * 100).toFixed(1)}%` : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
