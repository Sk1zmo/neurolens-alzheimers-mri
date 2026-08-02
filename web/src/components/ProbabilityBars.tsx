"use client";

import { useEffect, useState } from "react";
import { CLASSES } from "@/lib/classes";

/**
 * Emphasis form, not categorical: exactly one class is the answer and the rest
 * are context, so the predicted class carries the accent hue and everything
 * else recedes to muted ink. Painting four categorical hues here would make
 * the reader hunt for which bar is the point.
 *
 * Every bar is directly labelled, so identity never rests on color.
 */
export default function ProbabilityBars({
  probabilities,
  predictedId,
  compact = false,
}: {
  probabilities: number[];
  predictedId: number;
  compact?: boolean;
}) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const t = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(t);
  }, []);

  return (
    <div role="table" aria-label="Predicted probability by stage">
      <div className="sr-only" role="row">
        <span role="columnheader">Stage</span>
        <span role="columnheader">Probability</span>
      </div>
      <ul className={compact ? "space-y-1.5" : "space-y-2.5"}>
        {CLASSES.map((c) => {
          const p = probabilities[c.id] ?? 0;
          const isPredicted = c.id === predictedId;
          return (
            <li
              key={c.id}
              role="row"
              className="grid grid-cols-[minmax(72px,auto)_1fr_auto] items-center gap-3"
            >
              <span
                role="cell"
                className={`text-[0.8125rem] ${
                  isPredicted
                    ? "font-medium text-[var(--text-primary)]"
                    : "text-[var(--text-secondary)]"
                }`}
              >
                {compact ? c.short : c.label}
              </span>

              <span
                className="relative block overflow-hidden rounded-full"
                style={{
                  height: compact ? 6 : 8,
                  background: "var(--surface-3)",
                }}
                aria-hidden="true"
              >
                <span
                  className="absolute inset-y-0 left-0 rounded-full"
                  style={{
                    width: mounted ? `${Math.max(p * 100, p > 0 ? 1.5 : 0)}%` : "0%",
                    background: isPredicted
                      ? "var(--accent)"
                      : "var(--text-muted)",
                    opacity: isPredicted ? 1 : 0.4,
                    transition:
                      "width 620ms cubic-bezier(0.22, 1, 0.36, 1)",
                  }}
                />
              </span>

              <span
                role="cell"
                className={`tnum w-14 text-right text-[0.8125rem] ${
                  isPredicted
                    ? "font-medium text-[var(--text-primary)]"
                    : "text-[var(--text-secondary)]"
                }`}
              >
                {(p * 100).toFixed(1)}%
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
