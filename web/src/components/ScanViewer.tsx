"use client";

import { useState } from "react";

type View = "overlay" | "original" | "heatmap";

/**
 * Side-by-side is the wrong affordance for a saliency map — the reader has to
 * mentally register two images. A single frame with an opacity slider lets
 * them fade the evidence in and out over the anatomy instead.
 */
export default function ScanViewer({
  original,
  overlay,
  heatmap,
}: {
  original: string;
  overlay: string | null;
  heatmap: string | null;
}) {
  const [view, setView] = useState<View>(overlay ? "overlay" : "original");
  const [strength, setStrength] = useState(100);

  const tabs: { key: View; label: string; available: boolean }[] = [
    { key: "overlay", label: "Overlay", available: Boolean(overlay) },
    { key: "original", label: "Scan", available: true },
    { key: "heatmap", label: "Activation", available: Boolean(heatmap) },
  ];

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div
          className="inline-flex rounded-lg border border-[var(--border)] bg-[var(--surface-2)] p-0.5"
          role="tablist"
          aria-label="Scan view"
        >
          {tabs
            .filter((t) => t.available)
            .map((t) => (
              <button
                key={t.key}
                role="tab"
                aria-selected={view === t.key}
                onClick={() => setView(t.key)}
                className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                  view === t.key
                    ? "bg-[var(--surface)] text-[var(--text-primary)] shadow-sm"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                }`}
              >
                {t.label}
              </button>
            ))}
        </div>

        {view === "overlay" && overlay && (
          <label className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
            <span>Intensity</span>
            <input
              type="range"
              min={0}
              max={100}
              value={strength}
              onChange={(e) => setStrength(Number(e.target.value))}
              className="h-1 w-24 cursor-pointer accent-[var(--accent)]"
              aria-label="Overlay intensity"
            />
            <span className="tnum w-8 text-right">{strength}%</span>
          </label>
        )}
      </div>

      <div
        className="relative aspect-square w-full overflow-hidden rounded-xl border border-[var(--border)]"
        style={{ background: "#0b0b0b" }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={view === "heatmap" && heatmap ? heatmap : original}
          alt={
            view === "heatmap"
              ? "Class activation map"
              : "Uploaded brain MRI slice"
          }
          className="absolute inset-0 h-full w-full object-contain"
        />
        {view === "overlay" && overlay && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={overlay}
            alt="Class activation map blended over the scan"
            className="absolute inset-0 h-full w-full object-contain transition-opacity duration-200"
            style={{ opacity: strength / 100 }}
          />
        )}
      </div>

      {view !== "original" && (
        <div className="mt-3 flex items-center gap-3">
          <div
            className="h-1.5 flex-1 rounded-full"
            style={{
              background:
                "linear-gradient(90deg, rgb(92,34,12), rgb(176,62,22), rgb(235,104,52), rgb(255,198,128))",
            }}
            aria-hidden="true"
          />
          <span className="text-[0.6875rem] text-[var(--text-muted)]">
            low → high influence
          </span>
        </div>
      )}

      <p className="mt-2.5 text-[0.75rem] leading-relaxed text-[var(--text-muted)]">
        {view === "original"
          ? "The scan as uploaded, resized to the model's 224×224 input."
          : "Regions the classifier weighted most heavily for the predicted stage. " +
            "A class activation map shows where the model looked — not that the " +
            "highlighted tissue is abnormal."}
      </p>
    </div>
  );
}
