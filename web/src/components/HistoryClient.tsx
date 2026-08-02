"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import ProbabilityBars from "@/components/ProbabilityBars";
import { CLASSES, classById } from "@/lib/classes";
import { fullTime, relativeTime } from "@/lib/format";
import { getSessionId } from "@/lib/session";
import type { ScanRecord } from "@/lib/types";

export default function HistoryClient() {
  const [scans, setScans] = useState<ScanRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<number | "all">("all");
  const [open, setOpen] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/scans", {
        headers: { "x-session-id": getSessionId() },
        cache: "no-store",
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error ?? "Could not load history.");
      setScans(data.scans as ScanRecord[]);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load history.");
      setScans([]);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function remove(id: string) {
    setBusy(id);
    try {
      const res = await fetch(`/api/scans/${id}`, {
        method: "DELETE",
        headers: { "x-session-id": getSessionId() },
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error ?? "Delete failed.");
      setScans((prev) => (prev ?? []).filter((s) => s.id !== id));
      if (open === id) setOpen(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed.");
    } finally {
      setBusy(null);
    }
  }

  const visible = useMemo(
    () =>
      (scans ?? []).filter(
        (s) => filter === "all" || s.predicted_class_id === filter,
      ),
    [scans, filter],
  );

  const counts = useMemo(() => {
    const c = new Array(CLASSES.length).fill(0);
    for (const s of scans ?? []) c[s.predicted_class_id] += 1;
    return c;
  }, [scans]);

  if (scans === null) {
    return (
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3" aria-busy="true">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <div
            key={i}
            className="card h-56 pulse-soft"
            style={{ animationDelay: `${i * 90}ms` }}
          />
        ))}
      </div>
    );
  }

  if (error && scans.length === 0) {
    return (
      <div className="card p-6">
        <p className="text-sm font-medium">History is unavailable</p>
        <p className="mt-2 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
          {error}
        </p>
      </div>
    );
  }

  if (scans.length === 0) {
    return (
      <div className="card flex flex-col items-center p-10 text-center">
        <div
          className="mb-4 flex h-12 w-12 items-center justify-center rounded-full"
          style={{ background: "var(--surface-2)" }}
          aria-hidden="true"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <rect
              x="4"
              y="4"
              width="16"
              height="16"
              rx="2.5"
              stroke="var(--text-muted)"
              strokeWidth="1.6"
            />
            <path
              d="M4 15.5 9 11l4 3.5L16 12l4 3.5"
              stroke="var(--text-muted)"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <p className="text-sm font-medium">Nothing saved yet</p>
        <p className="mt-1.5 max-w-sm text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
          Analyze a scan and choose &ldquo;Save to history&rdquo; and it will show up
          here.
        </p>
        <Link href="/" className="btn btn-primary mt-5">
          Analyze a scan
        </Link>
      </div>
    );
  }

  return (
    <div>
      {/* Filters sit in one row above the content. */}
      <div className="mb-5 flex flex-wrap items-center gap-1.5">
        <button
          onClick={() => setFilter("all")}
          className={`chip ${filter === "all" ? "!border-[var(--accent)] !text-[var(--text-primary)]" : ""}`}
        >
          All <span className="tnum">{scans.length}</span>
        </button>
        {CLASSES.map((c) => (
          <button
            key={c.id}
            onClick={() => setFilter(c.id)}
            disabled={counts[c.id] === 0}
            className={`chip disabled:opacity-40 ${
              filter === c.id ? "!border-[var(--accent)] !text-[var(--text-primary)]" : ""
            }`}
          >
            {c.short} <span className="tnum">{counts[c.id]}</span>
          </button>
        ))}
      </div>

      {error && (
        <p className="mb-4 text-[0.8125rem]" style={{ color: "var(--critical)" }} role="alert">
          {error}
        </p>
      )}

      <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {visible.map((scan) => {
          const info = classById(scan.predicted_class_id);
          const isOpen = open === scan.id;
          return (
            <li key={scan.id} className="card overflow-hidden">
              <div
                className="relative aspect-square w-full"
                style={{ background: "#0b0b0b" }}
              >
                {scan.imageUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={scan.imageUrl}
                    alt={`Saved MRI slice predicted ${scan.predicted_label}`}
                    className="absolute inset-0 h-full w-full object-contain"
                    loading="lazy"
                  />
                ) : (
                  <div className="absolute inset-0 grid place-items-center text-[0.75rem] text-[var(--text-muted)]">
                    image unavailable
                  </div>
                )}
                {scan.out_of_distribution && (
                  <span
                    className="absolute left-2 top-2 rounded-md px-1.5 py-0.5 text-[0.6875rem] font-medium"
                    style={{ background: "var(--warning)", color: "#3a2703" }}
                  >
                    out of distribution
                  </span>
                )}
              </div>

              <div className="p-4">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="truncate text-[0.9375rem] font-medium leading-tight">
                      {info.label}
                    </p>
                    <p className="tnum mt-0.5 text-[0.75rem] text-[var(--text-secondary)]">
                      {(scan.confidence * 100).toFixed(1)}% confidence
                    </p>
                  </div>
                  <time
                    className="shrink-0 text-[0.6875rem] text-[var(--text-muted)]"
                    dateTime={scan.created_at}
                    title={fullTime(scan.created_at)}
                  >
                    {relativeTime(scan.created_at)}
                  </time>
                </div>

                {scan.note && (
                  <p className="mt-2 line-clamp-2 text-[0.75rem] leading-relaxed text-[var(--text-secondary)]">
                    {scan.note}
                  </p>
                )}

                {isOpen && (
                  <div className="mt-3 border-t border-[var(--border)] pt-3">
                    <ProbabilityBars
                      probabilities={scan.probabilities}
                      predictedId={scan.predicted_class_id}
                      compact
                    />
                    <dl className="mt-3 space-y-1 text-[0.6875rem] text-[var(--text-muted)]">
                      {scan.original_filename && (
                        <div className="flex justify-between gap-2">
                          <dt>File</dt>
                          <dd className="truncate">{scan.original_filename}</dd>
                        </div>
                      )}
                      {scan.model_version && (
                        <div className="flex justify-between gap-2">
                          <dt>Model</dt>
                          <dd>v{scan.model_version}</dd>
                        </div>
                      )}
                      <div className="flex justify-between gap-2">
                        <dt>Analyzed</dt>
                        <dd>{fullTime(scan.created_at)}</dd>
                      </div>
                    </dl>
                  </div>
                )}

                <div className="mt-3 flex gap-1.5">
                  <button
                    className="btn btn-ghost flex-1 !py-1.5 text-[0.75rem]"
                    onClick={() => setOpen(isOpen ? null : scan.id)}
                    aria-expanded={isOpen}
                  >
                    {isOpen ? "Less" : "Details"}
                  </button>
                  <button
                    className="btn btn-ghost !py-1.5 text-[0.75rem]"
                    onClick={() => void remove(scan.id)}
                    disabled={busy === scan.id}
                    style={{ color: "var(--critical)" }}
                  >
                    {busy === scan.id ? "Deleting…" : "Delete"}
                  </button>
                </div>
              </div>
            </li>
          );
        })}
      </ul>

      <p className="mt-6 text-[0.75rem] leading-relaxed text-[var(--text-muted)]">
        Deleting a scan removes it from this list and deletes the stored image.
        If you contributed it to the retraining pool and it has already been
        reviewed, the labelled record is kept so the training corpus stays
        reproducible — the note on{" "}
        <Link href="/about" className="text-[var(--accent)] hover:underline">
          how the data is used
        </Link>{" "}
        spells out exactly what that means.
      </p>
    </div>
  );
}
