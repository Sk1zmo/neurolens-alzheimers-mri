"use client";

import { useEffect, useState } from "react";
import { CLASSES } from "@/lib/classes";
import type { RetrainingStats, ScanStats } from "@/lib/types";

interface StatsResponse {
  ok: boolean;
  configured: boolean;
  scans: ScanStats | null;
  retraining: RetrainingStats | null;
  byClass: { id: number; label: string; short: string; count: number }[];
}

export default function CorpusPanel() {
  const [data, setData] = useState<StatsResponse | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    fetch("/api/stats")
      .then((r) => r.json())
      .then(setData)
      .catch(() => setFailed(true));
  }, []);

  if (failed) {
    return (
      <p className="text-[0.8125rem] text-[var(--text-secondary)]">
        Could not load corpus statistics.
      </p>
    );
  }

  if (!data) {
    return (
      <div className="space-y-2" aria-busy="true">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="h-8 rounded-lg pulse-soft"
            style={{ background: "var(--surface-2)", animationDelay: `${i * 120}ms` }}
          />
        ))}
      </div>
    );
  }

  if (!data.configured) {
    return (
      <p className="text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
        Supabase is not configured for this deployment, so uploads are not being
        collected. Set <code className="rounded bg-[var(--surface-2)] px-1 py-0.5 text-[0.75rem]">SUPABASE_URL</code>{" "}
        and{" "}
        <code className="rounded bg-[var(--surface-2)] px-1 py-0.5 text-[0.75rem]">
          SUPABASE_SERVICE_ROLE_KEY
        </code>{" "}
        and run <code className="rounded bg-[var(--surface-2)] px-1 py-0.5 text-[0.75rem]">supabase/schema.sql</code>.
      </p>
    );
  }

  const total = data.byClass.reduce((a, b) => a + b.count, 0);
  const max = Math.max(1, ...data.byClass.map((c) => c.count));
  const r = data.retraining;

  return (
    <div className="space-y-5">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
        {[
          { k: "Scans uploaded", v: data.scans?.total_scans ?? 0 },
          { k: "Queued for review", v: r?.queued_total ?? 0 },
          { k: "Ready to train on", v: r?.ready_for_training ?? 0 },
          { k: "Already trained on", v: r?.already_trained ?? 0 },
        ].map((s) => (
          <div key={s.k}>
            <dt className="text-[0.6875rem] uppercase tracking-wide text-[var(--text-muted)]">
              {s.k}
            </dt>
            <dd className="tnum mt-0.5 text-lg font-semibold leading-none">{s.v}</dd>
          </div>
        ))}
      </dl>

      {total > 0 ? (
        <div>
          <p className="mb-2.5 text-[0.75rem] font-medium uppercase tracking-wide text-[var(--text-muted)]">
            Predicted stage across uploads
          </p>
          <ul className="space-y-2">
            {CLASSES.map((c) => {
              const row = data.byClass.find((b) => b.id === c.id);
              const count = row?.count ?? 0;
              return (
                <li
                  key={c.id}
                  className="grid grid-cols-[minmax(88px,auto)_1fr_auto] items-center gap-3"
                >
                  <span className="text-[0.8125rem] text-[var(--text-secondary)]">
                    {c.label}
                  </span>
                  <span
                    className="relative block h-2 overflow-hidden rounded-full"
                    style={{ background: "var(--surface-3)" }}
                    aria-hidden="true"
                  >
                    <span
                      className="absolute inset-y-0 left-0 rounded-full"
                      style={{
                        width: `${(count / max) * 100}%`,
                        background: "var(--accent)",
                        opacity: 0.85,
                      }}
                    />
                  </span>
                  <span className="tnum w-10 text-right text-[0.8125rem]">{count}</span>
                </li>
              );
            })}
          </ul>
          <p className="mt-3 text-[0.75rem] leading-relaxed text-[var(--text-muted)]">
            These are the model&apos;s own predictions on user uploads, not verified
            labels — read it as what people are uploading, never as prevalence.
          </p>
        </div>
      ) : (
        <p className="text-[0.8125rem] text-[var(--text-secondary)]">
          No scans have been contributed yet.
        </p>
      )}
    </div>
  );
}
