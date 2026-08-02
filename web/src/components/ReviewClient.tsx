"use client";

import { useCallback, useEffect, useState } from "react";
import { CLASSES, classByDir } from "@/lib/classes";
import { relativeTime } from "@/lib/format";
import type { QueueRecord, RetrainingStats } from "@/lib/types";

const TOKEN_KEY = "neurolens.reviewToken";
type Tab = "pending" | "approved" | "rejected" | "all";

export default function ReviewClient() {
  const [token, setToken] = useState("");
  const [authed, setAuthed] = useState(false);
  const [items, setItems] = useState<QueueRecord[]>([]);
  const [stats, setStats] = useState<RetrainingStats | null>(null);
  const [tab, setTab] = useState<Tab>("pending");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    const saved = sessionStorage.getItem(TOKEN_KEY);
    if (saved) {
      setToken(saved);
      setAuthed(true);
    }
  }, []);

  const load = useCallback(
    async (t: string, status: Tab) => {
      if (!t) return;
      setLoading(true);
      try {
        const res = await fetch(`/api/review?status=${status}&limit=60`, {
          headers: { "x-review-token": t },
          cache: "no-store",
        });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error ?? "Request failed.");
        setItems(data.items as QueueRecord[]);
        setStats(data.stats as RetrainingStats | null);
        setError(null);
        setAuthed(true);
        sessionStorage.setItem(TOKEN_KEY, t);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Request failed.");
        if (String(e).includes("token")) {
          setAuthed(false);
          sessionStorage.removeItem(TOKEN_KEY);
        }
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (authed && token) void load(token, tab);
  }, [authed, token, tab, load]);

  async function patch(id: string, body: Record<string, unknown>) {
    setBusy(id);
    try {
      const res = await fetch("/api/review", {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          "x-review-token": token,
        },
        body: JSON.stringify({ id, ...body }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error ?? "Update failed.");
      setItems((prev) =>
        prev.map((it) =>
          it.id === id
            ? {
                ...it,
                verified_label:
                  "verifiedLabel" in body
                    ? (body.verifiedLabel as string | null)
                    : it.verified_label,
                status: (body.status as QueueRecord["status"]) ?? it.status,
              }
            : it,
        ),
      );
      setError(null);
      void load(token, tab);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Update failed.");
    } finally {
      setBusy(null);
    }
  }

  if (!authed) {
    return (
      <div className="card mx-auto max-w-md p-6">
        <h2 className="text-sm font-semibold">Reviewer access</h2>
        <p className="mt-2 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
          Enter the value of <code className="rounded bg-[var(--surface-2)] px-1 py-0.5 text-[0.75rem]">REVIEW_TOKEN</code>{" "}
          set on the server. It is held in sessionStorage for this tab only.
        </p>
        <form
          className="mt-4 space-y-3"
          onSubmit={(e) => {
            e.preventDefault();
            void load(token, tab);
          }}
        >
          <input
            type="password"
            className="field"
            placeholder="Review token"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            autoComplete="off"
          />
          <button className="btn btn-primary w-full" disabled={!token || loading}>
            {loading ? "Checking…" : "Unlock"}
          </button>
        </form>
        {error && (
          <p className="mt-3 text-[0.8125rem]" style={{ color: "var(--critical)" }} role="alert">
            {error}
          </p>
        )}
      </div>
    );
  }

  return (
    <div>
      {stats && (
        <dl className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-5">
          {[
            { k: "Queued", v: stats.queued_total },
            { k: "Labelled", v: stats.labelled },
            { k: "Ready to train", v: stats.ready_for_training },
            { k: "Trained on", v: stats.already_trained },
            { k: "Rejected", v: stats.rejected },
          ].map((s) => (
            <div
              key={s.k}
              className="rounded-xl border border-[var(--border)] bg-[var(--surface-2)] p-3"
            >
              <dt className="text-[0.6875rem] uppercase tracking-wide text-[var(--text-muted)]">
                {s.k}
              </dt>
              <dd className="tnum mt-1 text-xl font-semibold leading-none">{s.v}</dd>
            </div>
          ))}
        </dl>
      )}

      <div className="mb-5 flex flex-wrap items-center gap-1.5">
        {(["pending", "approved", "rejected", "all"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`chip capitalize ${
              tab === t ? "!border-[var(--accent)] !text-[var(--text-primary)]" : ""
            }`}
          >
            {t}
          </button>
        ))}
        <button
          className="btn btn-ghost ml-auto !py-1.5 text-[0.75rem]"
          onClick={() => void load(token, tab)}
          disabled={loading}
        >
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {error && (
        <p className="mb-4 text-[0.8125rem]" style={{ color: "var(--critical)" }} role="alert">
          {error}
        </p>
      )}

      {items.length === 0 ? (
        <div className="card p-10 text-center">
          <p className="text-sm font-medium">Nothing in this queue</p>
          <p className="mt-1.5 text-[0.8125rem] text-[var(--text-secondary)]">
            {tab === "pending"
              ? "Every contributed scan has been reviewed."
              : "No scans with this status yet."}
          </p>
        </div>
      ) : (
        <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {items.map((item) => {
            const guess = classByDir(item.predicted_label);
            return (
              <li key={item.id} className="card overflow-hidden">
                <div className="relative aspect-square w-full" style={{ background: "#0b0b0b" }}>
                  {item.imageUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={item.imageUrl}
                      alt="Contributed scan awaiting review"
                      className="absolute inset-0 h-full w-full object-contain"
                      loading="lazy"
                    />
                  ) : (
                    <div className="absolute inset-0 grid place-items-center text-[0.75rem] text-[var(--text-muted)]">
                      image unavailable
                    </div>
                  )}
                  {item.used_in_training && (
                    <span
                      className="absolute right-2 top-2 rounded-md px-1.5 py-0.5 text-[0.6875rem] font-medium"
                      style={{ background: "var(--good)", color: "#fff" }}
                    >
                      in training set
                    </span>
                  )}
                </div>

                <div className="p-4">
                  <div className="flex items-center justify-between gap-2 text-[0.75rem]">
                    <span className="text-[var(--text-secondary)]">
                      Model guessed{" "}
                      <span className="font-medium text-[var(--text-primary)]">
                        {guess?.short ?? item.predicted_label}
                      </span>
                      {item.predicted_confidence !== null && (
                        <span className="tnum">
                          {" "}
                          ({(item.predicted_confidence * 100).toFixed(0)}%)
                        </span>
                      )}
                    </span>
                    <time className="text-[var(--text-muted)]" dateTime={item.created_at}>
                      {relativeTime(item.created_at)}
                    </time>
                  </div>

                  <p className="mt-3 text-[0.6875rem] font-medium uppercase tracking-wide text-[var(--text-muted)]">
                    Verified label
                  </p>
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {CLASSES.map((c) => (
                      <button
                        key={c.id}
                        disabled={busy === item.id || item.used_in_training}
                        onClick={() =>
                          void patch(item.id, {
                            verifiedLabel:
                              item.verified_label === c.dir ? null : c.dir,
                          })
                        }
                        className={`chip !text-[0.6875rem] disabled:opacity-50 ${
                          item.verified_label === c.dir
                            ? "!border-[var(--accent)] !bg-[var(--accent-soft)] !text-[var(--text-primary)]"
                            : ""
                        }`}
                      >
                        {c.short}
                      </button>
                    ))}
                  </div>

                  <div className="mt-3 flex gap-1.5">
                    <button
                      className="btn flex-1 !py-1.5 text-[0.75rem]"
                      disabled={
                        busy === item.id ||
                        !item.verified_label ||
                        item.status === "approved"
                      }
                      onClick={() => void patch(item.id, { status: "approved" })}
                      title={
                        item.verified_label
                          ? "Add to the training pool"
                          : "Assign a verified label first"
                      }
                    >
                      Approve
                    </button>
                    <button
                      className="btn !py-1.5 text-[0.75rem]"
                      disabled={busy === item.id || item.status === "rejected"}
                      onClick={() => void patch(item.id, { status: "rejected" })}
                      style={{ color: "var(--critical)" }}
                    >
                      Reject
                    </button>
                  </div>

                  <p className="mt-2 text-[0.6875rem] text-[var(--text-muted)]">
                    Status:{" "}
                    <span className="font-medium text-[var(--text-secondary)]">
                      {item.status}
                    </span>
                    {item.used_in_training && " · already used in a training run"}
                  </p>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      <div className="card mt-6 p-5">
        <h2 className="text-sm font-semibold">Running a retraining pass</h2>
        <p className="mt-2 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
          Once approved scans have accumulated, pull them down and fine-tune. The
          held-out test split never changes, so the before/after numbers are
          directly comparable, and a new checkpoint is only promoted if it beats
          the incumbent on macro-F1.
        </p>
        <pre className="mt-3 overflow-x-auto rounded-lg bg-[var(--surface-2)] p-3 text-[0.75rem] leading-relaxed">
{`cd training
python retrain.py --min-new 25    # downloads, retrains, evaluates, promotes
cd ../web && vercel --prod        # ship the new ONNX model`}
        </pre>
      </div>
    </div>
  );
}
