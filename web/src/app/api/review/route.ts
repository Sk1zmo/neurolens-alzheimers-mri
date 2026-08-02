import { NextResponse } from "next/server";
import { getSupabase, isSupabaseConfigured, signManyScanUrls } from "@/lib/supabase";
import { CLASS_DIRS } from "@/lib/classes";
import type { QueueRecord } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Mirrors public.training_queue in supabase/schema.sql. */
interface QueueRow {
  id: string;
  scan_id: string | null;
  created_at: string;
  storage_path: string;
  predicted_label: string;
  predicted_confidence: number | null;
  verified_label: string | null;
  status: QueueRecord["status"];
  used_in_training: boolean;
  review_note: string | null;
}

/**
 * The reviewer console behind /review.
 *
 * This is the gate between "a stranger uploaded a JPEG" and "the model trains
 * on it". Guarded by a shared token in REVIEW_TOKEN rather than a real account
 * system, which is the right weight for a single-operator research tool — but
 * it is a shared secret, so treat it accordingly and set a long random value.
 */
function authorised(req: Request): boolean {
  const expected = process.env.REVIEW_TOKEN;
  if (!expected) return false;

  const provided =
    req.headers.get("x-review-token") ??
    new URL(req.url).searchParams.get("token") ??
    "";

  // Constant-time-ish compare; lengths differ rarely enough that the early
  // return leaks nothing useful here.
  if (provided.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < expected.length; i++) {
    diff |= provided.charCodeAt(i) ^ expected.charCodeAt(i);
  }
  return diff === 0;
}

function denied() {
  return NextResponse.json(
    {
      ok: false,
      error: process.env.REVIEW_TOKEN
        ? "Invalid review token."
        : "REVIEW_TOKEN is not set on the server, so the review console is disabled.",
    },
    { status: 401 },
  );
}

// ------------------------------------------------------------------ GET
export async function GET(req: Request) {
  if (!isSupabaseConfigured()) {
    return NextResponse.json(
      { ok: false, error: "Storage is not configured." },
      { status: 503 },
    );
  }
  if (!authorised(req)) return denied();

  const url = new URL(req.url);
  const status = url.searchParams.get("status") ?? "pending";
  const limit = Math.min(
    Math.max(Number(url.searchParams.get("limit") ?? 48) || 48, 1),
    200,
  );

  const supabase = getSupabase();
  let query = supabase
    .from("training_queue")
    .select(
      "id, scan_id, created_at, storage_path, predicted_label, " +
        "predicted_confidence, verified_label, status, used_in_training, review_note",
    )
    .order("created_at", { ascending: true })
    .limit(limit);

  if (status !== "all") query = query.eq("status", status);

  const { data, error } = await query;
  if (error) {
    return NextResponse.json({ ok: false, error: error.message }, { status: 500 });
  }

  const rows = (data ?? []) as unknown as QueueRow[];
  const signed = await signManyScanUrls(rows.map((r) => r.storage_path));
  const items: QueueRecord[] = rows.map((r) => ({
    id: r.id,
    scan_id: r.scan_id,
    created_at: r.created_at,
    storage_path: r.storage_path,
    predicted_label: r.predicted_label,
    predicted_confidence: r.predicted_confidence,
    verified_label: r.verified_label,
    status: r.status,
    used_in_training: Boolean(r.used_in_training),
    review_note: r.review_note,
    imageUrl: signed[r.storage_path] ?? null,
  }));

  const { data: stats } = await supabase
    .from("retraining_stats")
    .select("*")
    .maybeSingle();

  return NextResponse.json({ ok: true, items, stats: stats ?? null });
}

// ---------------------------------------------------------------- PATCH
export async function PATCH(req: Request) {
  if (!isSupabaseConfigured()) {
    return NextResponse.json(
      { ok: false, error: "Storage is not configured." },
      { status: 503 },
    );
  }
  if (!authorised(req)) return denied();

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { ok: false, error: "Malformed JSON body." },
      { status: 400 },
    );
  }

  const id = body.id;
  if (typeof id !== "string" || !/^[0-9a-fA-F-]{36}$/.test(id)) {
    return NextResponse.json({ ok: false, error: "Invalid id." }, { status: 400 });
  }

  const patch: Record<string, unknown> = {
    reviewed_at: new Date().toISOString(),
    reviewed_by:
      typeof body.reviewedBy === "string" ? body.reviewedBy.slice(0, 80) : "reviewer",
  };

  if ("verifiedLabel" in body) {
    const label = body.verifiedLabel;
    if (label !== null && (typeof label !== "string" || !CLASS_DIRS.includes(label))) {
      return NextResponse.json(
        {
          ok: false,
          error: `verifiedLabel must be null or one of: ${CLASS_DIRS.join(", ")}`,
        },
        { status: 400 },
      );
    }
    patch.verified_label = label;
  }

  if ("status" in body) {
    const status = body.status;
    if (!["pending", "approved", "rejected"].includes(String(status))) {
      return NextResponse.json(
        { ok: false, error: "status must be pending, approved or rejected." },
        { status: 400 },
      );
    }
    patch.status = status;
  }

  if (typeof body.reviewNote === "string") {
    patch.review_note = body.reviewNote.slice(0, 500);
  }

  // Approving without a label would put an unlabelled row in front of the
  // retrainer, which filters on verified_label IS NOT NULL and would silently
  // skip it. Fail loudly instead.
  if (patch.status === "approved" && patch.verified_label === null) {
    return NextResponse.json(
      { ok: false, error: "Cannot approve a scan without a verified label." },
      { status: 400 },
    );
  }

  const supabase = getSupabase();

  if (patch.status === "approved" && !("verified_label" in patch)) {
    const { data: existing } = await supabase
      .from("training_queue")
      .select("verified_label")
      .eq("id", id)
      .maybeSingle();
    if (!existing?.verified_label) {
      return NextResponse.json(
        { ok: false, error: "Cannot approve a scan without a verified label." },
        { status: 400 },
      );
    }
  }

  const { data, error } = await supabase
    .from("training_queue")
    .update(patch)
    .eq("id", id)
    .select("id, verified_label, status, used_in_training")
    .maybeSingle();

  if (error) {
    return NextResponse.json({ ok: false, error: error.message }, { status: 500 });
  }
  if (!data) {
    return NextResponse.json({ ok: false, error: "Not found." }, { status: 404 });
  }

  return NextResponse.json({ ok: true, item: data });
}
