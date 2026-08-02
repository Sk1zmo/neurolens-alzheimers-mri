import { NextResponse } from "next/server";
import { SCANS_BUCKET, getSupabase, isSupabaseConfigured } from "@/lib/supabase";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function sessionFrom(req: Request): string | null {
  const value = req.headers.get("x-session-id");
  if (!value) return null;
  return /^[A-Za-z0-9_-]{8,64}$/.test(value) ? value : null;
}

/**
 * Deleting a scan removes the user's copy and the stored image.
 *
 * The training_queue row survives by design (its scan_id FK is ON DELETE SET
 * NULL) *unless* it has not been reviewed yet — an unreviewed contribution is
 * still the user's to withdraw, and its image is about to disappear anyway.
 * A row already approved and used in training stays, because silently
 * shrinking the corpus behind the retraining pipeline's back would make its
 * before/after comparisons meaningless.
 */
export async function DELETE(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  if (!isSupabaseConfigured()) {
    return NextResponse.json(
      { ok: false, error: "Storage is not configured." },
      { status: 503 },
    );
  }

  const sessionId = sessionFrom(req);
  if (!sessionId) {
    return NextResponse.json(
      { ok: false, error: "Missing or malformed session id." },
      { status: 400 },
    );
  }

  const { id } = await params;
  if (!/^[0-9a-fA-F-]{36}$/.test(id)) {
    return NextResponse.json({ ok: false, error: "Invalid id." }, { status: 400 });
  }

  const supabase = getSupabase();

  // Scope by session_id so one visitor can never delete another's scan.
  const { data: row, error: findErr } = await supabase
    .from("scans")
    .select("id, storage_path")
    .eq("id", id)
    .eq("session_id", sessionId)
    .maybeSingle();

  if (findErr) {
    return NextResponse.json({ ok: false, error: findErr.message }, { status: 500 });
  }
  if (!row) {
    return NextResponse.json({ ok: false, error: "Not found." }, { status: 404 });
  }

  const { data: queued } = await supabase
    .from("training_queue")
    .select("id, used_in_training, status")
    .eq("scan_id", id);

  const withdrawable = (queued ?? []).filter(
    (q) => !q.used_in_training && q.status === "pending",
  );
  if (withdrawable.length > 0) {
    await supabase
      .from("training_queue")
      .delete()
      .in(
        "id",
        withdrawable.map((q) => q.id),
      );
  }

  const retained = (queued ?? []).length - withdrawable.length;

  const { error: delErr } = await supabase
    .from("scans")
    .delete()
    .eq("id", id)
    .eq("session_id", sessionId);
  if (delErr) {
    return NextResponse.json({ ok: false, error: delErr.message }, { status: 500 });
  }

  // Only drop the stored image once nothing still references it.
  if (retained === 0) {
    await supabase.storage.from(SCANS_BUCKET).remove([row.storage_path]);
  }

  return NextResponse.json({
    ok: true,
    deleted: id,
    trainingContributionRetained: retained > 0,
  });
}
