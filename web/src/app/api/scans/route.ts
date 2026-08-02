import { NextResponse } from "next/server";
import {
  SCANS_BUCKET,
  getSupabase,
  isSupabaseConfigured,
  signManyScanUrls,
} from "@/lib/supabase";
import { CLASS_DIRS, classById } from "@/lib/classes";
import type { ScanRecord } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * The Supabase client is untyped here (no generated Database types), so
 * `.select()` widens its result to a union that includes PostgREST's error
 * shape. These row types are the contract with supabase/schema.sql; keep them
 * in step if you alter the table.
 */
interface ScanRow {
  id: string;
  created_at: string;
  original_filename: string | null;
  storage_path: string;
  predicted_class_id: number;
  predicted_label: string;
  confidence: number;
  probabilities: number[] | null;
  margin: number | null;
  out_of_distribution: boolean;
  note: string | null;
  model_version: string | null;
}

const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
const ALLOWED_MIME = new Set(["image/jpeg", "image/png", "image/webp", "image/bmp"]);

function notConfigured() {
  return NextResponse.json(
    {
      ok: false,
      error:
        "Storage is not configured. Add SUPABASE_URL and " +
        "SUPABASE_SERVICE_ROLE_KEY, then run supabase/schema.sql.",
      code: "SUPABASE_NOT_CONFIGURED",
    },
    { status: 503 },
  );
}

function sessionFrom(req: Request, fallback?: unknown): string | null {
  const header = req.headers.get("x-session-id");
  const value = header || (typeof fallback === "string" ? fallback : null);
  if (!value) return null;
  // Keep it to the shape we mint client-side; blocks path traversal into
  // storage keys and stops a caller enumerating with wildcards.
  return /^[A-Za-z0-9_-]{8,64}$/.test(value) ? value : null;
}

// ---------------------------------------------------------------- GET list
export async function GET(req: Request) {
  if (!isSupabaseConfigured()) return notConfigured();

  const url = new URL(req.url);
  const sessionId = sessionFrom(req, url.searchParams.get("session"));
  if (!sessionId) {
    return NextResponse.json(
      { ok: false, error: "Missing or malformed session id." },
      { status: 400 },
    );
  }

  const limit = Math.min(
    Math.max(Number(url.searchParams.get("limit") ?? 60) || 60, 1),
    200,
  );

  const supabase = getSupabase();
  const { data, error } = await supabase
    .from("scans")
    .select(
      "id, created_at, original_filename, storage_path, predicted_class_id, " +
        "predicted_label, confidence, probabilities, margin, " +
        "out_of_distribution, note, model_version",
    )
    .eq("session_id", sessionId)
    .order("created_at", { ascending: false })
    .limit(limit);

  if (error) {
    return NextResponse.json({ ok: false, error: error.message }, { status: 500 });
  }

  const rows = (data ?? []) as unknown as ScanRow[];
  const signed = await signManyScanUrls(rows.map((r) => r.storage_path));
  const scans: ScanRecord[] = rows.map((r) => ({
    id: r.id,
    created_at: r.created_at,
    original_filename: r.original_filename,
    predicted_class_id: r.predicted_class_id,
    predicted_label: r.predicted_label,
    confidence: r.confidence,
    probabilities: r.probabilities ?? [],
    margin: r.margin,
    out_of_distribution: Boolean(r.out_of_distribution),
    note: r.note,
    model_version: r.model_version,
    imageUrl: signed[r.storage_path] ?? null,
  }));

  return NextResponse.json({ ok: true, scans });
}

// ---------------------------------------------------------------- POST save
export async function POST(req: Request) {
  if (!isSupabaseConfigured()) return notConfigured();

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { ok: false, error: "Malformed JSON body." },
      { status: 400 },
    );
  }

  const sessionId = sessionFrom(req, body.sessionId);
  if (!sessionId) {
    return NextResponse.json(
      { ok: false, error: "Missing or malformed session id." },
      { status: 400 },
    );
  }

  const prediction = body.prediction as Record<string, unknown> | undefined;
  const imageDataUrl = body.image as string | undefined;
  if (!prediction || typeof imageDataUrl !== "string") {
    return NextResponse.json(
      { ok: false, error: "Body must include 'prediction' and 'image'." },
      { status: 400 },
    );
  }

  const match = /^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/s.exec(imageDataUrl);
  if (!match) {
    return NextResponse.json(
      { ok: false, error: "'image' must be a base64 image data URL." },
      { status: 400 },
    );
  }

  const mime = match[1].toLowerCase();
  if (!ALLOWED_MIME.has(mime)) {
    return NextResponse.json(
      { ok: false, error: `Unsupported image type: ${mime}` },
      { status: 415 },
    );
  }

  const buffer = Buffer.from(match[2], "base64");
  if (buffer.byteLength === 0 || buffer.byteLength > MAX_IMAGE_BYTES) {
    return NextResponse.json(
      { ok: false, error: "Image is empty or larger than 12 MB." },
      { status: 413 },
    );
  }

  const classId = Number(prediction.class_id);
  if (!Number.isInteger(classId) || classId < 0 || classId >= CLASS_DIRS.length) {
    return NextResponse.json(
      { ok: false, error: "Prediction has an invalid class_id." },
      { status: 400 },
    );
  }
  const info = classById(classId);

  const supabase = getSupabase();
  const ext = mime.split("/")[1].replace("jpeg", "jpg");
  const storagePath = `${sessionId}/${crypto.randomUUID()}.${ext}`;

  const upload = await supabase.storage
    .from(SCANS_BUCKET)
    .upload(storagePath, buffer, { contentType: mime, upsert: false });
  if (upload.error) {
    return NextResponse.json(
      { ok: false, error: `Upload failed: ${upload.error.message}` },
      { status: 500 },
    );
  }

  const consent = body.contributeToTraining !== false;
  const size = prediction.image_size as [number, number] | undefined;

  // Copy 1 — the user's history.
  const { data: scanRow, error: scanErr } = await supabase
    .from("scans")
    .insert({
      session_id: sessionId,
      storage_path: storagePath,
      original_filename:
        typeof body.filename === "string" ? body.filename.slice(0, 200) : null,
      mime_type: mime,
      byte_size: buffer.byteLength,
      width: size?.[0] ?? null,
      height: size?.[1] ?? null,
      predicted_class_id: classId,
      predicted_label: info.label,
      confidence: Number(prediction.confidence) || 0,
      probabilities: (prediction.probabilities as number[]) ?? [],
      margin: Number(prediction.margin) || null,
      energy: Number(prediction.energy) || null,
      out_of_distribution: Boolean(prediction.out_of_distribution),
      input_check: prediction.input_check ?? null,
      model_name:
        (prediction.model as Record<string, unknown> | undefined)?.name ?? null,
      model_version:
        (prediction.model as Record<string, unknown> | undefined)?.version ?? null,
      note: typeof body.note === "string" ? body.note.slice(0, 500) : null,
    })
    .select("id, created_at")
    .single();

  if (scanErr) {
    // Don't leave the object orphaned in the bucket.
    await supabase.storage.from(SCANS_BUCKET).remove([storagePath]);
    return NextResponse.json({ ok: false, error: scanErr.message }, { status: 500 });
  }

  // Copy 2 — the retraining queue. Same object in storage, separate lifecycle:
  // it survives the user deleting their history, and it only becomes training
  // data once a reviewer attaches a verified_label in /review.
  if (consent) {
    const { error: queueErr } = await supabase.from("training_queue").insert({
      scan_id: scanRow.id,
      session_id: sessionId,
      storage_path: storagePath,
      predicted_label: info.dir,
      predicted_confidence: Number(prediction.confidence) || 0,
      consent: true,
    });
    if (queueErr) {
      // The user's own record is already safe; log rather than fail the request.
      console.error("training_queue insert failed:", queueErr.message);
    }
  }

  return NextResponse.json({
    ok: true,
    id: scanRow.id,
    created_at: scanRow.created_at,
    contributedToTraining: consent,
  });
}
