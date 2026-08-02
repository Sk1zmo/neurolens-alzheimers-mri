import { NextResponse } from "next/server";
import { getSupabase, isSupabaseConfigured } from "@/lib/supabase";
import { CLASSES } from "@/lib/classes";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Aggregate counters for the live-corpus panel.
 *
 * Everything here is a count — no scan contents, no session ids, nothing that
 * could identify a contributor — so it is safe to serve unauthenticated.
 */
export async function GET() {
  if (!isSupabaseConfigured()) {
    return NextResponse.json({
      ok: true,
      configured: false,
      scans: null,
      retraining: null,
      byClass: [],
    });
  }

  const supabase = getSupabase();

  const [scanStats, retrainStats, distribution] = await Promise.all([
    supabase.from("scan_stats").select("*").maybeSingle(),
    supabase.from("retraining_stats").select("*").maybeSingle(),
    supabase.from("scans").select("predicted_class_id"),
  ]);

  const counts = new Array(CLASSES.length).fill(0);
  for (const row of distribution.data ?? []) {
    const i = row.predicted_class_id as number;
    if (i >= 0 && i < counts.length) counts[i] += 1;
  }

  return NextResponse.json({
    ok: true,
    configured: true,
    scans: scanStats.data ?? null,
    retraining: retrainStats.data ?? null,
    byClass: CLASSES.map((c) => ({
      id: c.id,
      label: c.label,
      short: c.short,
      count: counts[c.id],
    })),
  });
}
