import { createClient, type SupabaseClient } from "@supabase/supabase-js";

/**
 * Server-only Supabase client.
 *
 * Uses the service-role key, which bypasses RLS — so it must never be imported
 * into a Client Component. Every route handler that uses it is responsible for
 * scoping its own queries (by session id, or by the reviewer token).
 */

export const SCANS_BUCKET = "scans";

let cached: SupabaseClient | null = null;

export function isSupabaseConfigured(): boolean {
  return Boolean(
    process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY,
  );
}

export function getSupabase(): SupabaseClient {
  if (cached) return cached;

  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) {
    throw new Error(
      "Supabase is not configured. Set SUPABASE_URL and " +
        "SUPABASE_SERVICE_ROLE_KEY in .env.local (or the Vercel project's " +
        "environment variables).",
    );
  }

  cached = createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
    global: { headers: { "x-application-name": "neurolens" } },
  });
  return cached;
}

/** Private bucket — history thumbnails are served through short-lived signed URLs. */
export async function signScanUrl(
  path: string,
  expiresInSeconds = 60 * 60,
): Promise<string | null> {
  const { data, error } = await getSupabase()
    .storage.from(SCANS_BUCKET)
    .createSignedUrl(path, expiresInSeconds);
  if (error || !data) return null;
  return data.signedUrl;
}

export async function signManyScanUrls(
  paths: string[],
  expiresInSeconds = 60 * 60,
): Promise<Record<string, string>> {
  if (paths.length === 0) return {};
  const { data, error } = await getSupabase()
    .storage.from(SCANS_BUCKET)
    .createSignedUrls(paths, expiresInSeconds);
  if (error || !data) return {};

  const out: Record<string, string> = {};
  for (const row of data) {
    if (row.signedUrl && row.path) out[row.path] = row.signedUrl;
  }
  return out;
}
