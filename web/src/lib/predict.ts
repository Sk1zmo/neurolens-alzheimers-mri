import type { ModelMetrics, PredictResponse } from "@/lib/types";

/**
 * `next dev` does not run Vercel's Python runtime, so the ONNX endpoint is
 * absent locally. Point NEXT_PUBLIC_PREDICT_URL at `python training/serve_local.py`
 * (default http://127.0.0.1:8000/api/predict) to develop against the real
 * model; in production the default relative path hits the deployed function.
 */
export const PREDICT_URL =
  process.env.NEXT_PUBLIC_PREDICT_URL || "/api/predict";

export async function runPrediction(file: File): Promise<PredictResponse> {
  const res = await fetch(PREDICT_URL, {
    method: "POST",
    headers: { "Content-Type": file.type || "application/octet-stream" },
    body: file,
  });

  let payload: unknown;
  try {
    payload = await res.json();
  } catch {
    throw new Error(
      res.status === 404
        ? "The prediction endpoint was not found. If you are running `next dev`, " +
          "start the local model server and set NEXT_PUBLIC_PREDICT_URL."
        : `The model service returned ${res.status} with a non-JSON body.`,
    );
  }

  const data = payload as PredictResponse | { ok: false; error: string; hint?: string };
  if (!res.ok || !data.ok) {
    const err = data as { error?: string; hint?: string };
    throw new Error(
      [err.error ?? `Prediction failed (${res.status}).`, err.hint]
        .filter(Boolean)
        .join(" "),
    );
  }
  return data;
}

export async function loadMetrics(): Promise<ModelMetrics | null> {
  try {
    const res = await fetch("/model/metrics.json", { cache: "force-cache" });
    if (!res.ok) return null;
    return (await res.json()) as ModelMetrics;
  } catch {
    return null;
  }
}

export function fileToDataUrl(file: File | Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Could not read that file."));
    reader.readAsDataURL(file);
  });
}
