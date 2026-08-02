import { jsPDF } from "jspdf";
import { CLASSES } from "@/lib/classes";
import type { ModelMetrics, PredictResponse } from "@/lib/types";

const INK = [11, 11, 11] as const;
const INK_2 = [82, 81, 78] as const;
const MUTED = [137, 135, 129] as const;
const ACCENT = [42, 120, 214] as const;
const RULE = [225, 224, 217] as const;
const WARN_BG = [253, 244, 224] as const;
const WARN_INK = [124, 82, 6] as const;

/**
 * Builds the report by drawing primitives rather than rasterising the DOM.
 * A screenshot of the page would carry the app's dark theme and hairline
 * borders into print at whatever the viewport happened to be; drawing gives a
 * fixed A4 layout with selectable text.
 */
export async function buildReport(
  result: PredictResponse,
  scanDataUrl: string,
  metrics: ModelMetrics | null,
): Promise<jsPDF> {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const W = doc.internal.pageSize.getWidth();
  const M = 48;
  let y = M;

  const rule = (yy: number) => {
    doc.setDrawColor(...RULE);
    doc.setLineWidth(0.7);
    doc.line(M, yy, W - M, yy);
  };

  // ---------------------------------------------------------------- header
  doc.setFont("helvetica", "bold");
  doc.setFontSize(17);
  doc.setTextColor(...INK);
  doc.text("NeuroLens", M, y);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(...MUTED);
  doc.text("Alzheimer's MRI stage classification report", M, y + 14);
  doc.text(new Date().toLocaleString(), W - M, y, { align: "right" });
  doc.text(
    `Model ${result.model.name ?? "—"} v${result.model.version ?? "—"}`,
    W - M,
    y + 14,
    { align: "right" },
  );

  y += 28;
  rule(y);
  y += 26;

  // ------------------------------------------------------------ prediction
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(...MUTED);
  doc.text("PREDICTED STAGE", M, y);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(23);
  doc.setTextColor(...ACCENT);
  doc.text(result.label, M, y + 24);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(...MUTED);
  doc.text("CALIBRATED CONFIDENCE", W / 2 + 20, y);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(23);
  doc.setTextColor(...INK);
  doc.text(`${(result.confidence * 100).toFixed(1)}%`, W / 2 + 20, y + 24);

  y += 44;

  // -------------------------------------------------------------- warnings
  const warnings: string[] = [];
  if (!result.input_check.looks_like_brain_mri) {
    warnings.push(
      "The uploaded image did not pass the brain-MRI structure check. The classification below is unreliable.",
    );
  }
  if (result.out_of_distribution) {
    warnings.push(
      "This image sits outside the distribution the model was trained on. Treat the result as unsupported.",
    );
  }
  if (result.margin < 0.15) {
    warnings.push(
      `The top two stages are separated by only ${(result.margin * 100).toFixed(1)} percentage points — the model is not clearly deciding between them.`,
    );
  }

  if (warnings.length) {
    const lines = warnings.flatMap((w) =>
      doc.splitTextToSize(`• ${w}`, W - 2 * M - 20) as string[],
    );
    const h = lines.length * 12 + 16;
    doc.setFillColor(...WARN_BG);
    doc.roundedRect(M, y, W - 2 * M, h, 5, 5, "F");
    doc.setFontSize(8.5);
    doc.setTextColor(...WARN_INK);
    doc.setFont("helvetica", "normal");
    lines.forEach((line, i) => doc.text(line, M + 10, y + 15 + i * 12));
    y += h + 18;
  }

  // ---------------------------------------------------------------- images
  const imgW = (W - 2 * M - 16) / 2;
  const imgTop = y;
  try {
    doc.addImage(scanDataUrl, "JPEG", M, y, imgW, imgW, undefined, "FAST");
  } catch {
    /* an unreadable preview should not sink the report */
  }
  if (result.overlay_png) {
    try {
      doc.addImage(
        result.overlay_png,
        "PNG",
        M + imgW + 16,
        y,
        imgW,
        imgW,
        undefined,
        "FAST",
      );
    } catch {
      /* ignore */
    }
  }
  y += imgW + 12;

  doc.setFontSize(8);
  doc.setTextColor(...MUTED);
  doc.text("Uploaded scan", M, y);
  if (result.overlay_png) {
    doc.text("Class activation map", M + imgW + 16, y);
  }
  y = imgTop + imgW + 30;

  // --------------------------------------------------------- probabilities
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10.5);
  doc.setTextColor(...INK);
  doc.text("Probability by stage", M, y);
  y += 16;

  const barX = M + 130;
  const barW = W - M - barX - 52;
  CLASSES.forEach((c) => {
    const p = result.probabilities[c.id] ?? 0;
    const predicted = c.id === result.class_id;

    // Spread into jsPDF's overloaded colour setters only works on a single
    // tuple, not a union of them — so index explicitly here.
    const labelInk = predicted ? INK : INK_2;

    doc.setFont("helvetica", predicted ? "bold" : "normal");
    doc.setFontSize(9);
    doc.setTextColor(labelInk[0], labelInk[1], labelInk[2]);
    doc.text(c.label, M, y + 7);

    doc.setFillColor(236, 235, 229);
    doc.roundedRect(barX, y, barW, 8, 4, 4, "F");
    if (p > 0.004) {
      if (predicted) doc.setFillColor(...ACCENT);
      else doc.setFillColor(...MUTED);
      doc.roundedRect(barX, y, Math.max(barW * p, 4), 8, 4, 4, "F");
    }

    doc.setTextColor(labelInk[0], labelInk[1], labelInk[2]);
    doc.text(`${(p * 100).toFixed(1)}%`, W - M, y + 7, { align: "right" });
    y += 17;
  });

  y += 12;
  rule(y);
  y += 20;

  // -------------------------------------------------------- model context
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10.5);
  doc.setTextColor(...INK);
  doc.text("How the model performs on held-out data", M, y);
  y += 15;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8.5);
  doc.setTextColor(...INK_2);

  if (metrics) {
    const perClass = metrics.per_class[result.label];
    const rows: [string, string][] = [
      ["Test accuracy", `${(metrics.headline.accuracy * 100).toFixed(1)}%`],
      [
        "Balanced accuracy",
        `${(metrics.headline.balanced_accuracy * 100).toFixed(1)}%`,
      ],
      ["Macro F1", metrics.headline.macro_f1.toFixed(3)],
      ["Macro AUC (one-vs-rest)", metrics.headline.macro_auc_ovr.toFixed(3)],
      ["Cohen's kappa", metrics.headline.cohen_kappa.toFixed(3)],
      ["Held-out test images", String(metrics.test_set.n)],
    ];
    if (perClass) {
      rows.push([
        `Recall for "${result.label}"`,
        `${(perClass.recall * 100).toFixed(1)}%`,
      ]);
      rows.push([
        `Precision for "${result.label}"`,
        `${(perClass.precision * 100).toFixed(1)}%`,
      ]);
    }
    const half = Math.ceil(rows.length / 2);
    rows.forEach(([k, v], i) => {
      const col = i < half ? M : W / 2 + 10;
      const row = i < half ? i : i - half;
      doc.setTextColor(...MUTED);
      doc.text(k, col, y + row * 13);
      doc.setTextColor(...INK);
      doc.text(v, col + 165, y + row * 13, { align: "right" });
    });
    y += half * 13 + 10;
  } else {
    doc.text("Metrics file not available in this deployment.", M, y);
    y += 16;
  }

  // ------------------------------------------------------------ disclaimer
  y += 6;
  rule(y);
  y += 16;

  doc.setFont("helvetica", "bold");
  doc.setFontSize(8.5);
  doc.setTextColor(...INK);
  doc.text("Not a medical device", M, y);
  y += 12;

  doc.setFont("helvetica", "normal");
  doc.setTextColor(...INK_2);
  const disclaimer =
    "NeuroLens is a research and educational demonstration. It was trained on the public Kaggle " +
    "augmented Alzheimer's MRI dataset, which is brain MRI rather than CT and carries no subject " +
    "identifiers — so slices from one brain may appear in both training and evaluation, and " +
    "real-world accuracy on an unseen patient is lower than the figures above. This report is not " +
    "a diagnosis, must not be used to inform clinical decisions, and does not replace assessment " +
    "by a qualified clinician.";
  (doc.splitTextToSize(disclaimer, W - 2 * M) as string[]).forEach((line, i) =>
    doc.text(line, M, y + i * 11),
  );

  return doc;
}

export async function downloadReport(
  result: PredictResponse,
  scanDataUrl: string,
  metrics: ModelMetrics | null,
): Promise<void> {
  const doc = await buildReport(result, scanDataUrl, metrics);
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  doc.save(`neurolens-report-${stamp}.pdf`);
}
