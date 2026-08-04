export interface InputCheck {
  looks_like_brain_mri: boolean;
  score: number;
  checks: Record<string, boolean>;
  signals: Record<string, number>;
}

export interface MetricZ {
  value: number;
  reference_mean: number;
  reference_sd: number;
  z: number;
  percentile: number;
}

export interface RegionAttribution {
  key: string;
  name: string;
  side: string;
  lobe: string;
  description: string;
  attention_share: number;
  attention_density: number;
}

export interface AnatomyResult {
  error?: string;
  view: {
    plane: string;
    plane_confidence: number;
    consistent_with_axial: boolean;
    estimated_level: string;
    checks: Record<string, boolean>;
    signals: Record<string, number>;
    limitation: string;
  };
  metrics: Record<string, number>;
  metric_info: Record<
    string,
    { label: string; unit: string; direction: string; meaning: string }
  >;
  z_scores: Record<string, MetricZ>;
  attribution: RegionAttribution[];
  pose: { rotation_deg: number; brain_area_px: number };
  convention: string;
  atlas_note: string;
}

export interface Finding {
  kind: "morphometry";
  metric: string;
  severity: "borderline" | "notable" | "marked";
  region: string;
  title: string;
  measurement: string;
  interpretation: string;
  rationale: string;
  references: string[];
  z: number;
}

export interface AttentionFinding {
  kind: "attention";
  region: string;
  side: string;
  lobe: string;
  attention_share: number;
  attention_density: number;
  measurement: string;
  anatomy: string;
  rationale: string;
  references: string[];
}

export interface FindingsReport {
  summary: string;
  findings: Finding[];
  attention: AttentionFinding[];
  references: Record<string, { citation: string; note: string }>;
  disclaimer: string;
}

export interface PredictResponse {
  ok: true;
  anatomy: AnatomyResult | null;
  report: FindingsReport | null;
  class_id: number;
  label: string;
  class_dir: string;
  confidence: number;
  confidence_uncalibrated: number;
  margin: number;
  probabilities: number[];
  classes: string[];
  logits: number[];
  energy: number;
  out_of_distribution: boolean;
  input_check: InputCheck;
  overlay_png: string | null;
  cam_png: string | null;
  image_size: [number, number];
  model: {
    name: string;
    version: string;
    temperature: number;
    test_accuracy?: number;
    test_macro_f1?: number;
  };
}

export interface PredictError {
  ok: false;
  error: string;
  hint?: string;
}

export type PredictResult = PredictResponse | PredictError;

export interface ScanRecord {
  id: string;
  created_at: string;
  original_filename: string | null;
  predicted_class_id: number;
  predicted_label: string;
  confidence: number;
  probabilities: number[];
  margin: number | null;
  out_of_distribution: boolean;
  note: string | null;
  model_version: string | null;
  imageUrl?: string | null;
}

export interface QueueRecord {
  id: string;
  scan_id: string | null;
  created_at: string;
  storage_path: string;
  predicted_label: string;
  predicted_confidence: number | null;
  verified_label: string | null;
  status: "pending" | "approved" | "rejected";
  used_in_training: boolean;
  review_note: string | null;
  imageUrl?: string | null;
}

export interface RetrainingStats {
  queued_total: number;
  labelled: number;
  ready_for_training: number;
  already_trained: number;
  rejected: number;
}

export interface ScanStats {
  total_scans: number;
  total_sessions: number;
  flagged_out_of_distribution: number;
  mean_confidence: number | null;
  last_upload_at: string | null;
}

/** Shape of public/model/metrics.json, written by training/evaluate.py. */
export interface ModelMetrics {
  model: string;
  architecture: string;
  classes: string[];
  class_dirs: string[];
  img_size: number;
  temperature: number;
  tta_horizontal_flip: boolean;
  test_set: { n: number; source: string; per_class_n: number[] };
  headline: {
    accuracy: number;
    balanced_accuracy: number;
    macro_f1: number;
    weighted_f1: number;
    cohen_kappa: number;
    macro_auc_ovr: number;
    brier_score: number;
    mse: number;
    ece_calibrated: number;
    ece_uncalibrated: number;
  };
  per_class: Record<
    string,
    {
      precision: number;
      recall: number;
      f1: number;
      support: number;
      auc: number;
    }
  >;
  confusion_matrix: number[][];
  roc: Record<string, { fpr: number[]; tpr: number[]; auc: number }>;
  caveats: string[];
}
