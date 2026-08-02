/**
 * The label space. Order is load-bearing: it matches the ONNX output axis and
 * training/config.py::CLASS_DIRS. Never reorder without retraining.
 */

export interface ClassInfo {
  id: number;
  dir: string;
  label: string;
  short: string;
  blurb: string;
  /** Ordinal severity, used for the ramp step — not a clinical score. */
  severity: 0 | 1 | 2 | 3;
}

export const CLASSES: ClassInfo[] = [
  {
    id: 0,
    dir: "NonDemented",
    label: "Non Demented",
    short: "Non",
    blurb: "No imaging pattern associated with dementia in this slice.",
    severity: 0,
  },
  {
    id: 1,
    dir: "VeryMildDemented",
    label: "Very Mild Demented",
    short: "Very mild",
    blurb: "Subtle changes consistent with the very earliest stage.",
    severity: 1,
  },
  {
    id: 2,
    dir: "MildDemented",
    label: "Mild Demented",
    short: "Mild",
    blurb: "Changes consistent with mild cognitive decline.",
    severity: 2,
  },
  {
    id: 3,
    dir: "ModerateDemented",
    label: "Moderate Demented",
    short: "Moderate",
    blurb: "Pronounced atrophy pattern consistent with moderate stage.",
    severity: 3,
  },
];

export const CLASS_LABELS = CLASSES.map((c) => c.label);
export const CLASS_DIRS = CLASSES.map((c) => c.dir);

export function classById(id: number): ClassInfo {
  return CLASSES[id] ?? CLASSES[0];
}

export function classByDir(dir: string): ClassInfo | undefined {
  return CLASSES.find((c) => c.dir === dir);
}

export function classByLabel(label: string): ClassInfo | undefined {
  return CLASSES.find((c) => c.label === label);
}
