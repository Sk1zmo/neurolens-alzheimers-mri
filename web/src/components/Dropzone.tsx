"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const IMAGE_MIME = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
// Browsers report DICOM and NIfTI inconsistently — often as an empty string or
// application/octet-stream — so extension is the only reliable signal, and the
// server sniffs the magic bytes regardless.
const MEDICAL_EXT = [".dcm", ".dicom", ".ima", ".nii", ".nii.gz", ".zip"];
const ACCEPT_ATTR = [...IMAGE_MIME, ...MEDICAL_EXT].join(",");
const MAX_BYTES = 60 * 1024 * 1024;

function hasMedicalExtension(name: string): boolean {
  const lower = name.toLowerCase();
  return MEDICAL_EXT.some((ext) => lower.endsWith(ext));
}

export function validateImage(file: File): string | null {
  const ok = IMAGE_MIME.includes(file.type) || hasMedicalExtension(file.name);
  if (!ok) {
    return `${file.type || "That file type"} isn't supported. Use JPEG, PNG, WebP, BMP, DICOM (.dcm), a zipped DICOM series, or NIfTI (.nii/.nii.gz).`;
  }
  if (file.size > MAX_BYTES) {
    return `That file is ${(file.size / 1024 / 1024).toFixed(1)} MB — the limit is 60 MB.`;
  }
  return null;
}

export default function Dropzone({
  onFile,
  disabled,
}: {
  onFile: (file: File) => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const accept = useCallback(
    (file: File | undefined | null) => {
      if (!file) return;
      const problem = validateImage(file);
      if (problem) {
        setError(problem);
        return;
      }
      setError(null);
      onFile(file);
    },
    [onFile],
  );

  // Radiology screenshots usually arrive via clipboard, so support paste too.
  useEffect(() => {
    function onPaste(e: ClipboardEvent) {
      if (disabled) return;
      const item = Array.from(e.clipboardData?.items ?? []).find((i) =>
        i.type.startsWith("image/"),
      );
      if (item) accept(item.getAsFile());
    }
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [accept, disabled]);

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) accept(e.dataTransfer.files?.[0]);
        }}
        className={`relative rounded-xl border border-dashed p-8 text-center transition-colors ${
          dragging
            ? "border-[var(--accent)] bg-[var(--accent-soft)]"
            : "border-[var(--border-strong)] bg-[var(--surface-2)]"
        } ${disabled ? "opacity-50" : ""}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTR}
          className="sr-only"
          disabled={disabled}
          onChange={(e) => {
            accept(e.target.files?.[0]);
            e.target.value = "";
          }}
        />

        <div
          className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-full"
          style={{ background: "var(--surface-3)" }}
          aria-hidden="true"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path
              d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5"
              stroke="var(--text-secondary)"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15"
              stroke="var(--text-secondary)"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
          </svg>
        </div>

        <p className="text-sm font-medium">Drop a scan here</p>
        <p className="mt-1 text-[0.8125rem] text-[var(--text-secondary)]">
          or{" "}
          <button
            type="button"
            className="font-medium text-[var(--accent)] underline underline-offset-2 disabled:no-underline"
            onClick={() => inputRef.current?.click()}
            disabled={disabled}
          >
            browse your files
          </button>{" "}
          — you can paste too
        </p>
        <p className="mt-3 text-[0.75rem] leading-relaxed text-[var(--text-muted)]">
          Image (JPEG/PNG/WebP/BMP), DICOM (.dcm), a zipped DICOM series, or
          NIfTI (.nii/.nii.gz) · up to 60 MB
        </p>
        <p className="mt-1 text-[0.75rem] leading-relaxed text-[var(--text-muted)]">
          Give it a volume and the axial slice matching the training level is
          selected automatically.
        </p>
      </div>

      {error && (
        <p
          role="alert"
          className="mt-2.5 flex items-start gap-1.5 text-[0.8125rem]"
          style={{ color: "var(--critical)" }}
        >
          <svg
            width="15"
            height="15"
            viewBox="0 0 24 24"
            fill="none"
            className="mt-0.5 shrink-0"
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" />
            <path
              d="M12 7.5v5.5M12 16.2v.2"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
            />
          </svg>
          {error}
        </p>
      )}
    </div>
  );
}
