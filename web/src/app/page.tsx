import Link from "next/link";
import Analyzer from "@/components/Analyzer";

export default function AnalyzePage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 sm:py-10">
      <header className="mb-8 max-w-2xl">
        <h1 className="text-[1.75rem] font-semibold tracking-tight sm:text-[2rem]">
          Alzheimer&apos;s stage classification from a brain MRI slice
        </h1>
        <p className="mt-3 text-[0.9375rem] leading-relaxed text-[var(--text-secondary)]">
          An EfficientNet-B0 classifier sorts an axial T1 slice into one of four
          stages and shows which regions drove the call. Confidence is
          temperature-calibrated, and every figure on this page comes from a
          held-out test set rather than the training data.
        </p>
        <p className="mt-3 text-[0.8125rem] leading-relaxed text-[var(--text-muted)]">
          Trained on brain{" "}
          <strong className="font-medium text-[var(--text-secondary)]">MRI</strong>,
          not CT — a CT scan will produce a confident and meaningless answer.{" "}
          <Link href="/about" className="text-[var(--accent)] hover:underline">
            How it works
          </Link>
        </p>
      </header>

      <Analyzer />
    </div>
  );
}
