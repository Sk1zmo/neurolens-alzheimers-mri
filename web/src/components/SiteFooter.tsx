import Link from "next/link";

export default function SiteFooter() {
  return (
    <footer className="mt-16 border-t border-[var(--border)] no-print">
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
          <div className="max-w-md">
            <p className="text-sm font-medium">NeuroLens</p>
            <p className="mt-1.5 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
              An EfficientNet-B0 classifier over the Kaggle augmented Alzheimer&apos;s
              MRI dataset, served as ONNX from a serverless function.{" "}
              <strong className="font-medium text-[var(--text-primary)]">
                Not a medical device.
              </strong>{" "}
              It cannot diagnose anyone and must not inform a clinical decision.
            </p>
          </div>
          <nav className="flex gap-8 text-[0.8125rem]" aria-label="Footer">
            <div className="flex flex-col gap-2">
              <Link href="/" className="text-[var(--text-secondary)] hover:text-[var(--text-primary)]">
                Analyze
              </Link>
              <Link href="/history" className="text-[var(--text-secondary)] hover:text-[var(--text-primary)]">
                History
              </Link>
            </div>
            <div className="flex flex-col gap-2">
              <Link href="/model" className="text-[var(--text-secondary)] hover:text-[var(--text-primary)]">
                Model card
              </Link>
              <Link href="/about" className="text-[var(--text-secondary)] hover:text-[var(--text-primary)]">
                How it works
              </Link>
            </div>
          </nav>
        </div>
      </div>
    </footer>
  );
}
