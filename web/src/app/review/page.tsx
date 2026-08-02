import ReviewClient from "@/components/ReviewClient";

export const metadata = { title: "Review console" };

export default function ReviewPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 sm:py-10">
      <header className="mb-7 max-w-2xl">
        <h1 className="text-[1.75rem] font-semibold tracking-tight">
          Review console
        </h1>
        <p className="mt-3 text-[0.9375rem] leading-relaxed text-[var(--text-secondary)]">
          The gate between an uploaded image and the training set. A queued scan
          becomes training data only once a human assigns it a verified label
          here — the model&apos;s own prediction is shown for triage but is never
          used as ground truth.
        </p>
      </header>
      <ReviewClient />
    </div>
  );
}
