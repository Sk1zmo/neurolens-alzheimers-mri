import RatingClient from "@/components/RatingClient";

export const metadata = { title: "Reader study" };

export default function RatePage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6 sm:py-10">
      <header className="mb-7 max-w-2xl">
        <h1 className="text-[1.75rem] font-semibold tracking-tight">
          Reader study
        </h1>
        <p className="mt-3 text-[0.9375rem] leading-relaxed text-[var(--text-secondary)]">
          Rate held-out slices without seeing the model&apos;s answer, then compare.
          This produces the human-agreement statistics a paper needs — Cohen&apos;s
          κ between reader and ground truth, and between reader and model.
        </p>
        <p className="mt-3 text-[0.8125rem] leading-relaxed text-[var(--text-muted)]">
          The model&apos;s prediction stays hidden until you have committed a rating,
          so your judgement cannot be anchored by it. Ratings are stored in this
          browser and exported as CSV — nothing is uploaded.
        </p>
      </header>
      <RatingClient />
    </div>
  );
}
