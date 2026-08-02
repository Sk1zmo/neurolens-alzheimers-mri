import HistoryClient from "@/components/HistoryClient";

export const metadata = { title: "Your scan history" };

export default function HistoryPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6 sm:py-10">
      <header className="mb-7 max-w-2xl">
        <h1 className="text-[1.75rem] font-semibold tracking-tight">
          Your scan history
        </h1>
        <p className="mt-3 text-[0.9375rem] leading-relaxed text-[var(--text-secondary)]">
          Every scan you chose to save, with the prediction exactly as it was
          returned at the time. There are no accounts here — this list is tied to
          a random id stored in this browser, so clearing site data ends it.
        </p>
      </header>
      <HistoryClient />
    </div>
  );
}
