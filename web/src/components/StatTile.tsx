/**
 * A handful of headline numbers is a KPI row, not a grouped bar chart —
 * comparing "Cohen's kappa" against "accuracy" on a shared axis would be
 * meaningless. Each tile carries its own value and a one-line gloss.
 */
export default function StatTile({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "accent";
}) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-2)] p-4">
      <p className="text-[0.6875rem] font-medium uppercase tracking-[0.06em] text-[var(--text-muted)]">
        {label}
      </p>
      <p
        className="mt-1.5 text-[1.5rem] font-semibold leading-none tracking-tight"
        style={tone === "accent" ? { color: "var(--accent)" } : undefined}
      >
        {value}
      </p>
      {hint && (
        <p className="mt-1.5 text-[0.6875rem] leading-snug text-[var(--text-muted)]">
          {hint}
        </p>
      )}
    </div>
  );
}
