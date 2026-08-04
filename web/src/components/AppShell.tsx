"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

/**
 * Jira/Slack-style shell: a persistent dark navigation rail on the left, a
 * compact top bar carrying breadcrumbs, and a dense content column. Chosen
 * over the previous centred marketing layout because this app is now a
 * multi-section workbench — analyze, model card, research, reader study,
 * review — and a horizontal nav bar cannot show that structure.
 */

interface NavItem {
  href: string;
  label: string;
  icon: React.ReactNode;
  badge?: string;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

const I = {
  scan: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
      <circle cx="12" cy="12" r="3.2" stroke="currentColor" strokeWidth="1.7" />
    </svg>
  ),
  history: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 7v5l3.2 1.9" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="12" cy="12" r="8" stroke="currentColor" strokeWidth="1.7" />
    </svg>
  ),
  model: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M4 19V9M9.3 19V5M14.7 19v-7M20 19v-4" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" />
    </svg>
  ),
  research: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M6 4h8l4 4v12a0 0 0 0 1 0 0H6a0 0 0 0 1 0 0V4Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M14 4v4h4M8.5 13h7M8.5 16.5h4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  ),
  rate: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 4.5 14.2 9l5 .7-3.6 3.5.9 5-4.5-2.4L7.5 18.2l.9-5L4.8 9.7l5-.7L12 4.5Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  ),
  review: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="m8.5 12.3 2.5 2.5 5-5.3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <rect x="4" y="4" width="16" height="16" rx="2.5" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  ),
  about: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="8" stroke="currentColor" strokeWidth="1.7" />
      <path d="M12 11v5M12 8.2v.2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  ),
};

const SECTIONS: NavSection[] = [
  {
    title: "Analysis",
    items: [
      { href: "/", label: "Analyze scan", icon: I.scan },
      { href: "/history", label: "Scan history", icon: I.history },
    ],
  },
  {
    title: "Evidence",
    items: [
      { href: "/model", label: "Model card", icon: I.model },
      { href: "/research", label: "Research", icon: I.research },
      { href: "/rate", label: "Reader study", icon: I.rate },
    ],
  },
  {
    title: "Operations",
    items: [
      { href: "/review", label: "Review queue", icon: I.review },
      { href: "/about", label: "How it works", icon: I.about },
    ],
  },
];

const CRUMBS: Record<string, string> = {
  "/": "Analyze scan",
  "/history": "Scan history",
  "/model": "Model card",
  "/research": "Research",
  "/rate": "Reader study",
  "/review": "Review queue",
  "/about": "How it works",
};

function Logo() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 3c-2.2 0-4 1.6-4 3.6 0 .3 0 .6.1.9C6.8 8 6 9.1 6 10.5c0 .9.4 1.8 1 2.4-.6.6-1 1.4-1 2.3C6 17 7.6 18.5 9.6 18.5c.5 0 1-.1 1.4-.3v1.3c0 .3.2.5.5.5h1c.3 0 .5-.2.5-.5v-1.3c.4.2.9.3 1.4.3 2 0 3.6-1.5 3.6-3.3 0-.9-.4-1.7-1-2.3.6-.6 1-1.5 1-2.4 0-1.4-.8-2.5-2.1-3 .1-.3.1-.6.1-.9C16 4.6 14.2 3 12 3Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  useEffect(() => {
    setTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light");
  }, []);
  return (
    <button
      className="btn btn-subtle h-8 w-8 !px-0"
      onClick={() => {
        const next = theme === "dark" ? "light" : "dark";
        document.documentElement.dataset.theme = next;
        localStorage.setItem("neurolens.theme", next);
        setTheme(next);
      }}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
      title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
    >
      {theme === "dark" ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="4.2" stroke="currentColor" strokeWidth="1.7" />
          <path d="M12 2.8v2M12 19.2v2M2.8 12h2M19.2 12h2M5.5 5.5l1.4 1.4M17.1 17.1l1.4 1.4M18.5 5.5l-1.4 1.4M6.9 17.1l-1.4 1.4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M20 14.4A8.4 8.4 0 0 1 9.6 4a8.4 8.4 0 1 0 10.4 10.4Z" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
        </svg>
      )}
    </button>
  );
}

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  useEffect(() => setOpen(false), [pathname]);

  const crumb = CRUMBS[pathname] ?? "";

  return (
    <div className="flex min-h-dvh">
      {/* -------------------------------------------------------- side rail */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-60 shrink-0 flex-col transition-transform lg:static lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
        style={{ background: "var(--nav)", borderRight: "1px solid var(--nav-border)" }}
      >
        <div
          className="flex h-14 items-center gap-2.5 px-4"
          style={{ borderBottom: "1px solid var(--nav-border)" }}
        >
          <span style={{ color: "var(--brand)" }}>
            <Logo />
          </span>
          <div className="min-w-0">
            <p
              className="truncate text-[0.875rem] font-semibold leading-tight"
              style={{ color: "var(--nav-text-strong)" }}
            >
              NeuroLens
            </p>
            <p
              className="truncate text-[0.6875rem] leading-tight"
              style={{ color: "var(--nav-section)" }}
            >
              MRI stage classification
            </p>
          </div>
        </div>

        <nav className="flex-1 overflow-y-auto py-2 scroll-thin" aria-label="Main">
          {SECTIONS.map((section) => (
            <div key={section.title}>
              <p className="nav-section">{section.title}</p>
              {section.items.map((item) => {
                const active =
                  item.href === "/"
                    ? pathname === "/"
                    : pathname.startsWith(item.href);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="nav-item"
                    aria-current={active ? "page" : undefined}
                  >
                    <span className="shrink-0 opacity-90">{item.icon}</span>
                    <span className="truncate">{item.label}</span>
                  </Link>
                );
              })}
            </div>
          ))}
        </nav>

        <div className="px-4 py-3" style={{ borderTop: "1px solid var(--nav-border)" }}>
          <span className="lozenge lozenge-warning">Research use only</span>
          <p
            className="mt-2 text-[0.6875rem] leading-relaxed"
            style={{ color: "var(--nav-section)" }}
          >
            Not a medical device. Cannot diagnose.
          </p>
        </div>
      </aside>

      {open && (
        <button
          className="fixed inset-0 z-40 bg-black/40 lg:hidden"
          onClick={() => setOpen(false)}
          aria-label="Close navigation"
        />
      )}

      {/* ---------------------------------------------------------- content */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header
          className="sticky top-0 z-30 flex h-14 items-center gap-3 px-4 sm:px-6"
          style={{
            background: "var(--surface)",
            borderBottom: "1px solid var(--border)",
          }}
        >
          <button
            className="btn btn-subtle h-8 w-8 !px-0 lg:hidden"
            onClick={() => setOpen(true)}
            aria-label="Open navigation"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
            </svg>
          </button>

          <nav aria-label="Breadcrumb" className="min-w-0 flex-1">
            <ol className="flex items-center gap-1.5 text-[0.75rem]">
              <li>
                <Link
                  href="/"
                  className="hover:underline"
                  style={{ color: "var(--text-muted)" }}
                >
                  NeuroLens
                </Link>
              </li>
              {crumb && (
                <>
                  <li aria-hidden="true" style={{ color: "var(--text-muted)" }}>
                    /
                  </li>
                  <li
                    className="truncate font-medium"
                    style={{ color: "var(--text-secondary)" }}
                    aria-current="page"
                  >
                    {crumb}
                  </li>
                </>
              )}
            </ol>
          </nav>

          <a
            href="https://github.com/Sk1zmo/neurolens-alzheimers-mri"
            target="_blank"
            rel="noreferrer noopener"
            className="btn btn-subtle btn-sm hidden sm:inline-flex"
          >
            Source
          </a>
          <ThemeToggle />
        </header>

        <main id="main" className="flex-1">
          {children}
        </main>

        <footer
          className="no-print px-4 py-4 sm:px-6"
          style={{ borderTop: "1px solid var(--border)" }}
        >
          <p
            className="text-[0.75rem] leading-relaxed"
            style={{ color: "var(--text-muted)" }}
          >
            EfficientNet-B0 over the Kaggle augmented Alzheimer&apos;s MRI dataset,
            served as ONNX from a serverless function.{" "}
            <strong style={{ color: "var(--text-secondary)" }}>
              Not a medical device.
            </strong>{" "}
            It cannot diagnose anyone and must not inform a clinical decision.
          </p>
        </footer>
      </div>
    </div>
  );
}
