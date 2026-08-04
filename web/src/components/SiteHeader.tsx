"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const NAV = [
  { href: "/", label: "Analyze" },
  { href: "/history", label: "History" },
  { href: "/model", label: "Model" },
  { href: "/research", label: "Research" },
  { href: "/rate", label: "Reader study" },
  { href: "/review", label: "Review" },
  { href: "/about", label: "About" },
];

function Mark() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 3c-2.2 0-4 1.6-4 3.6 0 .3 0 .6.1.9C6.8 8 6 9.1 6 10.5c0 .9.4 1.8 1 2.4-.6.6-1 1.4-1 2.3C6 17 7.6 18.5 9.6 18.5c.5 0 1-.1 1.4-.3v1.3c0 .3.2.5.5.5h1c.3 0 .5-.2.5-.5v-1.3c.4.2.9.3 1.4.3 2 0 3.6-1.5 3.6-3.3 0-.9-.4-1.7-1-2.3.6-.6 1-1.5 1-2.4 0-1.4-.8-2.5-2.1-3 .1-.3.1-.6.1-.9C16 4.6 14.2 3 12 3Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
      <path
        d="M12 6.5v11"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        opacity="0.45"
      />
    </svg>
  );
}

function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    const current = document.documentElement.dataset.theme;
    setTheme(current === "dark" ? "dark" : "light");
  }, []);

  function toggle() {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("neurolens.theme", next);
    setTheme(next);
  }

  return (
    <button
      onClick={toggle}
      className="btn btn-ghost h-9 w-9 !p-0"
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
      title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
    >
      {theme === "dark" ? (
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="4.2" stroke="currentColor" strokeWidth="1.5" />
          <path
            d="M12 2.8v2M12 19.2v2M2.8 12h2M19.2 12h2M5.5 5.5l1.4 1.4M17.1 17.1l1.4 1.4M18.5 5.5l-1.4 1.4M6.9 17.1l-1.4 1.4"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
      ) : (
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M20 14.4A8.4 8.4 0 0 1 9.6 4a8.4 8.4 0 1 0 10.4 10.4Z"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinejoin="round"
          />
        </svg>
      )}
    </button>
  );
}

export default function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <header className="sticky top-0 z-40 border-b border-[var(--border)] bg-[color-mix(in_srgb,var(--page)_88%,transparent)] backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-3 px-4 sm:px-6">
        <Link
          href="/"
          className="flex items-center gap-2 text-[var(--text-primary)]"
          aria-label="NeuroLens home"
        >
          <span className="text-[var(--accent)]">
            <Mark />
          </span>
          <span className="text-[0.9375rem] font-semibold tracking-tight">
            NeuroLens
          </span>
        </Link>

        <nav className="ml-4 hidden items-center gap-0.5 md:flex" aria-label="Main">
          {NAV.map((item) => {
            const active =
              item.href === "/"
                ? pathname === "/"
                : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={`rounded-lg px-3 py-1.5 text-sm transition-colors ${
                  active
                    ? "bg-[var(--surface-2)] font-medium text-[var(--text-primary)]"
                    : "text-[var(--text-secondary)] hover:bg-[var(--surface-2)] hover:text-[var(--text-primary)]"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-1.5">
          <span className="chip hidden lg:inline-flex">
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ background: "var(--warning)" }}
              aria-hidden="true"
            />
            Research use only
          </span>
          <ThemeToggle />
          <button
            className="btn btn-ghost h-9 w-9 !p-0 md:hidden"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-label="Toggle navigation"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d={open ? "M6 6l12 12M18 6L6 18" : "M4 7h16M4 12h16M4 17h16"}
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>
      </div>

      {open && (
        <nav
          className="border-t border-[var(--border)] bg-[var(--surface)] px-4 py-2 md:hidden"
          aria-label="Mobile"
        >
          {NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="block rounded-lg px-3 py-2 text-sm text-[var(--text-secondary)] hover:bg-[var(--surface-2)]"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
