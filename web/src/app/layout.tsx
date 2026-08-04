import type { Metadata, Viewport } from "next";
import AppShell from "@/components/AppShell";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "NeuroLens — Alzheimer's MRI stage classifier",
    template: "%s · NeuroLens",
  },
  description:
    "Upload a brain MRI slice and get a four-stage Alzheimer's classification " +
    "with a class activation map, calibrated confidence, and the model's real " +
    "held-out performance. Research and educational use only.",
  applicationName: "NeuroLens",
  robots: { index: true, follow: true },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f9f9f7" },
    { media: "(prefers-color-scheme: dark)", color: "#0d0d0d" },
  ],
};

/**
 * Resolve the theme to a concrete stamp on <html> before first paint.
 * Inline and synchronous on purpose — deferring it produces a flash of the
 * wrong theme on every navigation.
 */
const THEME_SCRIPT = `
(function () {
  try {
    var stored = localStorage.getItem('neurolens.theme');
    var mode = stored === 'light' || stored === 'dark' ? stored
      : (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    document.documentElement.dataset.theme = mode;
    if (!stored) {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function (e) {
        if (!localStorage.getItem('neurolens.theme')) {
          document.documentElement.dataset.theme = e.matches ? 'dark' : 'light';
        }
      });
    }
  } catch (e) {
    document.documentElement.dataset.theme = 'light';
  }
})();
`;

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-theme="light" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="min-h-dvh antialiased">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-[var(--surface)] focus:px-3 focus:py-2 focus:text-sm"
        >
          Skip to content
        </a>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
