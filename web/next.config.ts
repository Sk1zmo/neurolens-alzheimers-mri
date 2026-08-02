import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Note: do NOT add outputFileTracingExcludes for "./api/**/*" here. The glob
  // also matches node_modules/next/dist/compiled/@opentelemetry/api, which
  // strips it from the server bundle and makes every route handler crash with
  // "Cannot find module '.../@opentelemetry/api'". It is unnecessary regardless
  // — nothing in the Next build imports the Python function, so tracing never
  // pulls in /api or the ONNX model.
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "X-Frame-Options", value: "DENY" },
        ],
      },
    ];
  },
};

export default nextConfig;
