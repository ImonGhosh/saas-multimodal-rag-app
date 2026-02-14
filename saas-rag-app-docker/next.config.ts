import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactStrictMode: true,
  experimental: {
    proxyTimeout: 5 * 60 * 1000,
  },
  async rewrites() {
    return [
      {
        source: "/api",
        destination: "http://127.0.0.1:8000/api",
      },
      {
        source: "/ingest",
        destination: "http://127.0.0.1:8000/ingest",
      },
      {
        source: "/ingest-file/:path*",
        destination: "http://127.0.0.1:8000/ingest-file/:path*",
      }
    ];
  },
};

export default nextConfig;
