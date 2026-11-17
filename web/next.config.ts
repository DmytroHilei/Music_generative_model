import type { NextConfig } from 'next';

// `next dev` forwards /v1 to the laptop engine (python -m musicar.serve), so the page and the engine share an origin
// and need no CORS while developing. ENGINE_URL points it elsewhere.
const ENGINE = process.env.ENGINE_URL ?? 'http://127.0.0.1:8000';

const nextConfig: NextConfig = {
  compress: false, // gzip would hold back the engine's event stream
  async redirects() {
    // the side-by-side clips and the blind test are a static page (public/compare, eval/compare/h2h_listen.py)
    return [
      { source: '/compare', destination: '/compare/index.html', permanent: false },
      { source: '/samples', destination: '/samples/index.html', permanent: false },
    ];
  },
  async rewrites() {
    return [{ source: '/v1/:path*', destination: `${ENGINE}/v1/:path*` }];
  },
};

export default nextConfig;
