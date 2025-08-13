/** @type {import('next').NextConfig} */
const nextConfig = {
  devIndicators: false,
  // No rewrites: API routes under /api/* proxy to backend with no keep-alive
};

export default nextConfig;
