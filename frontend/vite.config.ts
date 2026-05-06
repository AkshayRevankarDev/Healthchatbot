import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    port: 3000,
    host: "0.0.0.0",   // bind to all interfaces — makes network URL work
    proxy: {
      // Proxy /api/* → FastAPI on port 8502 (same machine, server-side)
      // This means the browser only ever needs to reach port 3000 —
      // no cross-port CORS issues, works from any device on the LAN.
      "/api": {
        target: "http://127.0.0.1:8502",
        changeOrigin: true,
      },
    },
  },
});
