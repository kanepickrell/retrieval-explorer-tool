import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  // VITE_BASE=/explorer/ when Open WebUI proxies it into its own page (see Dockerfile).
  base: process.env.VITE_BASE || "/",
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: { port: 5180 },
});
