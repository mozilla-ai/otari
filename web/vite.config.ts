import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vitest/config";

// The dashboard is served by the gateway at "/", so we build straight into the
// Python package (src/gateway/static/dashboard) and commit the output. That
// keeps `pip install otari` and the Docker image self-contained without a Node
// build stage. See AGENTS.md ("Web dashboard").
const outDir = fileURLToPath(new URL("../src/gateway/static/dashboard", import.meta.url));

export default defineConfig({
  base: "/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    outDir,
    emptyOutDir: true,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: true,
  },
});
