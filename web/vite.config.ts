import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vitest/config";

// The dashboard is served by the gateway at "/", so we build straight into the
// Python package (src/gateway/static/dashboard) and commit the output. That
// keeps `pip install otari` and the Docker image self-contained without a Node
// build stage. See AGENTS.md ("Web dashboard").
const outDir = fileURLToPath(new URL("../src/gateway/static/dashboard", import.meta.url));

// The gateway serves the dashboard and the API from one origin, so the app
// fetches "/v1/..." and "/health" as same-origin paths. `npm run dev` serves
// only the SPA, so proxy those to a running gateway. Override the target to
// develop against a deployed gateway instead of a local one:
//   OTARI_DEV_API=https://your-app.up.railway.app npm run dev
const apiTarget = process.env.OTARI_DEV_API ?? "http://localhost:8000";
const apiProxy = { target: apiTarget, changeOrigin: true };

// Which gateway the dev server talks to decides which master key signs you in,
// and the app reports an unreachable or unauthorized gateway as an invalid key.
// Print the target so it is obvious which one is in play.
const announceApiTarget = {
  name: "announce-api-target",
  apply: "serve",
  configureServer(server: { httpServer: { once: (e: string, cb: () => void) => void } | null }) {
    server.httpServer?.once("listening", () => {
      const origin = process.env.OTARI_DEV_API ? "OTARI_DEV_API" : "default";
      // eslint-disable-next-line no-console
      console.log(`\n  ➜  API:     ${apiTarget}  (${origin})\n`);
    });
  },
} as const;

export default defineConfig({
  base: "/",
  plugins: [react(), tailwindcss(), announceApiTarget],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    proxy: {
      "/v1": apiProxy,
      "/health": apiProxy,
    },
    // Edits written through a bind mount (e.g. by an agent in a container) do
    // not always reach a watcher on the host as filesystem events. Set
    // VITE_USE_POLLING=1 if hot reload misses changes.
    watch: process.env.VITE_USE_POLLING ? { usePolling: true, interval: 300 } : undefined,
  },
  build: {
    outDir,
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: {
          heroui: ["@heroui/react"],
          react: ["react", "react-dom", "react-router-dom"],
          "tanstack-query": ["@tanstack/react-query"],
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: true,
    // Vitest owns the component tests under src/; the Playwright specs in e2e/
    // run in a real browser and must not be collected here.
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
  },
});
