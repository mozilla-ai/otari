import { fileURLToPath } from "node:url";

import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vitest/config";

// The dashboard is served by the gateway at "/", so we build straight into the
// Python package (src/gateway/static/dashboard). The output is gitignored, not
// committed: Vite content-hashes every asset filename, so a committed bundle made
// any two branches touching web/src conflict by construction. The Docker image
// builds it in a Node stage instead. See AGENTS.md ("Web dashboard").
const outDir = fileURLToPath(new URL("../src/gateway/static/dashboard", import.meta.url));

// The dashboard bundles the operator user guide (docs/dashboard.md) so the
// running app ships the guide it matches, rather than pointing at a separate
// docs site that may describe a different version. The guide lives outside web/,
// so the dev server and Vitest need read access to it (the production build
// resolves it through Rollup regardless). Grant only web/ and docs/: an explicit
// server.fs.allow replaces Vite's default, so widening it to the whole repo root
// would serve gitignored secrets (config.yml, otari.db) at /@fs/... over the dev
// server. web/ must be listed since the default is replaced.
const webRoot = fileURLToPath(new URL("./", import.meta.url));
const docsDir = fileURLToPath(new URL("../docs", import.meta.url));

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
  plugins: [
    // Generates src/routeTree.gen.ts from src/routes/, and splits each route's
    // component into its own chunk (autoCodeSplitting), which is why a route
    // file may export nothing but `Route`. Must precede the React plugin: it
    // rewrites the route modules that plugin then compiles.
    tanstackRouter({ target: "react", autoCodeSplitting: true }),
    react(),
    tailwindcss(),
    announceApiTarget,
  ],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    // Allow reading the bundled user guide (docs/dashboard.md) from docs/;
    // web/ stays listed because an explicit allow list replaces the default.
    fs: { allow: [webRoot, docsDir] },
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
          react: ["react", "react-dom", "react-dom/client", "react/jsx-runtime"],
          "tanstack-query": ["@tanstack/react-query"],
          // Its own chunk rather than folded in with React. The router pulls in
          // @tanstack/router-core, history and store, and putting that graph in
          // the react chunk reorders the entry's imports enough that HeroUI
          // evaluates before React's exports exist ("Cannot read properties of
          // undefined (reading 'createContext')" at first paint).
          "tanstack-router": ["@tanstack/react-router"],
          // recharts (and its d3 deps) is a large, self-contained vendor lib. Split
          // it out so it loads with the chart-bearing route bundles, not the shell.
          recharts: ["recharts"],
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/shared/test/setup.ts"],
    css: true,
    // Vitest owns the component tests under src/; the Playwright specs in e2e/
    // run in a real browser and must not be collected here.
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
  },
});
