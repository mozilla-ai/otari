import { fileURLToPath } from "node:url"
import babel from "@rolldown/plugin-babel"
import tailwindcss from "@tailwindcss/vite"
import { tanstackRouter } from "@tanstack/router-plugin/vite"
import react, { reactCompilerPreset } from "@vitejs/plugin-react"
import { defineConfig } from "vitest/config"
import { pwaManifest } from "./pwaManifest.ts"

// The dashboard is served by the gateway at "/", so we build straight into the
// Python package (src/gateway/static/dashboard). The output is gitignored, not
// committed: Vite content-hashes every asset filename, so a committed bundle made
// any two branches touching web/src conflict by construction. The Docker image
// builds it in a Node stage instead. See AGENTS.md ("Web dashboard").
const outDir = fileURLToPath(
  new URL("../src/gateway/static/dashboard", import.meta.url),
)

// The dashboard bundles the operator user guide (docs/dashboard.md) so the
// running app ships the guide it matches, rather than pointing at a separate
// docs site that may describe a different version. The guide lives outside web/,
// so the dev server and Vitest need read access to it (the production build
// resolves it through Rollup regardless). Grant only web/ and docs/: an explicit
// server.fs.allow replaces Vite's default, so widening it to the whole repo root
// would serve gitignored secrets (config.yml, otari.db) at /@fs/... over the dev
// server. web/ must be listed since the default is replaced.
const webRoot = fileURLToPath(new URL("./", import.meta.url))
const docsDir = fileURLToPath(new URL("../docs", import.meta.url))

// The gateway serves the dashboard and the API from one origin, so the app
// fetches "/v1/..." and "/health" as same-origin paths. `pnpm run dev` serves
// only the SPA, so proxy those to a running gateway. Override the target to
// develop against a deployed gateway instead of a local one:
//   OTARI_DEV_API=https://your-app.up.railway.app pnpm run dev
const apiTarget = process.env.OTARI_DEV_API ?? "http://localhost:8000"
const apiProxy = { target: apiTarget, changeOrigin: true }

// Which gateway the dev server talks to decides which master key signs you in,
// and the app reports an unreachable or unauthorized gateway as an invalid key.
// Print the target so it is obvious which one is in play.
const announceApiTarget = {
  name: "announce-api-target",
  apply: "serve",
  configureServer(server: {
    httpServer: { once: (e: string, cb: () => void) => void } | null
  }) {
    server.httpServer?.once("listening", () => {
      const origin = process.env.OTARI_DEV_API ? "OTARI_DEV_API" : "default"
      console.log(`\n  ➜  API:     ${apiTarget}  (${origin})\n`)
    })
  },
} as const

export default defineConfig({
  base: "/",
  plugins: [
    // Generates src/routeTree.gen.ts from src/routes/, and splits each route's
    // component into its own chunk (autoCodeSplitting), which is why a route
    // file may export nothing but `Route`. Must precede the React plugin: it
    // rewrites the route modules that plugin then compiles.
    tanstackRouter({ target: "react", autoCodeSplitting: true }),
    react(),
    // The React Compiler memoizes components and derived values at build time,
    // which is why the frontend standards say not to sprinkle useMemo/useCallback
    // by hand. It is a Babel pass, and as of @vitejs/plugin-react 6 that plugin no
    // longer hosts one: the pass runs as its own plugin here, configured by the
    // `reactCompilerPreset` helper that plugin still exports (a preset over
    // babel-plugin-react-compiler with the include/exclude filter already set, so
    // node_modules stays out of it, as the old `include` default did).
    //
    // Deliberately Babel and not `react({ compiler: true })`, the plugin's other
    // route: that one swaps in oxc-transform-react, a Rust reimplementation the
    // plugin still labels experimental, so it would change which compiler decides
    // what to memoize. otari-ai/frontend runs the Babel pass; keep the two
    // configured alike. React 19 is the preset's default target.
    babel({ presets: [reactCompilerPreset()] }),
    tailwindcss(),
    // The web app manifest, generated from the resolved `base` because a
    // static file in public/ is copied verbatim and pointed installs back at
    // the origin root when the bundle was built under a base path (#857).
    pwaManifest(),
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
    watch: process.env.VITE_USE_POLLING
      ? { usePolling: true, interval: 300 }
      : undefined,
  },
  build: {
    outDir,
    emptyOutDir: true,
    rollupOptions: {
      output: {
        // Vite 8 bundles with Rolldown, which takes the same five vendor chunks as
        // `codeSplitting.groups` matched on module id rather than as the map of
        // chunk name to entry module Rollup's `manualChunks` accepted (an object
        // there is a type error here, and `manualChunks` itself is deprecated).
        // Each group still pulls its own dependency graph in with it
        // (includeDependenciesRecursively defaults to true), which is what the
        // entry-module form used to express.
        codeSplitting: {
          groups: [
            // First, because a tie in `priority` is broken by array order and
            // React must not be absorbed into whichever vendor group happens to
            // reach it first. The trailing separator keeps this off react-icons,
            // react-markdown and react-aria-components.
            {
              name: "react",
              test: /[\\/]node_modules[\\/](react|react-dom|scheduler)[\\/]/,
            },
            { name: "heroui", test: /[\\/]node_modules[\\/]@heroui[\\/]/ },
            {
              name: "tanstack-query",
              test: /[\\/]node_modules[\\/]@tanstack[\\/]react-query[\\/]/,
            },
            // Its own chunk rather than folded in with React. The router pulls in
            // @tanstack/router-core, history and store, and putting that graph in
            // the react chunk reorders the entry's imports enough that HeroUI
            // evaluates before React's exports exist ("Cannot read properties of
            // undefined (reading 'createContext')" at first paint).
            {
              name: "tanstack-router",
              test: /[\\/]node_modules[\\/]@tanstack[\\/]react-router[\\/]/,
            },
            // recharts (and its d3 deps) is a large, self-contained vendor lib. Split
            // it out so it loads with the chart-bearing route bundles, not the shell.
            { name: "recharts", test: /[\\/]node_modules[\\/]recharts[\\/]/ },
          ],
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/tests/setup.ts"],
    css: true,
    // Vitest owns the component tests under src/; the Playwright specs in e2e/
    // run in a real browser and must not be collected here.
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
  },
})
