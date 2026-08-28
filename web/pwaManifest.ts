import type { Plugin } from "vite"

/**
 * The web app manifest, generated at build time rather than shipped from
 * `public/`.
 *
 * Everything in `public/` is copied into the bundle verbatim while Vite
 * rewrites the URLs around it, so a static manifest was the one asset that did
 * not follow `base`: built under a base path, `index.html` linked the manifest
 * from the right place and the manifest pointed `start_url`, `scope`, `id` and
 * every icon `src` back at the origin root, so installing the dashboard opened
 * whatever site lives there instead (#857). Generating it from the resolved
 * `base` makes that drift impossible by construction.
 *
 * A module of its own, not a closure inside vite.config.ts, on purpose: the
 * superset build (`otari-ai/frontend/superset/vite.config.ts`, which serves
 * the dashboard under a base path and is where #857 bit) mirrors that config
 * by copying because it is not a module one can extend, but it can import
 * this plugin and stay on the same manifest.
 */

/** Where the manifest lands in the bundle, beside the icons `public/pwa/` ships. */
export const MANIFEST_PATH = "pwa/manifest.webmanifest"

export function buildManifest(base: string) {
  // Vite's resolved `base` always ends in "/"; normalize anyway so a caller
  // holding a bare prefix cannot produce "/dashboardpwa/icon-192.png".
  const prefix = base.endsWith("/") ? base : `${base}/`
  const icon = (name: string, sizes: string, purpose: "any" | "maskable") => ({
    src: `${prefix}pwa/${name}`,
    sizes,
    type: "image/png",
    purpose,
  })
  return {
    id: prefix,
    name: "Otari Dashboard",
    short_name: "Otari",
    description:
      "Otari admin dashboard: browse and price models, manage aliases, and toggle runtime settings.",
    start_url: prefix,
    scope: prefix,
    display: "standalone",
    orientation: "any",
    // Literal copies of --color-background and --color-primary: a manifest
    // cannot read a custom property. src/styles/foundation.test.ts holds them
    // to the tokens, and index.html's theme-color meta is the third copy.
    background_color: "#ffffff",
    theme_color: "#4a7d8f",
    icons: [
      icon("icon-192.png", "192x192", "any"),
      icon("icon-512.png", "512x512", "any"),
      icon("icon-maskable-512.png", "512x512", "maskable"),
    ],
  }
}

const serialize = (base: string) =>
  `${JSON.stringify(buildManifest(base), null, 2)}\n`

export function pwaManifest(): Plugin {
  let base = "/"
  return {
    name: "otari:pwa-manifest",
    configResolved(config) {
      base = config.base
    },
    generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: MANIFEST_PATH,
        source: serialize(base),
      })
    },
    // Vite rewrites an absolute URL in index.html under `base` only when it
    // resolves to a file it knows (public/ or the graph). This one no longer
    // does, so the link the emitted manifest answers is rewritten here.
    transformIndexHtml(html) {
      const prefix = base.endsWith("/") ? base : `${base}/`
      return html.replace(
        `href="/${MANIFEST_PATH}"`,
        `href="${prefix}${MANIFEST_PATH}"`,
      )
    },
    // The dev server serves `public/` directly and the file is not there, so
    // answer the URL index.html links (which carries `base`; this middleware
    // runs ahead of Vite's own, before the base prefix is stripped).
    configureServer(server) {
      const served = `${base.endsWith("/") ? base : `${base}/`}${MANIFEST_PATH}`
      server.middlewares.use((req, res, next) => {
        if (req.url?.split("?", 1)[0] === served) {
          res.setHeader("Content-Type", "application/manifest+json")
          res.end(serialize(base))
          return
        }
        next()
      })
    },
  }
}
