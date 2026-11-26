import { existsSync, readFileSync } from "node:fs"
import { join } from "node:path"

import { describe, expect, it } from "vitest"

import {
  buildManifest,
  MANIFEST_PATH,
  pwaManifest,
  rewriteManifestLink,
} from "../pwaManifest"

const HOSTED = "A hosted description."

type Middleware = (
  req: { url?: string },
  res: { setHeader: () => void; end: (body: string) => void },
  next: () => void,
) => void

// Drives the plugin's hooks the way Vite does, without a build: the resolved
// config first, then the bundle and the dev server.
function runPlugin(options?: Parameters<typeof pwaManifest>[0]) {
  const plugin = pwaManifest(options) as unknown as {
    configResolved: (config: { base: string }) => void
    generateBundle: (this: {
      emitFile: (file: { source: string }) => void
    }) => void
    configureServer: (server: {
      middlewares: { use: (middleware: Middleware) => void }
    }) => void
  }
  plugin.configResolved({ base: "/dashboard/" })

  let emitted = ""
  plugin.generateBundle.call({
    emitFile: (file) => {
      emitted = file.source
    },
  })

  let served = ""
  plugin.configureServer({
    middlewares: {
      use: (middleware) =>
        middleware(
          { url: `/dashboard/${MANIFEST_PATH}` },
          {
            setHeader: () => {},
            end: (body) => {
              served = body
            },
          },
          () => {},
        ),
    },
  })
  return { emitted: JSON.parse(emitted), served: JSON.parse(served) }
}

// Resolved from the Vitest root (web/) rather than import.meta.url, which the
// jsdom environment reports as an http URL. Same reason as src/routes.test.ts.
const WEB = process.cwd()

describe("the generated web app manifest", () => {
  it("says at the origin root exactly what the static file used to", () => {
    const manifest = buildManifest("/")
    expect(manifest.name).toBe("Otari")
    expect(manifest.short_name).toBe("Otari")
    expect(manifest.id).toBe("/")
    expect(manifest.start_url).toBe("/")
    expect(manifest.scope).toBe("/")
    expect(manifest.icons.map((icon) => icon.src)).toEqual([
      "/pwa/icon-192.png",
      "/pwa/icon-512.png",
      "/pwa/icon-maskable-512.png",
    ])
  })

  it("prefixes every self-reference with the base path", () => {
    // The reason it is generated at all (#857): built under a base path, the
    // static manifest was fetched from the right place and then pointed the
    // install back at the origin root.
    const manifest = buildManifest("/dashboard/")
    expect(manifest.id).toBe("/dashboard/")
    expect(manifest.start_url).toBe("/dashboard/")
    expect(manifest.scope).toBe("/dashboard/")
    for (const icon of manifest.icons) {
      expect(icon.src).toMatch(/^\/dashboard\/pwa\//)
    }
  })

  it("normalizes a base missing its trailing slash", () => {
    expect(buildManifest("/dashboard")).toEqual(buildManifest("/dashboard/"))
  })

  it("advertises only icons public/pwa/ actually ships", () => {
    // Vite copies public/ into the bundle unchanged, so a src whose file is
    // not there 404s and the launcher falls back to a screenshot of the page.
    for (const icon of buildManifest("/").icons) {
      expect(existsSync(join(WEB, "public", icon.src)), icon.src).toBe(true)
    }
  })

  it("is emitted at the path index.html links", () => {
    const html = readFileSync(join(WEB, "index.html"), "utf8")
    expect(html).toContain(`<link rel="manifest" href="/${MANIFEST_PATH}" />`)
  })

  it("rewrites the real index.html link under a base path", () => {
    // Vite stopped rewriting this href when the file left public/ (it no
    // longer resolves to an asset it knows), so the plugin owns the rewrite.
    // Held to the actual index.html rather than a fixture, so a drifted link
    // fails here instead of only in a manual base-path build.
    const html = readFileSync(join(WEB, "index.html"), "utf8")
    expect(rewriteManifestLink(html, "/dashboard/")).toContain(
      `<link rel="manifest" href="/dashboard/${MANIFEST_PATH}" />`,
    )
    expect(rewriteManifestLink(html, "/")).toBe(html)
  })

  it("keeps the static manifest retired", () => {
    // A file re-added to public/ would be copied over the emitted one and
    // reintroduce the hardcoded "/" silently; the plugin is the only source.
    expect(existsSync(join(WEB, "public", MANIFEST_PATH))).toBe(false)
  })

  it("takes a description from the build that imports it", () => {
    expect(buildManifest("/", { description: HOSTED }).description).toBe(HOSTED)
    expect(buildManifest("/", {})).toEqual(buildManifest("/"))
  })

  it("emits and dev-serves the same overridden description", () => {
    const { emitted, served } = runPlugin({ description: HOSTED })
    expect(emitted.description).toBe(HOSTED)
    expect(served).toEqual(emitted)
  })

  it("emits and dev-serves the default description without options", () => {
    const { emitted, served } = runPlugin()
    expect(emitted).toEqual(buildManifest("/dashboard/"))
    expect(served).toEqual(emitted)
  })
})
