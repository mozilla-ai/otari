import { existsSync, readFileSync } from "node:fs"
import { join } from "node:path"

import { describe, expect, it } from "vitest"

import { buildManifest, MANIFEST_PATH } from "../pwaManifest"

// Resolved from the Vitest root (web/) rather than import.meta.url, which the
// jsdom environment reports as an http URL. Same reason as src/routes.test.ts.
const WEB = process.cwd()

describe("the generated web app manifest", () => {
  it("says at the origin root exactly what the static file used to", () => {
    const manifest = buildManifest("/")
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

  it("keeps the static manifest retired", () => {
    // A file re-added to public/ would be copied over the emitted one and
    // reintroduce the hardcoded "/" silently; the plugin is the only source.
    expect(existsSync(join(WEB, "public", MANIFEST_PATH))).toBe(false)
  })
})
