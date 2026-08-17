import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"

import { describe, expect, it } from "vitest"

// Resolved from the Vitest root (web/) rather than import.meta.url, which the
// jsdom environment reports as an http URL. Same reason as src/routes.test.ts.
const WEB = process.cwd()
const FONTS_DIR = join(WEB, "public", "fonts")
const CSS = readFileSync(join(WEB, "src", "styles", "globals.css"), "utf8")

const entries = readdirSync(FONTS_DIR)
const faces = entries.filter((name) => name.endsWith(".woff2"))
const licenses = entries.filter((name) => name.endsWith("-OFL.txt"))

// Every `src: url("/fonts/…")` the stylesheet asks the browser for.
const referenced = [...CSS.matchAll(/url\("\/fonts\/([^"]+)"\)/g)].map(
  (match) => match[1],
)

describe("bundled fonts", () => {
  it("ships faces and licenses", () => {
    // A guard on the guard: an empty directory would leave every assertion
    // below iterating over nothing and reporting green.
    expect(faces.length).toBeGreaterThan(0)
    expect(licenses.length).toBeGreaterThan(0)
  })

  // The OFL's redistribution condition is that the copyright notice and the
  // license travel with the font. Nothing about a build fails when they come
  // apart, which is exactly why it is worth a test: the failure mode is a
  // license violation discovered by someone outside the project.
  it.each(faces)("%s is covered by a license file", (face) => {
    const covering = licenses.filter((license) =>
      face.startsWith(license.replace("-OFL.txt", "")),
    )
    expect(covering, `no *-OFL.txt in public/fonts covers ${face}`).not.toEqual(
      [],
    )
  })

  it.each(licenses)(
    "%s carries a copyright line and the license",
    (license) => {
      const text = readFileSync(join(FONTS_DIR, license), "utf8")
      expect(text).toMatch(/^Copyright .*\S/m)
      expect(text).toContain("SIL OPEN FONT LICENSE Version 1.1")
    },
  )

  it("names every license file in the fonts README", () => {
    const readme = readFileSync(join(FONTS_DIR, "README.md"), "utf8")
    for (const license of licenses) expect(readme).toContain(license)
  })

  it("declares every shipped face", () => {
    // An unreferenced face is dead weight in the bundle, and it is how a family
    // outlives the role it was added for.
    expect([...referenced].sort()).toEqual([...faces].sort())
  })

  it("resolves every @font-face source to a shipped file", () => {
    for (const source of referenced) expect(faces).toContain(source)
  })
})
