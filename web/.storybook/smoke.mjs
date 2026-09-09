import { chromium } from "@playwright/test"

// A story has rendered when it put something on screen: real text, an SVG, a
// portalled dialog, or something that PAINTS.
//
// That last one is why this is a function rather than a text check. The design
// system has primitives whose whole output is a shape: a toggle's track, a
// divider's hairline, a meter's bar, a skeleton's block. Each renders correctly
// with no text node and no SVG anywhere, so a text-or-svg predicate reports
// nine of them as timeouts and the catalog gate fails on components that are
// fine.
//
// It cannot be relaxed to `childElementCount`, for the reason this probe was
// written around: several stories wrap their subject in a sizing <div>, which
// exists immediately and would satisfy a child count while the component inside
// is still null. So "paints" is the narrower question, asked of the descendants
// rather than the wrapper: a box with a non-zero size AND either a fill or a
// border. A sizing <div> has neither, which is exactly the case that has to keep
// failing.
const RENDERED = () => {
  const b = document.body.classList
  if (b.contains("sb-show-errordisplay")) return true
  if (!b.contains("sb-show-main")) return false
  const root = document.querySelector("#storybook-root")
  const portal = document.querySelector('[role="alertdialog"], [role="dialog"]')
  if (portal) return true
  const text =
    (root?.textContent ?? "").trim().length +
    (portal?.textContent ?? "").trim().length
  if (text > 0) return true
  if (document.querySelectorAll("#storybook-root svg").length > 0) return true
  return [...(root?.querySelectorAll("*") ?? [])].some((el) => {
    const box = el.getBoundingClientRect()
    if (box.width === 0 || box.height === 0) return false
    const style = getComputedStyle(el)
    const filled =
      style.backgroundColor !== "transparent" &&
      style.backgroundColor !== "rgba(0, 0, 0, 0)"
    const edged =
      Number.parseFloat(style.borderTopWidth) > 0 ||
      Number.parseFloat(style.borderRightWidth) > 0 ||
      Number.parseFloat(style.borderBottomWidth) > 0 ||
      Number.parseFloat(style.borderLeftWidth) > 0
    return filled || edged
  })
}

const index = await fetch("http://localhost:6006/index.json").then((r) => r.json())
const ids = Object.keys(index.entries).filter((id) => index.entries[id].type === "story")

const browser = await chromium.launch()
const failures = []

for (const theme of ["light", "dark"]) {
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } })
  for (const id of ids) {
    const errors = []
    const onErr = (e) => errors.push(String(e.message ?? e).split("\n")[0])
    // Only uncaught exceptions and React errors count. A bare "Failed to load
    // resource" is this probe's own fault: it reuses one page across ~180 rapid
    // navigations, so in-flight requests abort as it moves on.
    const onConsole = (m) => {
      if (m.type() !== "error") return
      const text = m.text().split("\n")[0]
      if (/Failed to load resource/.test(text)) return
      if (text.trim() === "%o") return
      errors.push(text)
    }
    page.on("pageerror", onErr)
    page.on("console", onConsole)
    try {
      await page.goto(`http://localhost:6006/iframe.html?id=${id}&globals=theme:${theme}`, { waitUntil: "domcontentloaded" })
      // See RENDERED at the top for what counts as rendered and why. The one
      // thing worth repeating here: a card gated on two sequential queries
      // (context, then a per-workspace one) legitimately renders nothing until
      // the second resolves, which is why this waits rather than sampling once.
      await page.waitForFunction(RENDERED, undefined, { timeout: 20000 })
      const shown = await page.evaluate(() => ({
        errored: document.body.classList.contains("sb-show-errordisplay"),
        errorText: document.querySelector("#error-message")?.textContent?.slice(0, 300) || null,
        children:
          (document.querySelector("#storybook-root")?.childElementCount ?? 0) +
          (document.querySelector('[role="alertdialog"], [role="dialog"]') ? 1 : 0),
        text:
          (document.querySelector("#storybook-root")?.textContent ?? "").trim().length +
          (document.querySelector('[role="alertdialog"], [role="dialog"]')?.textContent ?? "").trim().length,
        svgs: document.querySelectorAll("#storybook-root svg").length,
      }))
      if (shown.errored) failures.push({ theme, id, why: "error display", errorText: shown.errorText })
      else if (shown.children === 0) failures.push({ theme, id, why: "empty root" })
      // Asks the same question the wait did. Without this a graphical primitive
      // clears the wait and is then reported as "no text and no svg".
      else if (shown.text === 0 && shown.svgs === 0 && !(await page.evaluate(RENDERED)))
        failures.push({ theme, id, why: "no text, no svg, and nothing painted" })
      else if (errors.length) failures.push({ theme, id, why: "console errors", errors: [...new Set(errors)].slice(0, 3) })
    } catch (e) {
      failures.push({ theme, id, why: "timeout", detail: String(e.message).split("\n")[0], errors: [...new Set(errors)].slice(0, 2) })
    }
    page.off("pageerror", onErr)
    page.off("console", onConsole)
  }
  await page.close()
}

console.log(`checked ${ids.length} stories x 2 themes`)
console.log(failures.length === 0 ? "ALL RENDERED CLEAN" : JSON.stringify(failures, null, 1))
await browser.close()
