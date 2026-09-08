import { chromium } from "@playwright/test"

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
      // sb-show-main / sb-show-errordisplay is how Storybook reports which of the
      // two it settled on; #error-message is always in the DOM, so it is no signal.
      // Wait for real CONTENT, not just a child element. Several stories wrap the
      // subject in a sizing <div>, which exists immediately and would satisfy a
      // childElementCount check while the component inside is still null -- and a
      // card gated on two sequential queries (context, then a per-workspace one)
      // legitimately renders nothing until the second resolves.
      await page.waitForFunction(() => {
        const b = document.body.classList
        if (b.contains("sb-show-errordisplay")) return true
        if (!b.contains("sb-show-main")) return false
        const root = document.querySelector("#storybook-root")
        const portal = document.querySelector('[role="alertdialog"], [role="dialog"]')
        const text =
          (root?.textContent ?? "").trim().length +
          (portal?.textContent ?? "").trim().length
        const svgs = document.querySelectorAll("#storybook-root svg").length
        return text > 0 || svgs > 0 || !!portal
      }, { timeout: 20000 })
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
      else if (shown.text === 0 && shown.svgs === 0) failures.push({ theme, id, why: "no text and no svg" })
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
