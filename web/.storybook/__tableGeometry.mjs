// Reads the computed geometry of every per-feature table context, in both
// themes, at three viewports, and prints it as JSON.
//
// The instrument for a change to table CSS. Run it, change the stylesheet, run
// it again, diff the two files: that is the proof that nothing moved, and it is
// what the screenshot suite would give if it were a gate (it is
// workflow-dispatch only, with gitignored baselines).
//
//   STORYBOOK_HARNESS=1 pnpm --dir web run storybook   # dev server on :6006
//   pnpm --dir web exec node .storybook/__tableGeometry.mjs > before.json
//   ...edit src/styles/globals.css...
//   pnpm --dir web exec node .storybook/__tableGeometry.mjs > after.json
//   diff <(jq -S . before.json) <(jq -S . after.json)
//
// It reads `harness/TableGeometry.stories.tsx`, which renders `DataTable` inside every
// wrapper class the stylesheet targets plus one with no wrapper at all. The
// contexts and their column ids are derived from globals.css rather than
// transcribed from the feature files, so it measures what the CSS reaches.
//
// It found the regression that made this worth writing: moving the row
// separator to the base looked correct and applied nothing, because an existing
// pair further down the file owned the same property at the same specificity.
import { chromium } from "@playwright/test"

const PORT = process.env.PORT ?? "6006"
const ID = "zz-measure-tablegeometry--all-contexts"

const browser = await chromium.launch()
const out = {}

for (const [label, viewport] of [
  ["desktop-large", { width: 1600, height: 1000 }],
  ["desktop-small", { width: 1024, height: 800 }],
  ["mobile", { width: 390, height: 844 }],
]) {
  for (const theme of ["light", "dark"]) {
    const page = await browser.newPage({ viewport })
    await page.goto(`http://localhost:${PORT}/iframe.html?id=${ID}&globals=theme:${theme}`, {
      waitUntil: "domcontentloaded",
    })
    await page.waitForSelector("[data-measure] table tbody tr", { timeout: 30000 })
    // Transitions suppressed: a computed style read while one is running
    // reports the start value, which web/AGENTS.md records as a trap that has
    // produced wrong findings here before.
    await page.addStyleTag({ content: "*{transition:none !important;animation:none !important}" })
    out[`${label}/${theme}`] = await page.evaluate(() => {
      const box = (el) => {
        if (!el) return null
        const s = getComputedStyle(el)
        const r = el.getBoundingClientRect()
        return {
          w: Math.round(r.width * 10) / 10,
          h: Math.round(r.height * 10) / 10,
          bg: s.backgroundColor,
          borderTop: `${s.borderTopWidth} ${s.borderTopColor}`,
          borderBottom: `${s.borderBottomWidth} ${s.borderBottomColor}`,
          borderRight: `${s.borderRightWidth} ${s.borderRightColor}`,
          padBlock: `${s.paddingTop}/${s.paddingBottom}`,
          position: s.position,
          left: s.left,
          zIndex: s.zIndex,
        }
      }
      const result = {}
      for (const section of document.querySelectorAll("[data-measure]")) {
        const place = section.getAttribute("data-measure")
        const table = section.querySelector("table")
        const header = section.querySelector("thead")
        const firstRow = section.querySelector("tbody tr")
        const cells = [...(firstRow?.querySelectorAll("td") ?? [])]
        const columns = [...(header?.querySelectorAll("th") ?? [])]
        result[place] = {
          frame: box(section.firstElementChild?.nextElementSibling ?? section.querySelector("div")),
          table: box(table),
          header: box(header),
          row: box(firstRow),
          cells: cells.map((c) => ({ key: c.getAttribute("data-key"), ...box(c) })),
          columns: columns.map((c) => ({ key: c.getAttribute("data-key"), ...box(c) })),
        }
      }
      return result
    })
    await page.close()
  }
}
await browser.close()
process.stdout.write(JSON.stringify(out, null, 1))
