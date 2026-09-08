import { chromium } from "@playwright/test"
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1000, height: 400 } })
await page.goto("http://localhost:6006/iframe.html?id=shared-statcard--row&globals=theme:dark", { waitUntil: "networkidle" })
await page.waitForTimeout1500 ?? await page.waitForTimeout(1500)
console.log(await page.evaluate(() => {
  const grid = document.querySelector("#storybook-root > div")
  const cs = grid ? getComputedStyle(grid) : null
  // Is the class present in any loaded stylesheet?
  let found = { "grid-cols-4": false, "grid-cols-2": false, "w-\\[60rem\\]": false }
  for (const sheet of document.styleSheets) {
    let rules
    try { rules = sheet.cssRules } catch { continue }
    for (const r of rules) {
      const t = r.cssText || ""
      if (t.includes("grid-cols-4") || t.includes("grid-template-columns:repeat(4")) found["grid-cols-4"] = true
      if (t.includes("grid-cols-2")) found["grid-cols-2"] = true
      if (t.includes("60rem")) found["w-\\[60rem\\]"] = true
    }
  }
  return {
    className: grid?.className,
    templateColumns: cs?.gridTemplateColumns,
    width: cs?.width,
    inStylesheet: found,
  }
}))
await browser.close()
