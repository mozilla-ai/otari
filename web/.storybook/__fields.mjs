import { chromium } from "@playwright/test"
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 900, height: 700 } })
for (const [id, label] of [
  ["shared-field--in-form", "shared/Field"],
  ["auth-authfields--sign-in-form", "AuthFields"],
  ["models-setpricedialog--single-model", "RateField (dialog)"],
]) {
  await page.goto(`http://localhost:6006/iframe.html?id=${id}&globals=theme:light`, { waitUntil: "networkidle" })
  await page.waitForTimeout(1200)
  const r = await page.evaluate(() => {
    const inputs = [...document.querySelectorAll("input")]
    return inputs.slice(0, 3).map((i) => {
      const cs = getComputedStyle(i)
      return {
        type: i.type,
        fontSize: cs.fontSize,
        autocomplete: i.getAttribute("autocomplete"),
        spellcheck: i.getAttribute("spellcheck"),
        ariaInvalid: i.getAttribute("aria-invalid"),
        hasLabel: !!(i.labels?.length || i.getAttribute("aria-labelledby")),
        describedby: !!i.getAttribute("aria-describedby"),
      }
    })
  })
  console.log(`\n${label}`)
  for (const i of r) console.log("  ", JSON.stringify(i))
}
await browser.close()
