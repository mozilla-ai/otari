import { chromium } from "@playwright/test"
const OUT = "/Users/fcattaneo/Documents/Work/repos/otari/web/.storybook/shots"

// A representative slice: the primitives an operator sees most, the two charts,
// the table, a modal, and the feature components with real data behind them.
const SHOTS = [
  ["shared-statcard--row", "statcard-row", 1000, 200],
  ["shared-statcard--statuses", "statcard-statuses", 820, 200],
  ["shared-datatable--selectable", "datatable", 1100, 420],
  ["shared-datatable--with-row-detail", "datatable-detail", 1100, 480],
  ["shared-charts--trend-stacked", "chart-stacked", 820, 320],
  ["shared-charts--sparklines", "chart-sparklines", 620, 240],
  ["shared-charts--tooltips", "chart-tooltips", 760, 220],
  ["shared-banners--errors", "banners", 620, 260],
  ["shared-emptystate--with-children", "emptystate", 640, 340],
  ["shared-copy--field-group", "copy-fields", 640, 260],
  ["shared-filterchips--interactive", "filterchips", 900, 220],
  ["shared-confirmdialog--danger", "confirmdialog", 900, 500],
  ["usage-sharecard--all-ratios-and-themes", "sharecard-matrix", 1500, 1000],
  ["activity-activitytimeline--with-errors", "activity-timeline", 900, 400],
  ["auth-publicauthlayout--sign-up", "auth-signup", 900, 700],
  ["onboarding-setupguidecard--waiting", "setupguide", 820, 700],
  ["onboarding-setupguidecard--activated", "setupguide-done", 820, 400],
  ["settings-maildeliverycard--ready", "mail-ready", 760, 460],
  ["models-pricingwarning--requests-being-refused", "pricing-warning", 780, 200],
  ["tools-workspacecodeexecutionpolicycard--enabled-with-ceilings", "codeexec", 780, 620],
  ["routing-routerreadiness--warm", "router-warm", 780, 520],
  ["auth-intromark--states", "intromark", 520, 180],
]

const browser = await chromium.launch()
for (const theme of ["light", "dark"]) {
  for (const [id, name, w, h] of SHOTS) {
    const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 })
    await page.goto(`http://localhost:6006/iframe.html?id=${id}&globals=theme:${theme}`, { waitUntil: "networkidle" })
    await page.waitForFunction(() => {
      const r = document.querySelector("#storybook-root")
      const p = document.querySelector('[role="alertdialog"],[role="dialog"]')
      return ((r?.textContent ?? "") + (p?.textContent ?? "")).trim().length > 0 ||
             document.querySelectorAll("#storybook-root svg").length > 0
    }, { timeout: 20000 })
    await page.waitForTimeout(700)
    await page.screenshot({ path: `${OUT}/${theme}-${name}.png`, fullPage: true })
    await page.close()
  }
}
await browser.close()
console.log("captured", SHOTS.length * 2, "screenshots")
