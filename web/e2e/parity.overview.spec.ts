import { expect, test } from "@playwright/test"

import { login, table, tableRows, tileValue } from "./helpers"
import { COUNTS, PARITY } from "./parity-data"

// The seeded fixture is shared, and nothing here mutates it, but the suite runs
// one worker against one gateway either way (see web/playwright.config.ts).
test.describe.configure({ mode: "serial" })

// The scratch rows are consumed by the Activity spec's bulk-delete flow, which
// runs before this file, so they are deliberately not counted here.
const SEEDED = COUNTS.priced + COUNTS.unpriced + COUNTS.errors

test.describe("overview", () => {
  test("reports the gateway's own totals once it has traffic", async ({
    page,
  }) => {
    await login(page)
    await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible()

    // Onboarding is an empty-gateway state. With providers configured and usage
    // recorded it has to be gone, or a working gateway keeps being told to set
    // itself up. This is the assertion that broke once already: the panel used to
    // gate on "no providers" alone, so a gateway serving imported usage through no
    // local provider config still opened on a getting-started screen.
    await expect(page.getByText("Get started with Otari")).toBeHidden()

    const requests = tileValue(page, "Requests, last 30 days")
    await expect(requests).not.toHaveText("—")
    // At least the fixture: the gateway also carries whatever the onboarding
    // flows left behind, so this is a floor rather than an equality.
    const counted = Number((await requests.innerText()).replaceAll(",", ""))
    expect(counted).toBeGreaterThanOrEqual(SEEDED)

    // The priced half of the fixture is the only costed traffic on this gateway,
    // so a dash (or a zero) here means pricing never reached the tiles.
    const spend = tileValue(page, "Spend, last 30 days")
    await expect(spend).toHaveText(/^\$/)
    await expect(spend).not.toHaveText("$0.00")

    // Errors are seeded, so the rate is a real percentage rather than the
    // "unknown" dash a failed summary would leave. Not anchored at the end: a
    // non-neutral rate also carries a status word ("Elevated"), which is there so
    // the tile never reports its health through color alone.
    await expect(tileValue(page, "Error rate, last 30 days")).toHaveText(
      /^\d+(\.\d+)?%/,
    )
  })

  test("previews the newest requests and opens the full log", async ({
    page,
  }) => {
    await login(page)

    const rows = tableRows(page, "Recent activity")
    // The preview is capped at five rows (useUsageLogs({}, 0, 5)).
    await expect(rows).toHaveCount(5)

    // Newest first, unfiltered. The fixture's priced rows are the most recent
    // traffic on the gateway (the parity seed runs last, and spreads its densest
    // set over the shortest step), so the top row naming that model is what
    // "newest first" means here. An ordering regression would put an older row,
    // or a row from an earlier flow, at the top instead.
    await expect(rows.first()).toContainText(PARITY.models.priced.model)

    await page.getByRole("link", { name: "View all" }).click()
    await expect(page.getByRole("heading", { name: "Activity" })).toBeVisible()
    await expect(table(page, "Activity log")).toBeVisible()
  })

  test("the budget tile is a way into budgets, not just a number", async ({
    page,
  }) => {
    await login(page)
    // The tile is wrapped in a link (StatCard `to`), which is the whole point of
    // it: a budget near its limit is read here and acted on there.
    await tileValue(page, "Budget health").click()
    await expect(
      page.getByRole("heading", { name: "Budgets", exact: true }),
    ).toBeVisible()
  })
})
