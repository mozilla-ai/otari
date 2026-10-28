import { expect, test } from "@playwright/test"

import { login, table, tableRows, tileValue } from "./helpers"
import { PARITY } from "./parity-data"

// The seeded fixture is shared, and nothing here mutates it, but the suite runs
// one worker against one gateway either way (see web/playwright.config.ts).
test.describe.configure({ mode: "serial" })

test.describe("overview", () => {
  test("opens on the usage tab with the gateway's own totals", async ({
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

    const tabs = page.getByRole("navigation", { name: "Overview views" })
    await expect(tabs.getByRole("link", { name: "Usage" })).toHaveAttribute(
      "aria-current",
      "page",
    )
    await expect(tileValue(page, "Tracked cost")).toHaveText(/^\$/)
  })

  test("Today narrows the window to local midnight, and the tabs keep it", async ({
    page,
  }) => {
    await login(page)

    await page.getByRole("button", { name: "Today", exact: true }).click()
    await expect(page).toHaveURL(/\/\?.*start_date=/)

    await page
      .getByRole("navigation", { name: "Overview views" })
      .getByRole("link", { name: "Requests" })
      .click()
    await expect(page).toHaveURL(/\/requests\?.*start_date=/)
    await expect(table(page, "Activity log")).toBeVisible()
  })

  test("the Requests tab lists the newest requests first", async ({ page }) => {
    await login(page)
    await page
      .getByRole("navigation", { name: "Overview views" })
      .getByRole("link", { name: "Requests" })
      .click()
    await expect(page).toHaveURL(/\/requests/)

    // The fixture's priced rows are the most recent traffic on the gateway (the
    // parity seed runs last, and spreads its densest set over the shortest
    // step), so the top row naming that model is what "newest first" means
    // here. An ordering regression would put an older row, or a row from an
    // earlier flow, at the top instead.
    await expect(tableRows(page, "Activity log").first()).toContainText(
      PARITY.models.priced.model,
    )
  })
})
