import { expect, test } from "@playwright/test"

import { login, nav, pickOption, table, tableRows } from "./helpers"
import { PARITY } from "./parity-data"

// The seeded fixture is shared, and nothing here mutates it, but the suite runs
// one worker against one gateway either way (see web/playwright.config.ts).
test.describe.configure({ mode: "serial" })

test.describe("Home and Sessions", () => {
  test("opens on Home, with no getting-started screen on a working gateway", async ({
    page,
  }) => {
    await login(page)
    await expect(
      page.getByRole("main").getByRole("heading", { name: "Home" }),
    ).toBeVisible()
    // Onboarding is an empty-gateway state. With providers configured and
    // usage recorded it has to be gone, or a working gateway keeps being told
    // to set itself up.
    await expect(page.getByText("Get started with Otari")).toBeHidden()
  })

  test("a window picked on one view carries to the other", async ({ page }) => {
    await login(page)
    await nav(page).getByRole("link", { name: "Sessions" }).click()
    const views = page.getByRole("navigation", { name: "Sessions views" })
    await views.getByRole("link", { name: "Requests" }).click()
    await expect(page).toHaveURL(/\/requests/)

    await pickOption(page, "Time range", "7d")
    await expect(page).toHaveURL(/\/requests\?.*range=7d/)

    await views.getByRole("link", { name: "Sessions" }).click()
    await expect(page).toHaveURL(/\/sessions\?.*range=7d/)
  })

  test("the Requests view lists the newest requests first", async ({
    page,
  }) => {
    await login(page)
    await nav(page).getByRole("link", { name: "Sessions" }).click()
    await page
      .getByRole("navigation", { name: "Sessions views" })
      .getByRole("link", { name: "Requests" })
      .click()
    await expect(page).toHaveURL(/\/requests/)
    await expect(table(page, "Activity log")).toBeVisible()

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
