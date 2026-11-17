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

  test("reports the window's spend, tokens and failures, and where it went", async ({
    page,
  }) => {
    await login(page)
    // The value beside each figure's label. The labels name the window, so
    // they are matched by shape rather than by a day count.
    const figure = (label: RegExp) =>
      page
        .getByRole("main")
        .getByText(label)
        .locator("xpath=following-sibling::span[1]")

    const spent = figure(/^Spent(, \d+ days| this month)$/)
    await expect(spent).toHaveText(/^\$/)
    await expect(spent).not.toHaveText("$0.00")
    // Tokens and failures come from different parts of the summary payload, so
    // a dash in either is a regression of its own.
    await expect(figure(/^Tokens(, \d+ days| this month)$/)).not.toHaveText("—")
    await expect(figure(/^Failures$/)).not.toHaveText("—")

    const where = page.getByRole("region", { name: "Where it went" })
    // The radio itself is visually hidden behind its label, which is what a
    // person presses.
    await where
      .getByRole("radiogroup", { name: "Break down by" })
      .getByText("Models", { exact: true })
      .click()
    await expect(
      where.getByRole("radio", { name: "Models", exact: true }),
    ).toBeChecked()
    await expect(tableRows(page, "Spend by models").first()).toBeVisible()
    await expect(where).toContainText(PARITY.models.priced.model)
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
