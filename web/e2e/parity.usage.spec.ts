import { expect, type Page, test } from "@playwright/test"

import {
  filterChip,
  login,
  nav,
  pickOption,
  table,
  tableRows,
  tileValue,
} from "./helpers"
import { COUNTS, PARITY } from "./parity-data"

test.describe.configure({ mode: "serial" })

async function openUsage(page: Page): Promise<void> {
  await login(page)
  await nav(page).getByRole("link", { name: "Usage" }).click()
  await expect(
    page.getByRole("heading", { name: "Usage & analytics" }),
  ).toBeVisible()
}

// A breakdown row, located by the name in its row-header cell. The tables are
// spend-ranked, so a row's position moves with the fixture; its name does not.
function breakdownRow(page: Page, dimension: string, name: string) {
  return tableRows(page, `Spend by ${dimension}`).filter({
    has: page.getByRole("rowheader", { name }),
  })
}

test.describe("usage & analytics", () => {
  test("reports the window's spend, tokens, cache and latency", async ({
    page,
  }) => {
    await openUsage(page)

    // The empty state is mutually exclusive with the tiles, so its absence is
    // part of the assertion that the window resolved to real data.
    await expect(page.getByText("No usage yet")).toBeHidden()

    const cost = tileValue(page, "Tracked cost")
    await expect(cost).toHaveText(/^\$/)
    await expect(cost).not.toHaveText("$0.00")

    // Billed tokens, cache and latency each come from a different part of the
    // summary payload, so a dash in any one of them is a distinct regression
    // rather than a repeat of the one above.
    await expect(tileValue(page, "Tokens (billed)")).not.toHaveText("—")
    await expect(tileValue(page, "Cache hit rate")).not.toHaveText("—")
    await expect(tileValue(page, "Avg latency")).not.toHaveText("—")
  })

  test("the chart changes what it measures and how it is split", async ({
    page,
  }) => {
    await openUsage(page)

    // Ungrouped requests split success from error, which is the encoding that
    // makes a failure visible without reading a second chart.
    await page.getByRole("button", { name: "Requests", exact: true }).click()
    await expect(
      page.getByRole("button", { name: "Requests", exact: true }),
    ).toHaveAttribute("aria-pressed", "true")
    await expect(page.getByText("Succeeded", { exact: true })).toBeVisible()
    await expect(page.getByText("Failed", { exact: true })).toBeVisible()

    // Grouping replaces that split with one series per group, named in the
    // legend: the point of the control is that the chart says which model is
    // which, not merely that it restacked.
    await pickOption(page, "Group by", "By model")
    await expect(
      page.getByText(PARITY.models.priced.model, { exact: true }).first(),
    ).toBeVisible()

    // Tokens ungrouped fall back to the billed composition, the same four
    // buckets the Activity row bar uses.
    await pickOption(page, "Group by", "No grouping")
    await page.getByRole("button", { name: "Tokens", exact: true }).click()
    await expect(page.getByText("Fresh input", { exact: true })).toBeVisible()
    await expect(page.getByText("Cache read", { exact: true })).toBeVisible()
  })

  test("breaks spend down by model and by user", async ({ page }) => {
    await openUsage(page)

    // Model is the default primary dimension.
    await expect(
      page.getByRole("heading", { name: "Spend by model" }),
    ).toBeVisible()
    await expect(
      breakdownRow(page, "model", PARITY.models.priced.model),
    ).toBeVisible()
    await expect(
      breakdownRow(page, "model", PARITY.models.priced.model),
    ).toContainText(String(COUNTS.priced))

    await page
      .getByRole("button", { name: "User", exact: true })
      .first()
      .click()
    await expect(
      page.getByRole("heading", { name: "Spend by user" }),
    ).toBeVisible()
    await expect(breakdownRow(page, "user", PARITY.users.heavy)).toBeVisible()
    await expect(breakdownRow(page, "user", PARITY.users.light)).toBeVisible()

    // The secondary strip answers "what ran" rather than "who was billed", and
    // sessions are its default.
    await expect(
      page.getByRole("heading", { name: "Spend by session" }),
    ).toBeVisible()
    await expect(
      breakdownRow(page, "session", PARITY.sessions.heavy),
    ).toBeVisible()

    await page.getByRole("button", { name: "Provider", exact: true }).click()
    await expect(
      page.getByRole("heading", { name: "Spend by provider" }),
    ).toBeVisible()
    await expect(
      breakdownRow(page, "provider", PARITY.models.priced.provider),
    ).toBeVisible()
  })

  test("drills from a model's spend into the requests behind it", async ({
    page,
  }) => {
    await openUsage(page)
    await breakdownRow(page, "model", PARITY.models.priced.model)
      .getByRole("rowheader")
      .click()

    // The drill is the join between the two pages: analytics says where the money
    // went, the log says which requests spent it.
    await expect(page.getByRole("heading", { name: "Activity" })).toBeVisible()
    await expect(
      filterChip(page, "Model", PARITY.models.priced.model),
    ).toBeVisible()
    await expect(tableRows(page, "Activity log")).toHaveCount(COUNTS.priced)
    // The window travels with the filter, so the log shows the same slice of time
    // the chart was showing rather than reverting to its own default.
    await expect(page).toHaveURL(/start_date=/)
  })

  test("drills from a session into the requests it ran", async ({ page }) => {
    await openUsage(page)
    await breakdownRow(page, "session", PARITY.sessions.light)
      .getByRole("rowheader")
      .click()

    await expect(page.getByRole("heading", { name: "Activity" })).toBeVisible()
    await expect(
      filterChip(page, "Session", PARITY.sessions.light),
    ).toBeVisible()
    // The light session carries both its succeeding and its failing requests.
    await expect(tableRows(page, "Activity log")).toHaveCount(
      COUNTS.unpriced + COUNTS.errors,
    )
    await expect(table(page, "Activity log")).toContainText(
      PARITY.models.unpriced.model,
    )
  })
})
