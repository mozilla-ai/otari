import { expect, test } from "@playwright/test"

import { login, nav } from "../helpers"
import { docScreenshot } from "./fixtures"

// The quickstart's key step shows the dialog api-keys.spec.ts captures, so this
// walk only checks the labels that step names.
test("the quickstart's pages", async ({ page }) => {
  await login(page)

  await nav(page).getByRole("link", { name: "API keys" }).click()
  await page.getByRole("button", { name: "Create key" }).click()
  const dialog = page.getByRole("dialog")
  await expect(dialog.getByRole("button", { name: "Advanced" })).toBeVisible()
  await dialog.getByRole("button", { name: "Cancel" }).click()

  // Not the Models step: this gateway reaches no provider, so its catalog is
  // empty and a model's page has no "Use this model" to press.
  await nav(page).getByRole("link", { name: "Overview" }).click()
  await page.getByRole("link", { name: "Requests", exact: true }).click()
  await expect(page.getByRole("row").nth(1)).toBeVisible()
  await docScreenshot(page, "quickstart-requests")
})
