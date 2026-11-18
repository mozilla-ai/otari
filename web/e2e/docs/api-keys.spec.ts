import { expect, test } from "@playwright/test"

import { login, nav } from "../helpers"
import { docScreenshot } from "./fixtures"

// docs/api-keys.md, "Create a key". The dialog is captured filled in and never
// submitted, so nothing is created.
test("create a key", async ({ page }) => {
  await login(page)
  await nav(page).getByRole("link", { name: "API keys" }).click()
  await page
    .getByRole("main")
    .getByRole("button", { name: "Create API key" })
    .click()

  const dialog = page.getByRole("dialog")
  await expect(dialog.getByLabel("Expires", { exact: true })).toBeVisible()
  await dialog.getByLabel("Name").fill("Production backend")
  await expect(
    dialog.getByRole("button", { name: "More options" }),
  ).toBeVisible()
  await expect(
    dialog.getByRole("button", { name: "Create API key" }),
  ).toBeVisible()
  await docScreenshot(dialog, "api-keys-create")
})
