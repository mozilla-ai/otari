import { expect, test } from "@playwright/test"

import { dismissComboBoxInDialog, login, nav } from "../helpers"
import { docScreenshot } from "./fixtures"

// docs/api-keys.md, "Create a key". The dialog is captured filled in and never
// submitted, so nothing is created. The owner is an id no other spec creates:
// an existing one would be offered as a suggestion, and the open list would be
// in the capture.
test("create a key", async ({ page }) => {
  await login(page)
  await nav(page).getByRole("link", { name: "API keys" }).click()
  await page.getByRole("button", { name: "Create key" }).click()

  const dialog = page.getByRole("dialog")
  await expect(dialog.getByLabel("Expires (optional)")).toBeVisible()
  const ownerBox = dialog.getByPlaceholder("Pick a user, or type a new id…")
  await ownerBox.fill("maria@example.com")
  await dismissComboBoxInDialog(ownerBox)
  // After the owner, so the capture is not taken with focus in the combobox.
  await dialog.getByLabel("Name").fill("Production backend")
  await expect(dialog.getByRole("button", { name: "Advanced" })).toBeVisible()
  await expect(dialog.getByRole("button", { name: "Create key" })).toBeVisible()
  await docScreenshot(dialog, "api-keys-create")
})
