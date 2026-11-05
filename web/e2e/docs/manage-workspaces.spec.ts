import { expect, test } from "@playwright/test"

import { login, nav } from "../helpers"
import { docScreenshot } from "./fixtures"

// The dialog is captured filled in and never submitted, so nothing is created.
test("create a workspace", async ({ page }) => {
  await login(page)
  await page.getByRole("link", { name: "Organization", exact: true }).click()
  await nav(page).getByRole("link", { name: "Workspaces" }).click()
  await expect(
    page.getByRole("button", { name: /^Delete Default workspace/ }),
  ).toBeVisible()
  await docScreenshot(page, "manage-workspaces-list")

  await page.getByRole("button", { name: "Create workspace" }).click()
  const dialog = page.getByRole("dialog")
  await dialog.getByLabel("Name").fill("staging")
  await dialog
    .getByLabel("Description (optional)")
    .fill("Pre-production traffic")
  await expect(
    dialog.getByRole("button", { name: "Create workspace" }),
  ).toBeEnabled()
  await docScreenshot(dialog, "manage-workspaces-create")
})
