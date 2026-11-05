import { expect, test } from "@playwright/test"

import { login, nav } from "../helpers"
import { docScreenshot } from "./fixtures"

// The dialog is captured filled in and never submitted, so nobody is invited.
test("invite a member", async ({ page }) => {
  await login(page)
  await page.getByRole("link", { name: "Organization", exact: true }).click()
  await nav(page).getByRole("link", { name: "Members & roles" }).click()

  await page.getByRole("button", { name: "Invite member" }).click()
  const dialog = page.getByRole("dialog")
  await dialog
    .getByLabel("Email addresses")
    .fill("sam@example.com, priya@example.com")
  await dialog.getByRole("button", { name: /Role/ }).click()
  await page.getByRole("option", { name: "Admin" }).click()
  await expect(
    dialog.getByRole("checkbox", { name: "Default workspace" }),
  ).toBeChecked()
  // The submit counts what it sends: "Invite member" for one address.
  await expect(
    dialog.getByRole("button", { name: "Invite 2 members" }),
  ).toBeEnabled()
  await docScreenshot(dialog, "invite-member")
})
