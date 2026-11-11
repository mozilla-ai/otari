import { expect, type Page, test } from "@playwright/test"

import { login, nav } from "../helpers"
import { call, docScreenshot, listed } from "./fixtures"

const DOMAIN = "docs-example.com"

/** Remove the claim the walk made, so the specs sharing this database never see it. */
async function removeClaim(page: Page): Promise<void> {
  const base = "/organizations/me/domains"
  for (const row of await listed<{ id: string; domain: string }>(page, base)) {
    if (row.domain === DOMAIN) await call(page, "DELETE", `${base}/${row.id}`)
  }
}

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

// docs/members.md, "Let colleagues join by email domain". The claim is made and
// then removed: no DNS record is published, so it is never verified.
test("claim an email domain", async ({ page }) => {
  await removeClaim(page)
  try {
    await login(page)
    await page.getByRole("link", { name: "Organization", exact: true }).click()
    await nav(page).getByRole("link", { name: "Email domains" }).click()
    await page.getByRole("button", { name: "Claim domain" }).click()
    const dialog = page.getByRole("dialog")
    await dialog.getByLabel("Domain").fill(DOMAIN)
    await dialog.getByRole("button", { name: "Claim domain" }).click()
    await expect(dialog).toBeHidden()
    await expect(
      page.getByRole("button", { name: "Verify domain" }),
    ).toBeVisible()
  } finally {
    await removeClaim(page)
  }
})
