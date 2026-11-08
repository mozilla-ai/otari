import { expect, type Locator, type Page, test } from "@playwright/test"

import { login, nav } from "../helpers"
import { docScreenshot } from "./fixtures"

/** Pick the first option of a select, by the select's label. */
async function pickFirst(page: Page, dialog: Locator, label: RegExp) {
  await dialog.getByRole("button", { name: label }).click()
  await page.getByRole("option").first().click()
}

// Both dialogs are captured filled in and never submitted.
test("configure and mandate a guardrail", async ({ page }) => {
  await login(page)
  await page.getByRole("link", { name: "Organization", exact: true }).click()
  await nav(page).getByRole("link", { name: "Guardrails" }).click()

  await page.getByRole("button", { name: "Configure guardrail" }).click()
  const definition = page.getByRole("dialog")
  await pickFirst(page, definition, /What do you want checked\?/)
  await pickFirst(page, definition, /Which guardrail\?/)
  await definition.getByLabel("Name", { exact: true }).fill("content-check")
  await expect(
    definition.getByRole("button", { name: "Configure guardrail" }),
  ).toBeVisible()
  await docScreenshot(definition, "manage-guardrails-configure")
  await definition.getByRole("button", { name: "Cancel" }).click()
  await definition.getByRole("button", { name: "Discard" }).click()

  await page.getByRole("button", { name: "Mandate a guardrail" }).click()
  const mandate = page.getByRole("dialog")
  await mandate.getByLabel("Guardrail profile").fill("pii-strict")
  await mandate
    .getByLabel("Endpoint", { exact: true })
    .fill("https://guardrails.example.com")
  await expect(mandate.getByRole("button", { name: /Mode$/ })).toBeVisible()
  await expect(
    mandate.getByRole("button", { name: /If unreachable$/ }),
  ).toBeVisible()
  await expect(
    mandate.getByRole("checkbox", { name: "Default workspace" }),
  ).toBeVisible()
  await docScreenshot(mandate, "manage-guardrails-mandate")
})
