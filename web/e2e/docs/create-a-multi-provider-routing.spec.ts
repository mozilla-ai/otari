import { expect, test } from "@playwright/test"

import { dismissComboBoxInDialog, login, nav, openNested } from "../helpers"
import { docScreenshot } from "./fixtures"

// The guide's provider key and API key steps are the dialogs the
// manage-provider-keys and api-keys specs capture; this walk checks the labels
// those steps name and captures the policy dialog, which it never submits.
test("create a fallback policy", async ({ page }) => {
  await login(page)

  await page.getByRole("link", { name: "Organization", exact: true }).click()
  await nav(page).getByRole("button", { name: "Providers" }).click()
  await nav(page).getByRole("link", { name: "Inference" }).click()
  await page.getByRole("button", { name: "Add provider key" }).click()
  const keyDialog = page.getByRole("dialog")
  const provider = keyDialog.getByRole("combobox", { name: "Provider" })
  for (const name of ["OpenAI", "Anthropic"]) {
    await provider.fill(name.toLowerCase())
    await expect(
      page.getByRole("option", { name: new RegExp(name, "i") }).first(),
    ).toBeVisible()
  }
  await dismissComboBoxInDialog(provider)
  await keyDialog.getByRole("button", { name: "Cancel" }).click()
  await page.getByRole("link", { name: /^Back to / }).click()

  await nav(page).getByRole("link", { name: "API keys" }).click()
  await expect(page.getByRole("button", { name: "Create key" })).toBeVisible()

  await openNested(page, "Routing", "Policies")
  await page.getByRole("button", { name: "Create policy" }).first().click()
  const dialog = page.getByRole("dialog")
  for (const kind of [
    "Split traffic",
    "Spill over at rate limits",
    "Tier down on budget",
  ]) {
    await expect(dialog.getByRole("radio", { name: kind })).toBeVisible()
  }
  await expect(dialog.getByRole("radio", { name: "Fallback" })).toBeChecked()
  await docScreenshot(dialog, "routing-policy-kind")
  await dialog.getByRole("button", { name: "Continue" }).click()

  await dialog.getByRole("textbox", { name: /^Name/ }).fill("production-chat")
  const main = dialog.getByRole("combobox", { name: "Main model" })
  await main.fill("openai:gpt-5-mini")
  await dismissComboBoxInDialog(main)
  await expect(dialog.getByText("If unavailable, use")).toBeVisible()
  const backup = dialog.getByRole("combobox", { name: "Backup 1" })
  await backup.fill("anthropic:claude-haiku-4-5")
  await dismissComboBoxInDialog(backup)
  await docScreenshot(dialog, "routing-policy-fallback")
})
