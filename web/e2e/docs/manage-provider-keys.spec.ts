import { expect, type Page, test } from "@playwright/test"

import { dismissComboBoxInDialog, login, nav } from "../helpers"
import { call, docScreenshot, listed } from "./fixtures"

const KEYS = ["production", "backup"] as const

/** Archive and delete the keys the walk added: a key is deleted once archived. */
async function cleanUp(page: Page): Promise<void> {
  const base = "/organizations/me/provider-keys"
  for (const key of await listed<{ id: string; name: string }>(page, base)) {
    if ((KEYS as readonly string[]).includes(key.name)) {
      await call(page, "POST", `${base}/${key.id}/archive`)
      await call(page, "DELETE", `${base}/${key.id}`)
    }
  }
}

async function openProviders(page: Page): Promise<void> {
  await page.getByRole("link", { name: "Organization", exact: true }).click()
  await nav(page).getByRole("button", { name: "Providers" }).click()
  await nav(page).getByRole("link", { name: "Inference" }).click()
}

// Each key's credential is fake: adding a key tries the provider for its model
// list and keeps the key when that fails.
test("add provider keys and choose which ones a workspace uses", async ({
  page,
}) => {
  await cleanUp(page)
  try {
    await login(page)
    await openProviders(page)

    for (const name of KEYS) {
      await page.getByRole("button", { name: "Add provider key" }).click()
      const dialog = page.getByRole("dialog")
      const provider = dialog.getByRole("combobox", { name: "Provider" })
      await provider.fill("OpenAI")
      await page.getByRole("option", { name: "OpenAI", exact: true }).click()
      await dismissComboBoxInDialog(provider)
      if (name === "production") {
        // The prefix is asked for only once the key has an endpoint of its own.
        const apiBase = dialog.getByLabel("API base URL")
        await apiBase.fill("https://llm.example.com/v1")
        await expect(dialog.getByLabel("Request prefix")).toBeVisible()
        await apiBase.fill("")
      }
      await dialog.getByLabel("Name").fill(name)
      await dialog.getByLabel("API key").fill("sk-docs-example-not-a-real-key")
      if (name === "production") {
        await docScreenshot(dialog, "manage-provider-keys-add")
      }
      await dialog.getByRole("button", { name: "Add provider key" }).click()
      await expect(dialog).toBeHidden()
    }

    await page
      .getByRole("row", { name: /production/ })
      .getByRole("button", { name: "Make default provider" })
      .click()
    await expect(
      page.getByRole("button", { name: "Default provider, click to remove" }),
    ).toBeVisible()
    await docScreenshot(page, "manage-provider-keys-list")

    await nav(page).getByRole("link", { name: "Workspaces" }).click()
    await page
      .getByRole("main")
      .getByRole("link", { name: "Default workspace", exact: true })
      .click()
    await page
      .getByRole("main")
      .getByRole("button", { name: "Providers" })
      .click()
    // A provider's keys fold under one row until it is opened.
    await page.getByRole("button", { name: /^openai 2 keys/ }).click()
    await expect(page.getByText("backup").first()).toBeVisible()
    await docScreenshot(page, "manage-provider-keys-workspace")

    await page.getByRole("button", { name: "More for openai / backup" }).click()
    await expect(
      page.getByRole("menuitem", { name: "Always use this key" }),
    ).toBeVisible()
    await page.keyboard.press("Escape")
  } finally {
    await cleanUp(page)
  }
})
