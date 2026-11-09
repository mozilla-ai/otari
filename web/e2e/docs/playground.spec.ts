import { expect, type Page, test } from "@playwright/test"

import { login } from "../helpers"
import { PARITY } from "../parity-data"
import { call, docScreenshot, listed } from "./fixtures"

const KEY_NAME = "playground"
const KEYS = "/organizations/me/provider-keys"

/**
 * An organization key offering the seeded model, so the workspace has a model
 * to chat with. The Playground lists only what its workspace may call, and a
 * pricing row alone is not that. The credential is fake: nothing is sent.
 */
async function offerModel(page: Page): Promise<void> {
  const { provider, model } = PARITY.models.priced
  await call(page, "POST", KEYS, {
    provider,
    name: KEY_NAME,
    api_key: "sk-docs-playground-not-a-secret",
  })
  const key = await keyId(page)
  if (key) await call(page, "POST", `${KEYS}/${key}/models`, { model })
}

async function keyId(page: Page): Promise<string | undefined> {
  const keys = await listed<{ id: string; name: string }>(page, KEYS)
  return keys.find((key) => key.name === KEY_NAME)?.id
}

/** A key is deleted once archived. */
async function cleanUp(page: Page): Promise<void> {
  const key = await keyId(page)
  if (!key) return
  await call(page, "POST", `${KEYS}/${key}/archive`)
  await call(page, "DELETE", `${KEYS}/${key}`)
}

test.afterEach(async ({ page }) => {
  await cleanUp(page)
})

// docs/playground.md. The page's controls are checked by the labels the steps
// give them; sending a prompt needs a provider this gateway does not have, so
// the capture is the page a reader lands on.
test("the playground", async ({ page }) => {
  await offerModel(page)
  await login(page)
  await page.getByRole("link", { name: "Playground", exact: true }).click()

  await expect(page.getByRole("radio", { name: "Single" })).toBeChecked()
  await expect(page.getByRole("radio", { name: "Compare" })).toBeVisible()
  await expect(page.getByText("Starting points")).toBeVisible()
  await expect(page.getByRole("button", { name: "Add tools" })).toContainText(
    "Tools",
  )
  await docScreenshot(page, "playground-single")

  await page.getByRole("button", { name: "History" }).click()
  const history = page.getByRole("dialog", { name: "History" })
  for (const filter of ["All", "Chats", "Comparisons"]) {
    await expect(history.getByRole("radio", { name: filter })).toBeVisible()
  }
  await expect(history.getByLabel("Search history")).toBeVisible()
  await expect(history.getByRole("button", { name: "View all" })).toBeVisible()
})
