import { expect, type Page, test } from "@playwright/test"

import { API_ROOT } from "@/shared/api/client"

import { authHeaders, expectOk, login } from "../helpers"
import { PARITY } from "../parity-data"
import { call, docScreenshot } from "./fixtures"

const KEYS = "/organizations/me/provider-keys"

/**
 * An organization key offering the seeded model, so the workspace has a model
 * to chat with. The Playground lists only what its workspace may call, and a
 * pricing row alone is not that. The credential is fake: nothing is sent.
 * Returns the key's id, for the clean-up.
 */
async function offerModel(page: Page): Promise<string> {
  const { provider, model } = PARITY.models.priced
  const created = await page.request.post(`${API_ROOT}${KEYS}`, {
    headers: authHeaders,
    data: {
      provider,
      name: "playground",
      api_key: "sk-docs-playground-not-a-secret",
    },
  })
  await expectOk(created, `POST ${KEYS}`)
  const { id } = (await created.json()) as { id: string }
  await call(page, "POST", `${KEYS}/${id}/models`, { model })
  return id
}

/** A key is deleted once archived. */
async function removeKey(page: Page, id: string): Promise<void> {
  await call(page, "POST", `${KEYS}/${id}/archive`)
  await call(page, "DELETE", `${KEYS}/${id}`)
}

// docs/playground.md. The page's controls are checked by the labels the steps
// give them; sending a prompt needs a provider this gateway does not have, so
// the capture is the page a reader lands on.
test("the playground", async ({ page }) => {
  const keyId = await offerModel(page)
  try {
    await walkThePlayground(page)
  } finally {
    await removeKey(page, keyId)
  }
})

async function walkThePlayground(page: Page): Promise<void> {
  await login(page)
  await page.getByRole("link", { name: "Playground", exact: true }).click()

  await expect(page.getByRole("radio", { name: "Single" })).toBeChecked()
  await expect(page.getByRole("radio", { name: "Compare" })).toBeVisible()
  await expect(page.getByText("Starting points")).toBeVisible()
  await expect(page.getByRole("button", { name: "Add tools" })).toContainText(
    "Tools",
  )
  // The capture shows the model the key above offers, not one from elsewhere.
  await expect(page.getByRole("button", { name: "Model" })).toHaveText(
    `${PARITY.models.priced.provider}/${PARITY.models.priced.model}`,
  )
  await docScreenshot(page, "playground-single")

  await page.getByRole("button", { name: "History" }).click()
  const history = page.getByRole("dialog", { name: "History" })
  for (const filter of ["All", "Chats", "Comparisons"]) {
    await expect(history.getByRole("radio", { name: filter })).toBeVisible()
  }
  await expect(history.getByLabel("Search history")).toBeVisible()
  await expect(history.getByRole("button", { name: "View all" })).toBeVisible()
}
