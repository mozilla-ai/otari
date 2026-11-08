import { expect, test } from "@playwright/test"

import { login } from "../helpers"
import { docScreenshot } from "./fixtures"

// docs/playground.md. The page's controls are checked by the labels the steps
// give them; sending a prompt needs a provider this gateway does not have, so
// the capture is the page a reader lands on.
test("the playground", async ({ page }) => {
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
