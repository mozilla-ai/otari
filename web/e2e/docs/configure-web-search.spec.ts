import { expect, test } from "@playwright/test"

import { login } from "../helpers"
import { docScreenshot, openNestedPage } from "./fixtures"

// The e2e gateway names a search backend and turns fetching on
// (e2e/otari.yml), so the page's controls are live. Nothing is searched.
test("a workspace's web search policy", async ({ page }) => {
  await login(page)
  await openNestedPage(page, "Tools", "Web search")

  await expect(page.getByText("Allow in Default workspace")).toBeVisible()
  for (const label of [
    "Search with",
    "Fetch with",
    "Who runs provider web search",
    "Who runs provider web fetch",
  ]) {
    await expect(page.getByText(label, { exact: true })).toBeVisible()
  }
  await expect(
    page.getByRole("button", { name: "Not set Who runs provider web search" }),
  ).toBeVisible()
  await docScreenshot(page, "configure-web-search")

  await page.getByRole("button", { name: "Advanced" }).click()
  for (const label of [
    "Max results for this workspace",
    "Allowed domains",
    "Blocked domains",
  ]) {
    await expect(page.getByText(label).first()).toBeVisible()
  }
})
