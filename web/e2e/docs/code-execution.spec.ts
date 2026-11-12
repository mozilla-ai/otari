import { expect, test } from "@playwright/test"

import { login, openNested } from "../helpers"
import { docScreenshot } from "./fixtures"

// The e2e gateway names a sandbox (e2e/otari.yml), so the policy is live. No
// code runs.
test("a workspace's code execution policy", async ({ page }) => {
  await login(page)
  await openNested(page, "Tools", "Code execution")

  await expect(page.getByText("Allow in Default workspace")).toBeVisible()
  await expect(
    page.getByRole("switch", { name: "Allow code execution" }),
  ).toBeEnabled()
  await expect(
    page.getByText("Who runs provider code tools", { exact: true }).first(),
  ).toBeVisible()
  await docScreenshot(page, "code-execution")
})
