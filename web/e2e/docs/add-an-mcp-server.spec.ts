import { expect, test } from "@playwright/test"

import { login } from "../helpers"
import { docScreenshot, openNestedPage } from "./fixtures"

// The dialog is captured filled in and never submitted, so nothing is added.
test("add an MCP server", async ({ page }) => {
  await login(page)
  await openNestedPage(page, "Tools", "MCP servers")
  await page.getByRole("button", { name: "Add MCP server" }).click()

  const dialog = page.getByRole("dialog")
  await dialog.getByLabel("Name").fill("github")
  await dialog.getByLabel("URL").fill("https://mcp.example.com/github")
  await dialog.getByLabel("Authorization token").fill("ghp_example")
  await dialog
    .getByLabel("Purpose hint")
    .fill("Use for repository and issue lookups.")
  await dialog.getByLabel("Allowed tools").fill("search_issues, get_file")
  await expect(dialog.getByRole("checkbox", { name: "Enabled" })).toBeChecked()
  await expect(
    dialog.getByRole("button", { name: "Add MCP server" }),
  ).toBeEnabled()
  await docScreenshot(dialog, "add-an-mcp-server")
})
