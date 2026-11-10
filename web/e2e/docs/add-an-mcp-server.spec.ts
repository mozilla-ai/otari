import { expect, type Page, test } from "@playwright/test"

import { login } from "../helpers"
import { call, docScreenshot, listed, openNestedPage } from "./fixtures"

const SERVER = "github"

/** Remove the server the walk added, so the specs sharing this database never see it. */
async function cleanUp(page: Page): Promise<void> {
  const workspaces = await listed<{ id: string; name: string }>(
    page,
    "/workspaces?skip=0&limit=1000",
  )
  const workspace = workspaces.find((w) => w.name === "Default workspace")
  if (!workspace) return
  const base = `/workspaces/${workspace.id}/mcp-servers`
  for (const server of await listed<{ id: string; name: string }>(page, base)) {
    if (server.name === SERVER) {
      await call(page, "DELETE", `${base}/${server.id}`)
    }
  }
}

test("add an MCP server", async ({ page }) => {
  await cleanUp(page)
  try {
    await login(page)
    await openNestedPage(page, "Tools", "MCP servers")
    await page.getByRole("button", { name: "Add MCP server" }).click()

    const dialog = page.getByRole("dialog")
    await dialog.getByLabel("Name").fill(SERVER)
    await dialog.getByLabel("URL").fill("https://example.com/mcp")
    const credentials = dialog.getByRole("radiogroup", { name: "Credentials" })
    await expect(
      credentials.getByRole("radio", { name: "Shared token" }),
    ).toBeChecked()
    await expect(
      credentials.getByRole("radio", { name: "Each user signs in" }),
    ).toBeVisible()
    await dialog.getByLabel("Authorization token").fill("ghp_example")
    await dialog
      .getByLabel("Purpose hint")
      .fill("Use for repository and issue lookups.")
    await dialog.getByLabel("Allowed tools").fill("search_issues, get_file")
    await expect(
      dialog.getByRole("checkbox", { name: "Enabled" }),
    ).toBeChecked()
    const policy = dialog.getByRole("radiogroup", { name: "Default policy" })
    await expect(policy.getByRole("radio", { name: "Allow" })).toBeChecked()
    await expect(policy.getByRole("radio", { name: "Deny" })).toBeVisible()
    await docScreenshot(dialog, "add-an-mcp-server")

    await dialog.getByRole("button", { name: "Add MCP server" }).click()
    await expect(dialog).toBeHidden()
    // Saving opens the server's own page, where its access rules live.
    await expect(
      page.getByRole("heading", { name: SERVER, level: 1 }),
    ).toBeVisible()
  } finally {
    await cleanUp(page)
  }
})
