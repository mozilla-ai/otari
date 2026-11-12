import { expect, type Page, test } from "@playwright/test"

import { API_ROOT } from "@/shared/api/client"

import {
  authHeaders,
  dismissComboBox,
  expectOk,
  login,
  openNested,
} from "../helpers"
import { call, docScreenshot, listed } from "./fixtures"

const SERVER = "github"
const KEY = "docs-mcp-reader"
const TOOL = "search_issues"

/** A key in the default workspace for the rule and the trace to name. Returns its id. */
async function createKey(page: Page): Promise<string> {
  const created = await page.request.post(`${API_ROOT}/keys`, {
    headers: authHeaders,
    data: { key_name: KEY, user_id: "docs-mcp@example.com" },
  })
  await expectOk(created, "POST /keys")
  const { id } = (await created.json()) as { id: string }
  return id
}

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

// docs/mcp-servers.md: "Connect a server", then each step of "Control
// who can use it" on the page saving opens.
test("configure an MCP server", async ({ page }) => {
  // Five dialogs and a trace, against a gateway it boots from cold.
  test.setTimeout(60_000)
  await cleanUp(page)
  const keyId = await createKey(page)
  try {
    await login(page)
    await openNested(page, "Tools", "MCP servers")
    await page.getByRole("button", { name: "Add MCP server" }).click()

    const dialog = page.getByRole("dialog")
    await dialog.getByLabel("Name").fill(SERVER)
    await dialog.getByLabel("URL").fill("https://example.com/mcp")
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
    await docScreenshot(dialog, "mcp-servers")

    await dialog.getByRole("button", { name: "Add MCP server" }).click()
    await expect(dialog).toBeHidden()
    // Saving opens the server's own page, where its access rules live.
    await expect(
      page.getByRole("heading", { name: SERVER, level: 1 }),
    ).toBeVisible()

    // Choose the default policy.
    await expect(page.getByRole("heading", { name: "ACL" })).toBeVisible()
    await page.getByRole("button", { name: "Edit the default policy" }).click()
    let rule = page.getByRole("dialog")
    let action = rule.getByRole("radiogroup", { name: "Action" })
    // The input is visually hidden under its label, which takes the press.
    await action.getByText("Deny", { exact: true }).click()
    await expect(action.getByRole("radio", { name: "Deny" })).toBeChecked()
    await rule.getByRole("button", { name: "Save rule" }).click()
    await expect(rule).toBeHidden()

    // Let a key only read: label a read tool first.
    await expect(
      page.getByRole("heading", { name: "Read tools" }),
    ).toBeVisible()
    await page.getByRole("button", { name: "Edit read tools" }).click()
    await page.getByLabel("Tool name").fill(TOOL)
    await page.getByRole("button", { name: "Add", exact: true }).click()
    await page.getByRole("button", { name: "Save read tools" }).click()
    await expect(
      page.getByRole("button", { name: "Edit read tools" }),
    ).toBeVisible()

    // Add a rule for a key.
    await page.getByRole("button", { name: "Add rule" }).click()
    rule = page.getByRole("dialog", { name: "Add rule" })
    await rule.getByRole("searchbox", { name: "Source" }).fill(KEY)
    await rule.getByText(KEY, { exact: true }).click()
    await expect(
      rule.getByRole("checkbox", { name: new RegExp(`^Key ${KEY} of`) }),
    ).toBeChecked()
    action = rule.getByRole("radiogroup", { name: "Action" })
    for (const choice of ["Allow any tool", "Allow specific tools…", "Deny"]) {
      await expect(action.getByRole("radio", { name: choice })).toBeVisible()
    }
    await action.getByText("Read tools only", { exact: true }).click()
    await rule.getByRole("button", { name: "Add rule" }).click()
    await expect(rule).toBeHidden()

    // Check what a key can reach.
    await expect(
      page.getByRole("heading", { name: "Trace a call" }),
    ).toBeVisible()
    // The option reads "<key> (<owner>)", so it is matched by its start.
    await page.getByRole("button", { name: /Source key$/ }).click()
    await page.getByRole("option", { name: new RegExp(`^${KEY} \\(`) }).click()
    const tool = page.getByRole("combobox", { name: "Tool" })
    await tool.fill(TOOL)
    await dismissComboBox(tool)
    await page.getByRole("button", { name: "Trace", exact: true }).click()
    await expect(page.getByRole("status")).toContainText(
      "ALLOW, matched the rule for",
    )
  } finally {
    await cleanUp(page)
    await call(page, "DELETE", `/keys/${keyId}`)
  }
})

test("browse the catalog", async ({ page }) => {
  await login(page)
  await openNested(page, "Tools", "MCP servers")
  await page.getByRole("button", { name: "Browse catalog" }).click()

  const dialog = page.getByRole("dialog", { name: "Browse catalog" })
  await expect(
    dialog.getByRole("searchbox", { name: "Search the catalog" }),
  ).toBeVisible()
  // This gateway never synced, so the list says why it is empty and nothing
  // can be reviewed.
  await expect(dialog.getByText(/The catalog is empty/)).toBeVisible()
  await expect(dialog.getByRole("button", { name: "Review" })).toBeDisabled()
  // The outcomes the page lists, in the words the dialog reports them with.
  // "Connected", "Needs sign-in", "Check failed", "Not added" and "Allow" are
  // covered by McpCatalogDialog.test.tsx and mcpCatalog.test.ts.
  await dialog.getByRole("button", { name: "Cancel" }).click()
  await expect(dialog).toBeHidden()
})
