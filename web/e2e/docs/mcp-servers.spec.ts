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
    const account = dialog.getByRole("radiogroup", { name: "Sign-in mode" })
    await expect(
      account.getByRole("radio", { name: /^Shared account/ }),
    ).toBeChecked()
    await expect(
      account.getByRole("radio", { name: /^Personal accounts/ }),
    ).toBeVisible()
    await dialog.getByLabel("Authorization token").fill("ghp_example")
    await dialog
      .getByLabel("Purpose hint")
      .fill("Use for repository and issue lookups.")
    await dialog.getByLabel("Allowed tools").fill("search_issues, get_file")
    const availability = dialog.getByRole("radiogroup", {
      name: "Availability",
    })
    await expect(
      availability.getByRole("radio", { name: /^On request/ }),
    ).toBeChecked()
    await expect(
      availability.getByRole("radio", { name: /^Every request/ }),
    ).not.toBeChecked()
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
    await expect(
      page.getByRole("heading", { name: "Access rules" }),
    ).toBeVisible()
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
    await page.getByLabel("Add a tool name").fill(TOOL)
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

/** A catalog entry as the gateway lists one. */
function catalogEntry(name: string, title: string, publisher: string) {
  return {
    id: crypto.randomUUID(),
    name,
    title,
    description: `${title}'s remote MCP server.`,
    url: `https://mcp.${title.toLowerCase()}.example/mcp`,
    requires_token: false,
    website_url: null,
    icon_url: null,
    logo_url: null,
    logo_tone: null,
    publisher_kind: publisher,
    version: "1.0.0",
    source: "registry",
  }
}

const LINEAR = catalogEntry("app.linear/linear", "Linear", "vendor")
const INTERCOM = catalogEntry(
  "io.github.someone/intercom",
  "Intercom",
  "community",
)

/**
 * Stand in for a synced catalog. This gateway never synced, and syncing would
 * call the MCP Registry, so the catalog's reads and the add's connection checks
 * are answered here; nothing in this walk stores a server.
 */
async function fakeCatalog(page: Page): Promise<void> {
  await page.route(`**${API_ROOT}/mcp-catalog/sync`, (route) =>
    route.fulfill({
      json: {
        mode: "manual",
        running: false,
        requested: false,
        outcome: "complete",
        error: null,
        started_at: null,
        finished_at: null,
        last_completed_at: new Date().toISOString(),
        next_sync_at: null,
        pages: 1,
        fetched: 2,
        kept: 2,
        removed: 0,
        server_count: 2,
        is_stale: false,
        next_request_at: null,
      },
    }),
  )
  await page.route(`**${API_ROOT}/mcp-catalog/popular?*`, (route) =>
    route.fulfill({ json: { data: [LINEAR], count: 1 } }),
  )
  await page.route(`**${API_ROOT}/mcp-catalog/servers?*`, (route) =>
    route.fulfill({ json: { data: [INTERCOM], count: 1 } }),
  )
  await page.route(`**${API_ROOT}/workspaces/*/mcp-servers/check`, (route) =>
    route.fulfill({
      json: {
        status: "healthy",
        checked_at: new Date().toISOString(),
        error: null,
        http_status: null,
        tool_count: 1,
        missing_allowed_tools: [],
        tools: ["search"],
      },
    }),
  )
}

test("browse the catalog", async ({ page }) => {
  await fakeCatalog(page)
  await login(page)
  await openNested(page, "Tools", "MCP servers")
  await page.getByRole("button", { name: "Browse catalog" }).click()

  const dialog = page.getByRole("dialog", { name: "Browse catalog" })
  const popular = dialog.getByRole("list", { name: "Popular" })
  // Linear runs its own server on its own domain, which its Official chip says.
  await expect(
    popular
      .getByRole("listitem")
      .filter({ hasText: "Official" })
      .filter({ hasText: "its publisher's own domain" }),
  ).toBeVisible()
  await expect(
    dialog.getByRole("button", { name: "Add it by URL" }),
  ).toBeVisible()
  // `force`: the real input sits behind the box that draws it (dashboard.spec.ts
  // says why), so a plain press waits on it forever. A press rather than
  // `check`, which would wait on a row the pick has just moved out of the list.
  await popular
    .getByRole("checkbox", { name: "Select Linear" })
    .click({ force: true })
  // A pick moves up to Selected, where it stays through a search.
  await expect(
    dialog
      .getByRole("list", { name: "Selected" })
      .getByText("Linear", { exact: true }),
  ).toBeVisible()
  await expect(
    dialog.getByRole("button", { name: "Add", exact: true }),
  ).toBeEnabled()
  await dialog
    .getByRole("searchbox", { name: "Search the catalog" })
    .fill("intercom")
  const results = dialog.getByRole("list", { name: "Results" })
  // Someone other than Intercom publishes this one, so it carries no chip.
  await expect(results.getByText("Official")).toBeHidden()
  await results
    .getByRole("checkbox", { name: "Select Intercom" })
    .click({ force: true })
  await dialog.getByRole("button", { name: "Add 2 servers" }).click()

  const confirm = page.getByRole("dialog")
  const intercom = confirm.getByRole("region", { name: "Intercom" })
  await expect(
    intercom.getByRole("radiogroup", { name: "Sign-in mode" }),
  ).toBeVisible()
  const availability = intercom.getByRole("radiogroup", {
    name: "Availability",
  })
  await expect(
    availability.getByRole("radio", { name: /^On request/ }),
  ).toBeChecked()
  for (const choice of [/^Off/, /^Every request/]) {
    await expect(
      availability.getByRole("radio", { name: choice }),
    ).toBeVisible()
  }
  await expect(confirm.getByRole("button", { name: "Confirm" })).toBeEnabled()
  // Not confirmed: the outcomes ("Connected", "Added", "Needs sign-in", "Check
  // failed", "Not added") are covered by McpCatalogDialog.test.tsx, and this
  // walk stores nothing.
  await confirm.getByRole("button", { name: "Back" }).click()
  await page.getByRole("dialog").getByRole("button", { name: "Cancel" }).click()
  await expect(page.getByRole("dialog")).toBeHidden()
})

test("an empty catalog offers to load it", async ({ page }) => {
  await login(page)
  await openNested(page, "Tools", "MCP servers")
  await page.getByRole("button", { name: "Browse catalog" }).click()

  const dialog = page.getByRole("dialog", { name: "Browse catalog" })
  // This gateway never synced. Loading is not pressed: it would call the registry.
  await expect(
    dialog.getByText(/The catalog has not been loaded yet/),
  ).toBeVisible()
  await expect(
    dialog.getByRole("button", { name: "Load catalog" }),
  ).toBeVisible()
  await expect(
    dialog.getByRole("button", { name: "Add", exact: true }),
  ).toBeDisabled()
  await dialog.getByRole("button", { name: "Cancel" }).click()
  await expect(dialog).toBeHidden()
})
