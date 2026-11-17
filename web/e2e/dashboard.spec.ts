import { expect, test } from "@playwright/test"
import { API_ROOT } from "@/shared/api/client"
import {
  dismissComboBoxInDialog,
  expectedKeyFingerprint,
  gotoRoute,
  login,
  MASTER_KEY,
  nav,
  openAccountMenu,
  openNested,
  openOrganization,
  openTab,
  pageHeading,
  pressRowAction,
} from "./helpers"

// One shared gateway + DB, so the flows build on each other and must run in order.
test.describe.configure({ mode: "serial" })

test.describe("dashboard core flows", () => {
  test("first-run Home guides the operator to provider setup", async ({
    page,
  }) => {
    await login(page)
    await expect(page.getByRole("heading", { name: "Home" })).toBeVisible()
    await expect(page.getByText("Get started with Otari")).toBeVisible()
  })

  test("add a provider from onboarding, and it opens on its own page", async ({
    page,
  }) => {
    await login(page)
    await page.getByRole("button", { name: "Add your first provider" }).click()
    await expect(page.getByText("Welcome to Otari")).toBeVisible()

    await page.getByRole("button", { name: "Add your first provider" }).click()
    const dialog = page.getByRole("dialog", { name: "Add hosted provider" })
    await dialog.getByRole("combobox", { name: "Provider" }).fill("OpenAI")
    await page.getByRole("option", { name: "OpenAI", exact: true }).click()
    await dialog.getByRole("textbox", { name: "Name" }).fill("e2e-llm")
    await dialog
      .getByRole("textbox", { name: "API base URL" })
      .fill("http://e2e-box:8000/v1")
    await dialog.getByRole("button", { name: "Add provider" }).click()

    await expect(pageHeading(page, "e2e-llm")).toBeVisible({ timeout: 30_000 })
    // Onboarding clears once a provider exists.
    await expect(page.getByText("Welcome to Otari")).toBeHidden()
  })

  // Runs here, right after a provider exists, because that is the deployment
  // state the guide is for, and it ends by skipping: the dismissal is
  // per-workspace and permanent, so every project after this one sees the
  // Overview without it (parity.setup.ts asks for the same thing again, rather
  // than depending on this spec having run).
  test("the first-run sheet hands out a key, then skips for good", async ({
    page,
  }) => {
    await login(page)

    // A sheet over the Overview, not a panel on it: a workspace with no traffic
    // has nothing on that page worth reading yet.
    const sheet = page.getByRole("dialog", { name: "Send your first request" })
    await expect(sheet).toBeVisible()

    // The key is minted on a press, never by the sheet opening (#3000). Shown
    // once, in a labeled field they can select and copy, and concealed there
    // until they ask for it (otari-ai#2111).
    const mintedKey = page.waitForResponse(
      (response) =>
        response.url().endsWith("/activation/key") &&
        response.request().method() === "POST" &&
        response.ok(),
    )
    await sheet.getByRole("button", { name: "Create API key" }).click()
    const { key: fullKey } = await (await mintedKey).json()
    // By role, not by label: `getByLabel` matches a substring, so it also
    // picks up the field's own "Show API key" toggle.
    const key = sheet.getByRole("textbox", { name: "Your API key" })
    await expect(key).toBeVisible()
    // The fingerprint the key-creation dialog shows, derived from the key the
    // mint returned rather than matched by shape: a stand-in built from another
    // key would have the same shape.
    const concealedKey = expectedKeyFingerprint(fullKey)
    await expect(key).toHaveValue(concealedKey)

    // The examples are tabs, and the agent prompt is the one offered first.
    await expect(
      sheet.getByRole("region", { name: "agent code" }),
    ).toBeVisible()
    await sheet.getByRole("button", { name: "cURL" }).click()
    const curl = sheet.getByRole("region", { name: "curl code" })
    await expect(curl).toBeVisible()
    // Concealed in the example too, because it is the same secret.
    await expect(curl).toContainText(`export OTARI_API_KEY='${concealedKey}'`)
    await expect(curl).not.toContainText(fullKey)

    await sheet.getByRole("button", { name: "Show Your API key" }).click()
    await expect(key).toHaveValue(fullKey)
    await expect(curl).toContainText(`export OTARI_API_KEY=${fullKey}`)
    await sheet.getByRole("button", { name: "Hide Your API key" }).click()
    await expect(key).toHaveValue(concealedKey)
    await expect(curl).toContainText(`export OTARI_API_KEY='${concealedKey}'`)
    await expect(curl).not.toContainText(fullKey)

    // It is watching for the request while all of that is on screen.
    await expect(
      sheet.getByText("Listening for your first request"),
    ).toBeVisible()

    await sheet.getByRole("button", { name: "Skip" }).click()
    await expect(sheet).toBeHidden()

    // Permanent: the offer does not come back on the next page load.
    await page.reload()
    await expect(page.getByRole("heading", { name: "Home" })).toBeVisible()
    await expect(sheet).toBeHidden()
  })

  test("navigate the management pages", async ({ page }) => {
    await login(page)
    // The workspace rail, then the organization one. The sidebar label and the
    // page heading are no longer always the same word, so both are named.
    await nav(page).getByRole("link", { name: "Models", exact: true }).click()
    await expect(pageHeading(page, "Models")).toBeVisible()

    // Routing is a tab of Rules, which titles the page, and each tool a view
    // of Tools.
    await openTab(page, "Rules", "Routing")
    await expect(pageHeading(page, "Rules")).toBeVisible()
    await openTab(page, "Tools", "Web search")
    await expect(pageHeading(page, "Tools")).toBeVisible()
    // The one Tools view that is not a filtered view of the settings page, so
    // reaching it proves its own route resolves rather than that the filter did.
    await openTab(page, "Tools", "MCP servers")
    await expect(
      page.getByRole("button", { name: "Add MCP server" }),
    ).toBeVisible()

    await openOrganization(page)
    await openNested(page, "Providers", "Inference")
    await expect(pageHeading(page, "Inference providers")).toBeVisible()
    await openNested(page, "Providers", "Web search")
    await expect(pageHeading(page, "Web search providers")).toBeVisible()
    await nav(page)
      .getByRole("link", { name: "Spend & budgets", exact: true })
      .click()
    await expect(pageHeading(page, "Budgets")).toBeVisible()

    // The retired Model pricing path still answers, because bookmarks and links
    // to it outlive the page. `override` named this organization's own rate for
    // a model and still does, so it survives the redirect; `model` named the
    // deployment's rate, whose editor went with the price table, so it is
    // dropped rather than left in a URL nothing reads.
    await gotoRoute(page, "/organization/pricing?override=openai:gpt-4o")
    await expect(pageHeading(page, "Inference providers")).toBeVisible()
    await expect(page).toHaveURL(/override=openai(%3A|:)gpt-4o/)
    await gotoRoute(page, "/organization/pricing?model=openai:gpt-4o")
    await expect(pageHeading(page, "Inference providers")).toBeVisible()
    await expect(page).not.toHaveURL(/model=/)

    // The deployment's own rail, which neither loop above reaches: it is entered
    // from the account menu rather than from a row, and the menu closes on the
    // way, so the rail is what carries you between its pages once you are
    // in. Exact, because the organization rail also carries "Org settings".
    const menu = await openAccountMenu(page)
    await menu.getByRole("link", { name: "Deployment" }).click()
    await expect(pageHeading(page, "Settings")).toBeVisible()
    await nav(page).getByRole("link", { name: "Accounts", exact: true }).click()
    await expect(pageHeading(page, "Accounts")).toBeVisible()
    await nav(page)
      .getByRole("link", { name: "Hosted providers", exact: true })
      .click()
    await expect(pageHeading(page, "Hosted providers")).toBeVisible()
    // The way out names where it goes. Queried on the page rather than inside
    // the Sidebar landmark: the back row is in the scope band above the nav, the
    // same place the workspace switcher sits on the other rail.
    await expect(page.getByRole("link", { name: /^Back to / })).toBeVisible()
  })

  test("create a budget", async ({ page }) => {
    await login(page)
    await openOrganization(page)
    await nav(page).getByRole("link", { name: "Spend & budgets" }).click()
    await page.getByRole("button", { name: "Create your first budget" }).click()
    // Scoped: the heading's trigger and the dialog's submit both say "Create
    // budget", so an unscoped press is ambiguous.
    const dialog = page.getByRole("dialog")
    await dialog.getByLabel("Name (optional)").fill("e2e-budget")
    await dialog.getByLabel("Spending limit (USD)").fill("100")
    await dialog.getByRole("button", { name: "Create budget" }).click()

    // The shared table renders on react-aria, so non-row-header cells are gridcells.
    await expect(page.getByRole("gridcell", { name: "$100.00" })).toBeVisible()
    await expect(page.getByText("e2e-budget")).toBeVisible()
    await expect(page.getByText("No budgets yet")).toBeHidden()
  })

  test("assign the budget to a person", async ({ page }) => {
    // There is no page that creates a spend identity any more: one is minted
    // when a member is added or when a key is issued. Seeded over the API here
    // so this spec stays about the assignment, which is the part that moved onto
    // the budget form when the users page went away.
    //
    // The seed has to happen before `login(page)`, not after it. Loading the app
    // fetches `GET /v1/users`, and `useUsers` caches it with `staleTime: 60_000`
    // (web/src/shared/api/hooks.ts), so a list fetched before alice exists stays
    // fresh through `BudgetsPage` mounting: no refetch, no alice in the combobox,
    // and the option click below waits out the 30s test timeout. Seeding first
    // makes the order deterministic instead of a race with the app's own request.
    const created = await page.request.post(`${API_ROOT}/users`, {
      headers: { "Otari-Key": MASTER_KEY },
      data: { user_id: "alice@example.com" },
    })
    expect(created.ok() || created.status() === 409).toBeTruthy()

    await login(page)
    await openOrganization(page)
    await nav(page).getByRole("link", { name: "Spend & budgets" }).click()
    const budgetRow = page.getByRole("row", { name: /e2e-budget/ })
    await pressRowAction(budgetRow, "Edit")
    // The field's visible label is its accessible name now: the picker used to
    // carry a hidden "Add a person" beside a heading that labeled nothing, so
    // one control had two names.
    const editDialog = page.getByRole("dialog")
    const owners = editDialog.getByRole("combobox", {
      name: "Assign to people (optional)",
    })
    await owners.fill("alice@example.com")
    await page.getByRole("option", { name: /alice@example\.com/ }).click()
    await dismissComboBoxInDialog(owners)
    await editDialog.getByRole("button", { name: "Save" }).click()

    // The budget now reports one holder in its People column, which is the
    // assignment landing.
    // Exact: the row also holds "$100.00" and a "Select row ..." checkbox, both
    // of which substring-match a bare "1".
    await expect(
      page
        .getByRole("row", { name: /e2e-budget/ })
        .getByRole("gridcell", { name: "1", exact: true }),
    ).toBeVisible()
  })

  test("create an API key owned by a chosen user", async ({ page }) => {
    await login(page)
    await nav(page).getByRole("link", { name: "API keys" }).click()
    // A bootstrap key already exists, so use the header action, not the empty
    // list's.
    await page
      .getByRole("main")
      .getByRole("button", { name: "Create API key" })
      .click()
    // Scoped from here on, because "Create API key" is now on screen twice:
    // the heading's trigger stays visible while the dialog is open, and the
    // labels rule makes the dialog's submit say the same words.
    const dialog = page.getByRole("dialog")
    await dialog.getByLabel("Name").fill("ci-bot")
    // The owner is under More options. Reuse the user created earlier; type it
    // and close the combobox popover so it does not aria-hide the submit button.
    await dialog.getByRole("button", { name: "More options" }).click()
    const ownerBox = dialog.getByPlaceholder("Pick a user, or type a new id…")
    await ownerBox.fill("alice@example.com")
    await dismissComboBoxInDialog(ownerBox)
    await dialog.getByRole("button", { name: "Create API key" }).click()

    // The one-time reveal appears; acknowledge it.
    await page.getByRole("button", { name: /saved this key/i }).click()

    const row = page.getByRole("row", { name: /ci-bot/ })
    await expect(row).toBeVisible()
    // The key is owned by the named user, not an anonymous virtual one.
    await expect(row.getByText("alice@example.com")).toBeVisible()
  })

  test("create a routing policy", async ({ page }) => {
    await login(page)
    await openTab(page, "Rules", "Routing")
    // `.first()` is the heading's action. An empty routing list also offers
    // the same words from its empty state, and a press has to name which.
    await page.getByRole("button", { name: "Create policy" }).first().click()
    // Scoped from here: the dialog's submit says "Create policy" too.
    const dialog = page.getByRole("dialog")
    // The dialog asks what the policy is for first; a fallback is the default.
    await expect(dialog.getByRole("radio", { name: "Fallback" })).toBeChecked()
    await dialog.getByRole("button", { name: "Continue" }).click()
    // Role-scoped for the same reason as the user form: policy rows carry a
    // "Copy policy name" control.
    await dialog.getByRole("textbox", { name: /^Name/ }).fill("fast")
    // "Main model" is a model combobox (allows custom values); type the selector,
    // then put the popover away so it does not aria-hide the submit button.
    const serves = dialog.getByRole("combobox", { name: "Main model" })
    await serves.fill("openai:gpt-4o")
    await dismissComboBoxInDialog(serves)
    // A fallback needs a backup before it can be created.
    const backup = dialog.getByRole("combobox", { name: "Backup 1" })
    await backup.fill("anthropic:claude-3-5-haiku-latest")
    await dismissComboBoxInDialog(backup)
    await dialog.getByRole("button", { name: "Create policy" }).click()

    // The policy name is the table's row-header cell (react-aria rowheader).
    await expect(page.getByRole("rowheader", { name: "fast" })).toBeVisible()
  })

  test("grows a policy a fallback chain", async ({ page }) => {
    await login(page)
    await openTab(page, "Rules", "Routing")
    // `.first()` is the heading's action. An empty routing list also offers
    // the same words from its empty state, and a press has to name which.
    await page.getByRole("button", { name: "Create policy" }).first().click()
    // Scoped from here: the dialog's submit says "Create policy" too.
    const dialog = page.getByRole("dialog")
    await dialog.getByRole("button", { name: "Continue" }).click()
    await dialog.getByRole("textbox", { name: /^Name/ }).fill("chained")
    const serves = dialog.getByRole("combobox", { name: "Main model" })
    await serves.fill("openai:gpt-4o")
    await dismissComboBoxInDialog(serves)

    // A fallback opens on its first backup row, which it cannot be created
    // without.
    await expect(
      dialog.getByRole("button", { name: "Create policy" }),
    ).toBeDisabled()
    const fallback = page.getByRole("combobox", { name: "Backup 1" })
    await fallback.fill("anthropic:claude-3-5-haiku-latest")
    await dismissComboBoxInDialog(fallback)
    await dialog.getByRole("button", { name: "Create policy" }).click()

    // Scoped to the row this test created: a backup chip anywhere on the page
    // would also be satisfied by another policy's chain, so a `chained` saved
    // without its backup would still pass.
    const chained = page
      .getByRole("row")
      .filter({ has: page.getByRole("rowheader", { name: "chained" }) })
    await expect(chained).toBeVisible()
    await expect(
      chained.getByRole("list", { name: "Backups" }).getByRole("listitem"),
    ).toHaveText(["claude-3-5-haiku-latest · Anthropic"])
  })

  test("renames a policy in place", async ({ page }) => {
    // The rename target has to be free for the 409 not to fire. serve.sh wipes the
    // database, so it is on a first run; a re-run against a warm one still carries
    // the `renamed` this test left behind, and would fail on its own leftovers.
    const dropped = await page.request.delete(
      `${API_ROOT}/routing/policies/renamed`,
      {
        headers: { Authorization: `Bearer ${MASTER_KEY}` },
      },
    )
    expect([204, 404]).toContain(dropped.status())

    await login(page)
    await openTab(page, "Rules", "Routing")

    const chained = page
      .getByRole("row")
      .filter({ has: page.getByRole("rowheader", { name: "chained" }) })
    await pressRowAction(chained, "Edit")
    // Edit opens the policy's own page, where the name is a section of its own.
    await expect(pageHeading(page, "chained")).toBeVisible()
    await page.getByRole("textbox", { name: /^Name/ }).fill("renamed")
    await page.getByRole("button", { name: "Rename" }).click()
    // The page follows the policy to its new address.
    await expect(pageHeading(page, "renamed")).toBeVisible()
    await page.getByRole("link", { name: "← Routing" }).click()

    // A rename moves the row rather than copying it, so the old name has to be
    // gone: two rows would mean callers could still reach the policy either way.
    const renamed = page
      .getByRole("row")
      .filter({ has: page.getByRole("rowheader", { name: "renamed" }) })
    await expect(renamed).toBeVisible()
    await expect(
      renamed.getByRole("list", { name: "Backups" }).getByRole("listitem"),
    ).toHaveCount(1)
    await expect(page.getByRole("rowheader", { name: "chained" })).toBeHidden()
  })
})
