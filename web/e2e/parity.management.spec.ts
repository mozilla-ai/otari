import { expect, type Locator, type Page, test } from "@playwright/test"

import {
  dismissComboBox,
  dismissComboBoxInDialog,
  gotoRoute,
  login,
  nav,
  openNested,
  openOrganization,
  pageHeading,
  tableRows,
} from "./helpers"
import { PARITY } from "./parity-data"

// Each flow creates the object it acts on and removes it again, so the file can
// run against a warm database and leaves the fixture the other specs read
// untouched. They still run in order: the suite is single-worker against one
// gateway either way.
test.describe.configure({ mode: "serial" })

const PROVIDER = "parity-provider"
const KEY_NAME = "parity-key"
const BUDGET = "parity-budget"
const POLICY = "parity-chain"

// Nothing is listening on the discard port, so a provider pointed at it fails to
// connect immediately and deterministically, with no network egress and no
// dependency on a name that has to resolve.
const UNREACHABLE = "http://127.0.0.1:9/v1"

async function openPage(
  page: Page,
  link: string,
  heading: string,
): Promise<void> {
  await nav(page).getByRole("link", { name: link }).click()
  await expect(
    page.getByRole("heading", { name: heading, exact: true }),
  ).toBeVisible()
}

function row(page: Page, ariaLabel: string, name: string | RegExp): Locator {
  return tableRows(page, ariaLabel).filter({
    has: page.getByRole("rowheader", { name }),
  })
}

// A routing form's model pickers take any selector, so the value is typed rather
// than chosen. Their popover has to be put away afterwards or it aria-hides the
// controls below it, including the form's own submit.
//
// The page-level dismissal, deliberately, even though this form is a dialog now.
// The model box does not reopen on focus, so the `blur()` puts it away and the
// wait passes. The people picker is the one that does: it is `menuTrigger="focus"`,
// so blurring it inside a modal hands focus back and the popover returns, which
// is what `dismissComboBoxInDialog` exists for. Move this to that helper if the
// model box ever opens on focus too.
async function fillModelBox(
  page: Page,
  name: RegExp,
  value: string,
): Promise<void> {
  const box = page.getByRole("combobox", { name })
  await box.fill(value)
  await dismissComboBox(box)
}

test.describe("standalone provider setup", () => {
  test("adds a provider, reports what testing it found, edits and removes it", async ({
    page,
  }) => {
    await login(page)
    await openPage(page, "Providers", "Providers")

    await page.getByRole("button", { name: "Add provider" }).click()
    await page.getByRole("button", { name: "Custom endpoint" }).click()
    await page.getByLabel("Name").fill(PROVIDER)
    await page.getByLabel("API base").fill(UNREACHABLE)
    await page.getByRole("button", { name: "Add provider" }).click()

    const provider = row(page, "Providers", PROVIDER)
    await expect(provider).toBeVisible()
    // A stored provider is one the dashboard owns, as against one read out of
    // config.yml, which is what decides whether it can be edited here at all.
    // Uppercase in the source rather than through a `text-transform`, so this is
    // the text content and the match is case-sensitive.
    await expect(provider).toContainText("STORED")

    // Testing has to come back with a reported outcome, not just stop spinning:
    // an assertion that it merely does not say "Connected." would also pass if the
    // check never ran at all. Nothing is listening, so the outcome is a failure,
    // and the alternation covers the shapes TestOutcome can render (the sanitized
    // provider error, its own fallback, and the no-model-listing branch).
    await provider.getByRole("button", { name: "Test" }).click()
    await expect(
      provider.getByText(
        /Connection error\.|Connection failed\.|Could not list models/,
      ),
    ).toBeVisible({ timeout: 30_000 })
    await expect(provider).not.toContainText("Connected.")

    await provider.getByRole("button", { name: "Edit" }).click()
    await page.getByLabel("API base").fill("http://127.0.0.1:9/v2")
    await page.getByRole("button", { name: "Save changes" }).click()
    // Wait on Cancel, not on the button that was just pressed: that one reads
    // "Saving…" while the request is in flight, so asserting "Save changes" is
    // hidden passes the instant it is pressed, and the reload below then aborts
    // the PATCH and loses the edit. Cancel is present for as long as the form is,
    // and the form closes only in the mutation's onSuccess.
    await expect(page.getByRole("button", { name: "Cancel" })).toBeHidden()

    // Reopening is what proves the edit was persisted rather than only echoed
    // back into a form that never closed. Reload first: the form seeds its fields
    // from the provider on mount only, so reopening it while the invalidated
    // provider query is still refetching would seed the stale value and then never
    // correct itself, which no amount of retrying the assertion can recover.
    await page.reload()
    await row(page, "Providers", PROVIDER)
      .getByRole("button", { name: "Edit" })
      .click()
    await expect(page.getByLabel("API base")).toHaveValue(
      "http://127.0.0.1:9/v2",
    )
    await page.getByRole("button", { name: "Cancel" }).click()

    // The confirmation is a modal, so it names the provider rather than relying
    // on the row behind the backdrop to say which one this is about.
    await row(page, "Providers", PROVIDER)
      .getByRole("button", { name: "Delete" })
      .click()
    const confirmProvider = page.getByRole("alertdialog")
    await expect(confirmProvider).toContainText(PROVIDER)
    await confirmProvider
      .getByRole("button", { name: "Delete provider" })
      .click()
    await expect(row(page, "Providers", PROVIDER)).toHaveCount(0)
  })
})

test.describe("api keys", () => {
  test("issues a key, and will not delete one until it is disabled", async ({
    page,
  }) => {
    await login(page)
    await openPage(page, "API keys", "API keys")

    await page.getByRole("button", { name: "Create key" }).click()
    // Scoped from here on, because "Create key" is now on screen twice: the
    // heading's trigger stays visible while the dialog is open, and the labels
    // rule makes the dialog's submit say the same words.
    const dialog = page.getByRole("dialog")
    await dialog.getByLabel("Name").fill(KEY_NAME)
    // Owner is required: this is what keeps the dashboard from minting the
    // anonymous virtual users an omitted id would.
    const ownerBox = dialog.getByPlaceholder("Pick a user, or type a new id…")
    await ownerBox.fill(PARITY.users.heavy)
    await dismissComboBoxInDialog(ownerBox)
    await dialog.getByRole("button", { name: "Create key" }).click()

    // The secret is shown exactly once, behind an acknowledgement that is the
    // only way out: the dialog is `isDismissable={false}`, so Escape and the
    // close control are gone. The reveal keeps its own `role="alert"` inside
    // the dialog body, which is what the first two assertions read.
    const reveal = page.getByRole("alert", { name: /API key created/ })
    await expect(reveal).toBeVisible()
    await expect(reveal).toContainText("shown only once")
    // Scoped to the dialog rather than to the reveal: the acknowledgement is
    // the dialog's own submit, in the footer, so it sits outside the alert
    // region the announcement covers.
    await dialog.getByRole("button", { name: /saved this key/i }).click()

    const key = row(page, "API keys", KEY_NAME)
    await expect(key).toContainText("Active")
    await expect(key).toContainText(PARITY.users.heavy)
    // Permanent delete is withheld while a key is live, so a caller in production
    // cannot be broken (and its audit trail erased) in a single click.
    await expect(key.getByRole("button", { name: "Delete" })).toHaveCount(0)

    await key.getByRole("button", { name: "Disable" }).click()
    await expect(key).toContainText("Disabled")
    await expect(key.getByRole("button", { name: "Enable" })).toBeVisible()

    await key.getByRole("button", { name: "Delete" }).click()
    // Scoped to the dialog rather than to `key`: the confirmation is a modal
    // now, outside the table entirely.
    const confirmKey = page.getByRole("alertdialog")
    await expect(confirmKey).toContainText(KEY_NAME)
    await confirmKey.getByRole("button", { name: "Delete permanently" }).click()
    await expect(row(page, "API keys", KEY_NAME)).toHaveCount(0)
  })
})

test.describe("budgets", () => {
  test("creates a budget against a user, edits its limit, and removes it", async ({
    page,
  }) => {
    await login(page)
    await openOrganization(page)
    await openPage(page, "Spend & budgets", "Budgets")

    await page.getByRole("button", { name: "Create budget" }).click()
    // Scoped: the heading's trigger and the dialog's submit both say "Create
    // budget", so an unscoped press is ambiguous.
    const dialog = page.getByRole("dialog")
    await dialog.getByLabel("Name (optional)").fill(BUDGET)
    await dialog.getByLabel("Spending limit (USD)").fill("25")
    // Assigning at creation is the path that makes a budget enforceable; a budget
    // with no users caps nothing.
    const owner = dialog.getByRole("combobox", {
      name: "Assign to people (optional)",
    })
    await owner.fill(PARITY.users.heavy)
    // Plain string, not a RegExp built from the id: an address is full of regex
    // metacharacters, so `.` would match any character and the pattern could pick
    // a neighbouring option. Playwright matches an accessible name by substring
    // here, which is what an aliased user's "id (alias)" label needs anyway.
    await page.getByRole("option", { name: PARITY.users.heavy }).click()
    await dismissComboBoxInDialog(owner)
    // The picked user becomes a removable chip, which is the form's own record of
    // who this budget will cap before it is submitted.
    await expect(
      dialog.getByRole("button", { name: `Remove ${PARITY.users.heavy}` }),
    ).toBeVisible()
    await dialog.getByRole("button", { name: "Create budget" }).click()

    const budget = row(page, "Budgets", BUDGET)
    await expect(budget).toContainText("$25.00")
    // One assigned user, which is also what turns the Usage cell from "No users
    // assigned" into a real spend-against-allocation reading.
    await expect(budget).not.toContainText("No users assigned")

    await budget.getByRole("button", { name: "Edit" }).click()
    const editDialog = page.getByRole("dialog")
    await editDialog.getByLabel("Spending limit (USD)").fill("50")
    await editDialog.getByRole("button", { name: "Save" }).click()
    await expect(row(page, "Budgets", BUDGET)).toContainText("$50.00")

    // Reset history is a per-budget panel, not a page: it opens under the table
    // and names the budget it belongs to.
    await row(page, "Budgets", BUDGET)
      .getByRole("button", { name: "History" })
      .click()
    await expect(page.getByText(`Reset history — ${BUDGET}`)).toBeVisible()
    await page.getByRole("button", { name: "Close" }).click()

    // The confirmation names the budget it is about to drop rather than asking
    // in the abstract, which is half of why it is a modal: the row it names is
    // behind the backdrop.
    await row(page, "Budgets", BUDGET)
      .getByRole("button", { name: "Delete", exact: true })
      .click()
    const confirmBudget = page.getByRole("alertdialog")
    await expect(confirmBudget).toContainText(BUDGET)
    await confirmBudget
      .getByRole("button", { name: "Delete permanently" })
      .click()
    await expect(row(page, "Budgets", BUDGET)).toHaveCount(0)
  })
})

test.describe("fallback routing", () => {
  test("grows and shrinks a policy's failure chain", async ({ page }) => {
    await login(page)
    await openNested(page, "Routing", "Policies")
    await expect(pageHeading(page, "Routing")).toBeVisible()

    // `.first()` is the heading's action. An empty routing list also offers
    // the same words from its empty state, and a press has to name which.
    await page.getByRole("button", { name: "Create policy" }).first().click()
    // Scoped from here: the dialog's submit says "Create policy" too.
    const dialog = page.getByRole("dialog")
    await dialog.getByRole("textbox", { name: /Policy name/ }).fill(POLICY)
    await fillModelBox(page, /Serves/, "openai:gpt-4o")

    // The chain is summoned rather than presented, so naming one model stays a
    // short task; a second candidate is then added to the chain that exists.
    await dialog.getByRole("button", { name: /Add a fallback chain/ }).click()
    await fillModelBox(page, /Fallback 1/, "anthropic:claude-3-5-haiku-latest")
    await page.getByRole("button", { name: "+ Another fallback" }).click()
    await fillModelBox(page, /Fallback 2/, "groq:llama-3.3-70b-versatile")
    await dialog.getByRole("button", { name: "Create policy" }).click()

    // The table summarises the chain by its length, so the count is the assertion
    // that both fallbacks were saved and not just the first.
    const policy = row(page, "Routing policies", POLICY)
    await expect(policy).toContainText(/\+2 on failure/)

    // Shrinking has to be possible too: a chain that could only grow would strand
    // an operator who added a candidate by mistake.
    await policy.getByRole("button", { name: "Edit" }).click()
    const fallbackTwo = page.getByRole("combobox", { name: /Fallback 2/ })
    await expect(fallbackTwo).toBeVisible()
    // Scoped to the row holding Fallback 2 rather than taken by index off the
    // page: the form grows a Remove control per condition, per candidate in a
    // pool and per guardrail, so an index would start removing the wrong thing
    // the moment this policy grew one of those.
    // Innermost element holding both the box and its Remove: the combobox sits in
    // a wrapper of its own, so filtering on the box alone lands inside that
    // wrapper, which has no button in it.
    await page
      .locator("div")
      .filter({ has: fallbackTwo })
      .filter({ has: page.getByRole("button", { name: "Remove" }) })
      .last()
      .getByRole("button", { name: "Remove" })
      .click()
    await expect(
      page.getByRole("combobox", { name: /Fallback 2/ }),
    ).toHaveCount(0)
    await page.getByRole("button", { name: "Save" }).click()
    await expect(row(page, "Routing policies", POLICY)).toContainText(
      /\+1 on failure/,
    )

    await row(page, "Routing policies", POLICY)
      .getByRole("button", { name: "Delete" })
      .click()
    const confirmPolicy = page.getByRole("alertdialog")
    await expect(confirmPolicy).toContainText(POLICY)
    await confirmPolicy.getByRole("button", { name: "Delete policy" }).click()
    await expect(row(page, "Routing policies", POLICY)).toHaveCount(0)
  })
})

// Two URL contracts the route table owes anyone holding an old link. Neither is
// reachable from the UI, so nothing else in the suite would notice them
// breaking, and both are one line in a route file: easy to drop in a refactor.
test.describe("legacy routes", () => {
  test("keeps the retired aliases path and unknown paths pointing somewhere real", async ({
    page,
  }) => {
    await login(page)

    // Aliases were folded into Routing, which lists them as the one-target
    // policies they are; the old path is a bookmark that still has to land.
    await gotoRoute(page, "/aliases")
    await expect(page).toHaveURL(/#\/routing$/)
    await expect(pageHeading(page, "Routing")).toBeVisible()

    // An unrecognised path is a stale link from a route that has since moved,
    // so it lands on the overview rather than a dead end.
    await gotoRoute(page, "/no-such-page")
    await expect(page).toHaveURL(/#\/$/)
  })
})
