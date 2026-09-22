import { expect, type Locator, type Page, test } from "@playwright/test"

import { API_ROOT } from "@/shared/api/client"
import { login, nav, openOrganization, pickOption, tableRows } from "./helpers"

// The tenancy pages against a real gateway: the organization a first boot
// provisions, its roster, and the workspaces under it. Each flow creates what it
// acts on and removes it again, so the file runs against a warm database and
// leaves the default organization and its default workspace as it found them.
test.describe.configure({ mode: "serial" })

const WORKSPACE = "parity-workspace"
const RENAMED_WORKSPACE = "parity-workspace-renamed"
const GUARDRAIL = "parity-lakera"
const PROFILE = "parity-injection"
// What provisioning names the bootstrap identity (OPERATOR_FULL_NAME in
// provisioning_service.py). It has no email address, which is the point: a
// standalone operator is a label, not a sign-in.
const OPERATOR = "Operator"

async function openPage(
  page: Page,
  link: string,
  heading: string,
): Promise<void> {
  await nav(page).getByRole("link", { name: link, exact: true }).click()
  await expect(
    page.getByRole("heading", { name: heading, exact: true }),
  ).toBeVisible()
}

// The value beside a term in the organization's detail list, the way
// `tileValue` reads the value beside a tile's label.
function organizationDetail(page: Page, term: string): Locator {
  return page
    .getByText(term, { exact: true })
    .locator("xpath=following-sibling::dd[1]")
}

// Renaming is a dialog away, and the dialog shows the name being replaced
// before it takes the one replacing it.
async function rename(page: Page, to: string): Promise<void> {
  await page.getByRole("button", { name: "Change organization name" }).click()
  // The title names the object, and the trigger and submit share the action.
  const dialog = page.getByRole("dialog", { name: "Organization name" })
  await expect(dialog.getByText("Current name")).toBeVisible()
  await dialog.getByLabel("New name").fill(to)
  await dialog.getByRole("button", { name: "Change organization name" }).click()
  await expect(dialog).toBeHidden()
}

function memberRow(page: Page, name: string | RegExp): Locator {
  return tableRows(page, "Organization members").filter({
    has: page.getByRole("rowheader", { name }),
  })
}

function guardrailRow(page: Page, table: string, name: string): Locator {
  return tableRows(page, table).filter({
    has: page.getByRole("rowheader", { name }),
  })
}

function workspaceRow(page: Page, name: string): Locator {
  return tableRows(page, "Workspaces").filter({
    has: page.getByRole("rowheader", { name }),
  })
}

test.describe("standalone tenancy", () => {
  test("shows the organization a first boot provisioned, and renames it back", async ({
    page,
  }) => {
    await login(page)
    await openOrganization(page)
    await openPage(page, "Org settings", "Organization")

    // The master key names no user, so the first authenticated request
    // provisions this: one organization, one owner identity, one workspace.
    const name = organizationDetail(page, "Organization name")
    const original = ((await name.textContent()) ?? "").trim()
    expect(original).not.toBe("")

    // The page reads the name rather than offering it in an open box, so
    // nothing here can be renamed by a stray keystroke on arrival.
    await expect(page.getByRole("main").getByRole("textbox")).toHaveCount(0)

    await rename(page, "Parity Organization")
    await expect(name).toHaveText("Parity Organization")

    // The slug is set at creation and deliberately does not follow a rename,
    // which is what makes it safe to key anything off.
    await rename(page, original)
    await expect(name).toHaveText(original)
  })

  test("lists the operator as an undemotable owner", async ({ page }) => {
    await login(page)
    await openOrganization(page)
    await openPage(page, "Members & roles", "Members")

    // Scoped to the operator's own row rather than to a global count: this
    // gateway is shared with the flow below, which adds a member and leaves a
    // suspended one behind on a re-run.
    const operator = memberRow(page, OPERATOR)
    const role = operator.getByRole("button", { name: /Role for / })
    await expect(role).toHaveText(/Owner/)
    // The last active owner cannot be demoted or removed: doing so would leave
    // the organization with nobody able to manage or delete it.
    await expect(role).toBeDisabled()
    await expect(
      operator.getByRole("button", { name: "Remove" }),
    ).toBeDisabled()
    // Status is shown, not set: suspending is what Remove does, behind a
    // confirmation, and a suspended membership leaves the roster entirely.
    await expect(
      operator.getByRole("button", { name: /Status for / }),
    ).toHaveCount(0)
    await expect(operator.getByText("Active")).toBeVisible()
  })

  test("invites a member, hands back a link that lets them in, gives them a role, and removes them", async ({
    browser,
    page,
  }) => {
    // Fresh per run: accepting sets the address's password, and a second run
    // against the same gateway would otherwise find an account that can
    // already sign in and get no password form.
    const email = `parity-member-${Date.now()}@example.com`
    await login(page)
    await openOrganization(page)
    await openPage(page, "Members & roles", "Members")

    await page.getByRole("button", { name: "Invite member" }).click()
    const inviteDialog = page.getByRole("dialog", { name: "Invitation" })
    await inviteDialog.getByLabel("Email address").fill(email)
    await pickOption(page, "Role", "Member", inviteDialog)
    // Scoped: the heading's trigger and the dialog's submit share the label.
    await inviteDialog.getByRole("button", { name: "Invite member" }).click()

    // This environment sends no mail, so the link comes back to the operator,
    // absolute even though the gateway knows no public address of its own.
    await expect(
      inviteDialog.getByText(/Otari did not send the email/),
    ).toBeVisible()
    const link = await inviteDialog
      .getByText(/^http:\/\/.+#\/accept-invitation\?token=/)
      .textContent()
    expect(link).toBeTruthy()
    await inviteDialog.getByRole("button", { name: "Done" }).click()
    await expect(memberRow(page, email).getByText(/^invited$/i)).toBeVisible()

    // The invitee holds no session: a separate context, as a forwarded link
    // would arrive in someone else's browser.
    const invitee = await browser.newContext()
    const inviteePage = await invitee.newPage()
    await inviteePage.goto(link ?? "")
    await inviteePage
      .getByLabel("Password", { exact: true })
      .fill("parity-password")
    await inviteePage.getByLabel("Confirm password").fill("parity-password")
    await inviteePage
      .getByRole("button", { name: "Accept and set password" })
      .click()
    await expect(inviteePage.getByText(/Your password is set/)).toBeVisible()
    const origin = new URL(link ?? "").origin
    const signedIn = await invitee.request.post(
      `${origin}${API_ROOT}/auth/session`,
      {
        data: { email, password: "parity-password" },
      },
    )
    expect(signedIn.status()).toBe(200)
    await invitee.close()

    await page.reload()
    const member = memberRow(page, email)
    await expect(member.getByText(/^active$/i)).toBeVisible()
    const role = member.getByRole("button", { name: /Role for / })
    await expect(role).toHaveText(/Member/)

    await pickOption(page, /Role for /, "Admin", member)
    await expect(
      memberRow(page, email).getByRole("button", { name: /Role for / }),
    ).toHaveText(/Admin/)

    // Removal suspends rather than deletes, and a suspended membership is not
    // listable, so the row leaves the roster while the attribution behind it
    // survives.
    await memberRow(page, email).getByRole("button", { name: "Remove" }).click()
    await page.getByRole("button", { name: "Remove member" }).click()
    await expect(memberRow(page, email)).toHaveCount(0)

    // Re-inviting the same address revives that membership rather than
    // starting a second one.
    await page.getByRole("button", { name: "Invite member" }).click()
    const reinviteDialog = page.getByRole("dialog", { name: "Invitation" })
    await reinviteDialog.getByLabel("Email address").fill(email)
    await reinviteDialog.getByRole("button", { name: "Invite member" }).click()
    await reinviteDialog.getByRole("button", { name: "Done" }).click()
    await expect(memberRow(page, email)).toHaveCount(1)

    // Leave the roster as this spec found it. The invitee's identity keeps its
    // password, so later specs see a sign-in screen offering both credentials,
    // which `login` handles.
    await memberRow(page, email).getByRole("button", { name: "Revoke" }).click()
    await page.getByRole("button", { name: "Revoke invitation" }).click()
    await expect(memberRow(page, email)).toHaveCount(0)
  })

  test("creates a workspace, renames it, and removes it", async ({ page }) => {
    await login(page)
    await openOrganization(page)
    await openPage(page, "Workspaces", "Workspaces")

    await page.getByRole("button", { name: "Create workspace" }).click()
    // Scoped: the heading's trigger and the dialog's submit both say "Create
    // workspace", so an unscoped press is ambiguous.
    const dialog = page.getByRole("dialog", { name: "New workspace" })
    await dialog.getByLabel("Name").fill(WORKSPACE)
    await dialog
      .getByLabel("Description (optional)")
      .fill("Created by the parity suite")
    await dialog.getByRole("button", { name: "Create workspace" }).click()

    const created = workspaceRow(page, WORKSPACE)
    await expect(created).toBeVisible()
    await expect(created).toContainText("Created by the parity suite")

    await created.getByRole("button", { name: "Edit" }).click()
    const editDialog = page.getByRole("dialog", { name: "Edit workspace" })
    await editDialog.getByLabel("Name").fill(RENAMED_WORKSPACE)
    await editDialog.getByRole("button", { name: "Save" }).click()
    await expect(editDialog).toBeHidden()

    const renamed = workspaceRow(page, RENAMED_WORKSPACE)
    await expect(renamed).toBeVisible()

    await renamed.getByRole("button", { name: "Delete" }).click()
    await page.getByRole("button", { name: "Delete workspace" }).click()
    await expect(workspaceRow(page, RENAMED_WORKSPACE)).toHaveCount(0)
  })

  test("reads the selected workspace's roster on its own page", async ({
    page,
  }) => {
    // The roster used to be reachable from a row on Workspaces as well as here,
    // the same component on two rails. That copy is gone, so this is the only
    // place it renders and the only place it can be covered.
    await login(page)
    await nav(page).getByRole("link", { name: "Members", exact: true }).click()

    // A workspace's members are a subset of the organization's, and a standalone
    // deployment has exactly one identity, which owns every workspace it made.
    await expect(page.getByText(/Members of /)).toBeVisible()
    // Said inside the dialog now rather than by a form under the roster. The
    // trigger stays either way, so this is the only place that sentence can be.
    await page.getByRole("button", { name: "Add member" }).click()
    const dialog = page.getByRole("dialog", { name: "New workspace member" })
    await expect(dialog.getByText(/already in this workspace/)).toBeVisible()
    await dialog.getByRole("button", { name: "Cancel" }).click()
  })

  test("defines a guardrail, mandates it, edits the mandate, and removes both", async ({
    page,
  }) => {
    await login(page)
    await openOrganization(page)
    await openPage(page, "Guardrails", "Guardrails")

    await page.getByRole("button", { name: "Set up guardrail" }).click()
    const setUp = page.getByRole("dialog", { name: "New guardrail" })
    await pickOption(
      page,
      "What do you want checked?",
      "Prompt injection",
      setUp,
    )
    await pickOption(page, "Which guardrail?", "Lakera Guard · Lakera", setUp)
    await setUp.getByLabel("Name").fill(GUARDRAIL)
    // No vendor is called here: whether the guardrail builds is not what this
    // flow checks, and CI has no Lakera account.
    await setUp.getByLabel("Api key").fill("parity-not-a-real-key")
    await setUp.getByRole("button", { name: "Set up guardrail" }).click()
    await expect(setUp).toBeHidden()
    const defined = guardrailRow(
      page,
      "Guardrails you have configured",
      GUARDRAIL,
    )
    await expect(defined).toContainText("Api key set")

    // Mandated nowhere: no workspace is chosen, so a guardrail that did not
    // build cannot refuse another spec's requests while it exists.
    await page.getByRole("button", { name: "Mandate a guardrail" }).click()
    const mandate = page.getByRole("dialog", { name: "Mandated guardrail" })
    await pickOption(page, "Guardrail", GUARDRAIL, mandate)
    await mandate.getByLabel("Profile a caller sends").fill(PROFILE)
    await mandate.getByRole("button", { name: "Mandate a guardrail" }).click()
    await expect(mandate).toBeHidden()
    const mandated = guardrailRow(page, "Where they run", PROFILE)
    await expect(mandated).toContainText(GUARDRAIL)

    // An edit of the mandate touches no secret, so the key stays set without
    // being typed again.
    await mandated.getByRole("button", { name: `Edit ${PROFILE}` }).click()
    const edit = page.getByRole("dialog", { name: "Mandated guardrail" })
    await pickOption(page, "Mode", "Block", edit)
    await edit.getByRole("button", { name: "Save mandate" }).click()
    await expect(edit).toBeHidden()
    await expect(mandated).toContainText("Block")
    await expect(defined).toContainText("Api key set")

    // Leave the organization as this spec found it: the mandate first, since
    // the definition cannot go while one still names it.
    await mandated.getByRole("button", { name: `Remove ${PROFILE}` }).click()
    await page.getByRole("button", { name: "Remove permanently" }).click()
    await expect(guardrailRow(page, "Where they run", PROFILE)).toHaveCount(0)
    await defined.getByRole("button", { name: `Remove ${GUARDRAIL}` }).click()
    await page.getByRole("button", { name: "Remove permanently" }).click()
    await expect(
      guardrailRow(page, "Guardrails you have configured", GUARDRAIL),
    ).toHaveCount(0)
  })

  test("leaves creating and switching to the scope switcher, and offers no delete", async ({
    page,
  }) => {
    await login(page)
    await openOrganization(page)
    await openPage(page, "Org settings", "Organization")

    // Creating an organization and moving between them are the scope switcher's,
    // which sits on the workspace rail and is replaced by the way back out on
    // this one, so neither control is here. Deleting one has no endpoint at all.
    await expect(
      page.getByRole("button", { name: /Create organization/ }),
    ).toHaveCount(0)
    await expect(page.getByRole("button", { name: "Switch" })).toHaveCount(0)
    await expect(
      page.getByRole("button", { name: /Delete organization/ }),
    ).toHaveCount(0)
  })
})
