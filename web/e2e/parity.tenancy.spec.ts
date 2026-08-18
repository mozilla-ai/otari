import { expect, type Locator, type Page, test } from "@playwright/test"

import { login, nav, tableRows } from "./helpers"

// The tenancy pages against a real gateway: the organization a first boot
// provisions, its roster, and the workspaces under it. Each flow creates what it
// acts on and removes it again, so the file runs against a warm database and
// leaves the default organization and its default workspace as it found them.
test.describe.configure({ mode: "serial" })

const WORKSPACE = "parity-workspace"
const RENAMED_WORKSPACE = "parity-workspace-renamed"

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
    await openPage(page, "General", "Organization")

    // The master key names no user, so the first authenticated request
    // provisions this: one organization, one owner identity, one workspace.
    const name = page.getByLabel("Organization name")
    const original = await name.inputValue()
    expect(original).not.toBe("")

    await name.fill("Parity Organization")
    await page.getByRole("button", { name: "Save name" }).click()
    await expect(name).toHaveValue("Parity Organization")

    // The slug is set at creation and deliberately does not follow a rename,
    // which is what makes it safe to key anything off.
    await name.fill(original)
    await page.getByRole("button", { name: "Save name" }).click()
    await expect(name).toHaveValue(original)
  })

  test("lists the operator as the organization's only, undemotable owner", async ({
    page,
  }) => {
    await login(page)
    await openPage(page, "Members", "Members")

    const roles = page.getByLabel(/^Role for /)
    await expect(roles).toHaveCount(1)
    await expect(roles.first()).toHaveValue("owner")
    // The last active owner cannot be demoted or removed: doing so would leave
    // the organization with nobody able to manage or delete it.
    await expect(roles.first()).toBeDisabled()
    await expect(page.getByRole("button", { name: "Remove" })).toBeDisabled()
  })

  test("creates a workspace, renames it, reads its roster, and removes it", async ({
    page,
  }) => {
    await login(page)
    await openPage(page, "Workspaces", "Workspaces")

    await page.getByRole("button", { name: "Create workspace" }).click()
    await page.getByLabel("Name").fill(WORKSPACE)
    await page
      .getByLabel("Description (optional)")
      .fill("Created by the parity suite")
    await page.getByRole("button", { name: "Create workspace" }).click()

    const created = workspaceRow(page, WORKSPACE)
    await expect(created).toBeVisible()
    await expect(created).toContainText("Created by the parity suite")

    await created.getByRole("button", { name: "Edit" }).click()
    await page.getByLabel("Name").fill(RENAMED_WORKSPACE)
    await page.getByRole("button", { name: "Save changes" }).click()

    const renamed = workspaceRow(page, RENAMED_WORKSPACE)
    await expect(renamed).toBeVisible()

    // A workspace's members are a subset of the organization's, and a
    // standalone deployment has exactly one identity, which created this
    // workspace and therefore already owns it.
    await renamed.getByRole("button", { name: "Members" }).click()
    await expect(
      page.getByText(`Members of ${RENAMED_WORKSPACE}`),
    ).toBeVisible()
    await expect(page.getByText(/already in this workspace/)).toBeVisible()

    await renamed.getByRole("button", { name: "Delete" }).click()
    await page.getByRole("button", { name: "Delete workspace" }).click()
    await expect(workspaceRow(page, RENAMED_WORKSPACE)).toHaveCount(0)
  })

  test("refuses to delete the only organization the operator belongs to", async ({
    page,
  }) => {
    await login(page)
    await openPage(page, "General", "Organization")

    // Every identity is pointed at exactly one organization, so deleting the
    // only one it belongs to would orphan it. The server refuses; the page says
    // so before the click rather than after it.
    await expect(
      page.getByRole("button", { name: "Delete organization" }),
    ).toBeDisabled()
    await expect(page.getByText(/only organization/)).toBeVisible()
    // With one organization there is nothing to switch between either.
    await expect(page.getByRole("button", { name: "Switch" })).toHaveCount(0)
  })
})
