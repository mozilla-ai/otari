import { expect, type Page, test } from "@playwright/test"

import { API_ROOT } from "@/shared/api/client"

import { authHeaders, expectOk, nav } from "../helpers"
import { docScreenshot } from "./fixtures"

const ADMIN = "dana@example.com"
const ADMIN_PASSWORD = "docs-admin-password"
const BUDGET = "Engineering monthly"
const CEILING = "Whole organization"

/**
 * Sign in as an organization admin, which is who docs/budgets.md is written
 * for: the master key signs in as the deployment operator, and an operator's
 * Spend & budgets is the deployment-wide page instead (`BudgetsPage`).
 */
async function signInAsAdmin(page: Page): Promise<void> {
  const invited = await page.request.post(
    `${API_ROOT}/organizations/me/member-invitations`,
    { headers: authHeaders, data: { email: ADMIN, role: "admin" } },
  )
  await expectOk(invited, "invite the admin")
  const { accept_link } = (await invited.json()) as { accept_link: string }
  const token = decodeURIComponent(
    /[?&]token=([^&]+)/.exec(accept_link)?.[1] ?? "",
  )
  const accepted = await page.request.post(`${API_ROOT}/invitations/accept`, {
    data: { token, password: ADMIN_PASSWORD, terms_accepted: true },
  })
  await expectOk(accepted, "accept the invitation")
  await page.goto("/")
  await page.getByLabel("Email").fill(ADMIN)
  await page.getByLabel("Password").fill(ADMIN_PASSWORD)
  await page.getByRole("button", { name: "Sign in" }).click()
  await expect(nav(page).getByRole("link", { name: "Overview" })).toBeVisible()
}

async function listed<T>(page: Page, path: string): Promise<T[]> {
  const response = await page.request.get(`${API_ROOT}${path}`, {
    headers: authHeaders,
  })
  await expectOk(response, `list ${path}`)
  return ((await response.json()) as { data: T[] }).data
}

async function remove(page: Page, path: string): Promise<void> {
  const response = await page.request.delete(`${API_ROOT}${path}`, {
    headers: authHeaders,
  })
  await expectOk(response, `delete ${path}`)
}

/**
 * Remove what the walk created, so the behavioral specs sharing this database
 * never see it. The ceiling goes before its budget, which it holds.
 */
async function cleanUp(page: Page): Promise<void> {
  type Named = { name: string | null }
  const base = "/organizations/me"
  for (const ceiling of await listed<Named & { id: string }>(
    page,
    `${base}/spend-ceilings`,
  )) {
    if (ceiling.name === CEILING) {
      await remove(page, `${base}/spend-ceilings/${ceiling.id}`)
    }
  }
  for (const budget of await listed<Named & { budget_id: string }>(
    page,
    `${base}/budgets`,
  )) {
    if (budget.name === BUDGET) {
      await remove(page, `${base}/budgets/${budget.budget_id}`)
    }
  }
  for (const member of await listed<{
    email: string
    organization_member_id: string
  }>(page, `${base}/members?search=${encodeURIComponent(ADMIN)}`)) {
    if (member.email === ADMIN) {
      await remove(page, `${base}/members/${member.organization_member_id}`)
    }
  }
}

// docs/budgets.md, "Create a budget and a spend ceiling" and "Check a budget's
// status".
test("create a budget and a spend ceiling", async ({ page }) => {
  try {
    await signInAsAdmin(page)
    await page.getByRole("link", { name: "Organization", exact: true }).click()
    await nav(page).getByRole("link", { name: "Spend & budgets" }).click()

    await page.getByRole("button", { name: "Add budget" }).click()
    const budgetDialog = page.getByRole("dialog")
    await budgetDialog.getByLabel("Name").fill(BUDGET)
    await budgetDialog.getByLabel("Limit (USD)").fill("500")
    await expect(
      budgetDialog.getByRole("button", { name: /Resets/ }),
    ).toBeVisible()
    await docScreenshot(budgetDialog, "budgets-add-budget")
    await budgetDialog.getByRole("button", { name: "Add budget" }).click()
    await expect(budgetDialog).toBeHidden()

    await page.getByRole("button", { name: "Add ceiling" }).click()
    const ceilingDialog = page.getByRole("dialog")
    await expect(
      ceilingDialog.getByRole("button", { name: /Capping/ }),
    ).toBeVisible()
    await expect(
      ceilingDialog.getByRole("combobox", { name: "Provider instance" }),
    ).toBeVisible()
    await ceilingDialog.getByRole("button", { name: /Budget/ }).click()
    await page.getByRole("option", { name: BUDGET }).click()
    await ceilingDialog.getByLabel("Name").fill(CEILING)
    await docScreenshot(ceilingDialog, "budgets-add-ceiling")
    await ceilingDialog.getByRole("button", { name: "Add ceiling" }).click()
    await expect(ceilingDialog).toBeHidden()

    await expect(page.getByText(CEILING, { exact: true })).toBeVisible()
    await docScreenshot(page, "budgets-spend-and-budgets")
  } finally {
    await cleanUp(page)
  }
})
