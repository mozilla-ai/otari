import { expect, type Page } from "@playwright/test"

import { gotoRoute, login } from "../helpers"
import { captureScreenshot, test } from "./fixtures"

// One entry per destination the nav registry can reach. The matrix in
// playwright.config.ts multiplies each of these by three viewports and both
// themes, so adding a page here is the whole cost of covering it.
//
// Routes are opened directly rather than clicked to, because this suite is
// about how a page renders, not how it is reached: the parity specs own
// navigation. The router is on hash history, so `gotoRoute` builds "/#<route>".
const WORKSPACE_ROUTES: ReadonlyArray<{
  readonly route: string
  readonly name: string
  readonly heading: RegExp
}> = [
  { route: "/", name: "overview", heading: /overview/i },
  { route: "/models", name: "models", heading: /models/i },
  { route: "/routing", name: "routing", heading: /routing/i },
  { route: "/providers", name: "providers", heading: /provider/i },
  { route: "/keys", name: "keys", heading: /keys/i },
  // The selected workspace's roster, which is not the organization one below:
  // this page is on the workspace rail and stops at the workspace it names.
  { route: "/members", name: "members", heading: /members/i },
  { route: "/budgets", name: "budgets", heading: /budgets/i },
  // The member roster now carries what the users page used to: model access,
  // blocking, and per-person spend. It is on the organization rail, which this
  // matrix reaches by URL like every other route here.
  {
    route: "/organization/members",
    name: "organization-members",
    heading: /members/i,
  },
  { route: "/usage", name: "usage", heading: /usage/i },
  { route: "/activity", name: "activity", heading: /activity/i },
  { route: "/tools", name: "tools", heading: /tools/i },
  // The one Tools child with an entry of its own. The other two render the
  // page above filtered to one service, so its entry covers them; this one is a
  // page in its own right, and nothing else captures its table. Captured at
  // rest, so its two dialogs are covered by neither this nor the vitest suite.
  {
    route: "/tools/mcp-servers",
    name: "tools-mcp-servers",
    heading: /mcp servers/i,
  },
  { route: "/settings", name: "settings", heading: /settings/i },
  // Deployment-wide accounts, on the organization rail beside Settings. Not the
  // members roster above: that one stops at this organization's boundary.
  {
    route: "/admin/accounts",
    name: "admin-accounts",
    heading: /accounts/i,
  },
  // Reached from the account menu rather than the rail, and covered here all
  // the same: what this matrix is for is how a page renders on a phone and in
  // the dark, which does not depend on which control opens it.
  { route: "/account", name: "account", heading: /account settings/i },
  { route: "/docs", name: "docs", heading: /./ },
]

async function open(page: Page, route: string, heading: RegExp): Promise<void> {
  await gotoRoute(page, route)
  await expect(
    page.getByRole("heading", { name: heading }).first(),
  ).toBeVisible()
}

// Deliberately not `mode: "serial"`, though the ordering it provides is already
// there: playwright.config.ts runs the whole suite with one worker, so these run
// in declaration order regardless. What serial would add is the part this suite
// cannot afford, that a failing test skips every test after it in the file. One
// page whose look changed would then take every page below it out of the run,
// and a review would see one diff where there are five.
test.describe("workspace rail", () => {
  for (const { route, name, heading } of WORKSPACE_ROUTES) {
    test(name, async ({ page }) => {
      await login(page)
      await open(page, route, heading)
      await captureScreenshot(page, name)
    })
  }
})

test.describe("organization rail", () => {
  test("organization general", async ({ page }) => {
    await login(page)
    // Reached by URL like everything else in this file, not through the
    // sidebar's footer link: on the mobile viewport that link lives inside the
    // closed drawer, so clicking it times out and the page is never captured at
    // the size most worth looking at.
    await gotoRoute(page, "/organization")
    await expect(
      page.getByRole("heading", { name: /organization/i }).first(),
    ).toBeVisible()
    await captureScreenshot(page, "organization-general")
  })

  test("organization members", async ({ page }) => {
    await login(page)
    await gotoRoute(page, "/organization/members")
    await expect(
      page.getByRole("heading", { name: /members/i }).first(),
    ).toBeVisible()
    await captureScreenshot(page, "organization-members")
  })

  test("workspaces", async ({ page }) => {
    await login(page)
    await gotoRoute(page, "/workspaces")
    await expect(
      page.getByRole("heading", { name: /workspaces/i }).first(),
    ).toBeVisible()
    await captureScreenshot(page, "workspaces")
  })

  test("organization model pricing", async ({ page }) => {
    await login(page)
    await gotoRoute(page, "/organization/pricing")
    await expect(
      page.getByRole("heading", { name: /model pricing/i }).first(),
    ).toBeVisible()
    await captureScreenshot(page, "organization-model-pricing")
  })

  test("organization usage", async ({ page }) => {
    // This gateway is standalone, whose bootstrap does not report the
    // organization_usage surface (its organization is the deployment, so /usage
    // already answers it whole). The surface is patched in before the shell
    // boots, which is why the stub precedes `login` here: `gotoRoute` changes
    // only the hash, so the boot-time bootstrap read is the one that counts.
    // Everything behind the route is served for real, because
    // /v1/organizations/me/usage is mounted on both editions.
    await page.route("**/v1/bootstrap", async (route) => {
      const response = await route.fetch()
      const bootstrap = await response.json()
      await route.fulfill({
        json: {
          ...bootstrap,
          surfaces: [...bootstrap.surfaces, "organization_usage"],
        },
      })
    })
    await login(page)
    await gotoRoute(page, "/organization/usage")
    await expect(
      page.getByRole("heading", { name: /organization usage/i }).first(),
    ).toBeVisible()
    await captureScreenshot(page, "organization-usage")
  })
})

/**
 * The organization-admin view of Spend & budgets.
 *
 * `/budgets` above captures the deployment-operator page, because `login`
 * exchanges the master key and that session is the bootstrap operator, so
 * `deployment_operator` is true and the route resolves to the other page every
 * time. Nothing else in this suite renders the admin page, which the frontend
 * standards owe a screenshot for.
 *
 * A seeded admin identity with a password would be the faithful way in, and this
 * harness has no password login: `login` is the only auth path and it is
 * master-key only. So the caller's own context read is stubbed instead, which is
 * the technique `public.spec.ts` already uses for the invitation pages. The two
 * organization-scoped reads are stubbed with it, because a master-key session is
 * still what the gateway sees and it would answer them for the operator's
 * organization rather than the shape this page is being captured for.
 *
 * What this does and does not cover: the layout, both themes and all three
 * viewports, which is what the matrix is for. It is not a claim that the role
 * gate works, which the vitest suite and the route tests own.
 */
async function stubAdminSpendView(page: Page): Promise<void> {
  await page.route("**/v1/organizations/me", async (route) => {
    const response = await route.fetch()
    const context = await response.json()
    await route.fulfill({
      json: { ...context, role: "admin", deployment_operator: false },
    })
  })
  await page.route("**/v1/organizations/me/budgets*", async (route) => {
    await route.fulfill({
      json: {
        data: [
          {
            budget_id: "11111111-1111-1111-1111-111111111111",
            organization_id: "22222222-2222-2222-2222-222222222222",
            name: "Engineering monthly",
            max_budget: 2500,
            budget_duration_sec: null,
            reset_alignment: "calendar_month",
            ceiling_count: 2,
            created_at: "2026-08-01T00:00:00+00:00",
            updated_at: "2026-08-01T00:00:00+00:00",
          },
          {
            budget_id: "33333333-3333-3333-3333-333333333333",
            organization_id: "22222222-2222-2222-2222-222222222222",
            name: "Trials, daily",
            max_budget: 25,
            budget_duration_sec: null,
            reset_alignment: "calendar_day",
            ceiling_count: 0,
            created_at: "2026-08-02T00:00:00+00:00",
            updated_at: "2026-08-02T00:00:00+00:00",
          },
        ],
        count: 2,
      },
    })
  })
  await page.route("**/v1/organizations/me/spend-ceilings*", async (route) => {
    await route.fulfill({
      json: {
        data: [
          {
            id: "44444444-4444-4444-4444-444444444444",
            scope_type: "organization",
            scope_id: "22222222-2222-2222-2222-222222222222",
            provider_key_id: null,
            budget_id: "11111111-1111-1111-1111-111111111111",
            name: "Whole organization",
            max_budget: 2500,
            current_spend: 412.5,
            reserved_spend: 3.25,
            budget_duration_sec: null,
            reset_alignment: "calendar_month",
            period_start: "2026-08-01T00:00:00+00:00",
            period_end: "2026-09-01T00:00:00+00:00",
            manageable: true,
            created_at: "2026-08-01T00:00:00+00:00",
            updated_at: "2026-08-01T00:00:00+00:00",
          },
          {
            // The row the otari-ai cutover writes: enforcing, and its figure set
            // outside this organization. Included so the marker is captured.
            id: "55555555-5555-5555-5555-555555555555",
            scope_type: "workspace",
            scope_id: "66666666-6666-6666-6666-666666666666",
            provider_key_id: "openai-eu",
            budget_id: "77777777-7777-7777-7777-777777777777",
            name: null,
            max_budget: 100,
            current_spend: 12,
            reserved_spend: 0,
            budget_duration_sec: 86400,
            reset_alignment: null,
            period_start: "2026-08-30T00:00:00+00:00",
            period_end: "2026-08-31T00:00:00+00:00",
            manageable: false,
            created_at: "2026-08-01T00:00:00+00:00",
            updated_at: "2026-08-01T00:00:00+00:00",
          },
        ],
        count: 2,
      },
    })
  })
}

/**
 * The member view of the workspace Members page.
 *
 * The organization's own roster is read-only there for a caller who does not
 * manage the organization (otari-ai#1960), because Members & roles is on the
 * organization rail and the shell opens that rail to nobody else. `login` is the
 * bootstrap operator, who is an owner, so the only way to capture the other
 * caller is to stub their context, as `stubAdminSpendView` does above and for
 * the same reason. The roster itself is served for real.
 */
test.describe("organization member", () => {
  test("workspace members", async ({ page }) => {
    await login(page)
    await page.route("**/v1/organizations/me", async (route) => {
      const response = await route.fetch()
      const context = await response.json()
      await route.fulfill({
        json: { ...context, role: "member", deployment_operator: false },
      })
    })
    await gotoRoute(page, "/members")
    await expect(
      page.getByRole("heading", { name: /^members$/i }).first(),
    ).toBeVisible()
    // Awaited past the loading rows, so the capture is the populated table.
    await expect(
      page.getByRole("heading", { name: "Organization members" }),
    ).toBeVisible()
    await captureScreenshot(page, "members-organization-roster")
  })
})

test.describe("organization admin", () => {
  test("organization spend and budgets", async ({ page }) => {
    await login(page)
    await stubAdminSpendView(page)
    await gotoRoute(page, "/budgets")
    await expect(
      page.getByRole("heading", { name: /spend & budgets/i }).first(),
    ).toBeVisible()
    // Awaited past the loading rows, so the capture is the populated tables
    // rather than two spinners.
    await expect(page.getByText("Engineering monthly")).toBeVisible()
    await expect(page.getByText("Set at the deployment level")).toBeVisible()
    await captureScreenshot(page, "organization-spend-budgets")
  })
})
