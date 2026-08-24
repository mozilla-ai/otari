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
  // page above filtered to one service, so the entry covers them; this one is a
  // page in its own right, with a table and two dialogs nothing else captures.
  {
    route: "/tools/mcp-servers",
    name: "tools-mcp-servers",
    heading: /mcp servers/i,
  },
  { route: "/settings", name: "settings", heading: /settings/i },
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
})
