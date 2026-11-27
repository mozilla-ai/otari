import { expect, test } from "@playwright/test"

import { login, pageHeading } from "./helpers"

// Against the built bundle and the real gateway, because both halves of a
// dashboard address live there: the gateway answering a page's path with the
// dashboard, and the script in index.html moving a link from before path
// routing onto the path it names.
test.describe("dashboard addresses", () => {
  test("a reload of a page's own path opens that page", async ({ page }) => {
    await login(page)
    await page.goto("/routing")
    await expect(pageHeading(page, "Routing")).toBeVisible()

    await page.reload()
    await expect(page).toHaveURL(/\/routing$/)
    await expect(pageHeading(page, "Routing")).toBeVisible()
  })

  test("a link from before path routing lands on the path it named", async ({
    page,
  }) => {
    await login(page)
    await page.goto("/budgets")
    await expect(pageHeading(page, "Spend & budgets")).toBeVisible()
    // A page load: the script runs ahead of the bundle.
    await page.goto("/#/routing")
    await expect(page).toHaveURL(/:\d+\/routing$/)
    await expect(pageHeading(page, "Routing")).toBeVisible()
  })

  test("an old link opened in a tab already on the dashboard moves too", async ({
    page,
  }) => {
    await login(page)
    // `login` lands on the root, so this changes only the fragment and loads
    // nothing, which is the case the script's `hashchange` listener covers.
    await page.goto("/#/routing")
    await expect(page).toHaveURL(/:\d+\/routing$/)
    await expect(pageHeading(page, "Routing")).toBeVisible()
  })

  test("an API path a browser opens stays the API's answer", async ({
    page,
  }) => {
    const response = await page.goto("/api/v1/no-such-resource")
    expect(response?.status()).toBe(404)
    expect(response?.headers()["content-type"]).toContain("application/json")
  })
})
