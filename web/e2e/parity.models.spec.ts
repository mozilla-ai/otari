import { expect, type Page, test } from "@playwright/test"

import { gotoRoute, login, pageHeading } from "./helpers"

// Read-only, and the suite runs one worker against one gateway either way (see
// web/playwright.config.ts).
test.describe.configure({ mode: "serial" })

/** Open every collapsed group, so the rail is taller than the viewport. */
async function openEveryFilterGroup(page: Page): Promise<void> {
  const groups = page.locator(
    'aside[aria-label="Filters"] button[aria-expanded]',
  )
  const count = await groups.count()
  expect(count).toBeGreaterThan(0)
  for (let index = 0; index < count; index += 1) {
    const group = groups.nth(index)
    if ((await group.getAttribute("aria-expanded")) !== "true") {
      await group.click()
    }
  }
}

test.describe("models catalog filter rail", () => {
  // The rail runs past the fold once its groups are open, so a control low in
  // it is reached by scrolling the page. These cover what reaching it must not
  // do to the shell around it.

  test("the shell's outer box cannot be scrolled", async ({ page }) => {
    await login(page)
    await gotoRoute(page, "/models")
    await expect(pageHeading(page, "Models")).toBeVisible()
    await openEveryFilterGroup(page)

    const rail = page.locator('aside[aria-label="Filters"]')
    expect((await rail.boundingBox())?.height ?? 0).toBeGreaterThan(
      page.viewportSize()?.height ?? 0,
    )

    // `main` is the one scroller on the page. The box around it offers no
    // scrollbar and takes no wheel, so a scroll of it is one nothing can undo.
    const outer = page.locator("#root > div").first()
    expect(
      await outer.evaluate((element) => {
        element.scrollTop = element.scrollHeight
        return element.scrollTop
      }),
    ).toBe(0)
  })

  test("picking a filter low in the rail leaves the chrome in place", async ({
    page,
  }) => {
    await login(page)
    await gotoRoute(page, "/models")
    await expect(pageHeading(page, "Models")).toBeVisible()
    await openEveryFilterGroup(page)
    await page.locator("main#main-content").evaluate((element) => {
      element.scrollTop = element.scrollHeight
    })

    await page
      .locator('aside[aria-label="Filters"] label')
      .filter({ hasText: "Past 3 years" })
      .first()
      .click()
    await expect(page).toHaveURL(/released=1095/)

    expect((await page.getByRole("banner").boundingBox())?.y).toBe(0)
    await expect(
      page.getByRole("navigation", { name: "Sidebar" }),
    ).toBeInViewport()
  })
})
