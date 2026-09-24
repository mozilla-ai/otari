import { expect, type Page, test } from "@playwright/test"
import { login, MASTER_KEY } from "./helpers"

const MESSAGE =
  "We use Otari to track research spending. Comparing groups would help."

// Feedback is off by default, and the e2e gateway runs the default. Turning it
// on in the bootstrap alone is enough, because the submission is intercepted
// below and never reaches the gateway.
async function enableFeedback(page: Page) {
  await page.route("**/api/v1/bootstrap", async (route) => {
    const response = await route.fetch()
    await route.fulfill({
      response,
      json: { ...(await response.json()), feedback_enabled: true },
    })
  })
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() =>
    localStorage.setItem("otari.dashboard.theme", "light"),
  )
})

test("desktop: the top bar opens it beside Documentation, a failure keeps the draft, a retry thanks", async ({
  page,
}) => {
  const payloads: unknown[] = []
  await page.route("**/api/v1/feedback", async (route) => {
    payloads.push(route.request().postDataJSON())
    await route.fulfill(
      payloads.length === 1
        ? { status: 503, json: { detail: "Unavailable" } }
        : { status: 204 },
    )
  })
  await enableFeedback(page)
  await login(page)
  // The account menu's row is the phone's entry, so on a desktop it is hidden.
  await page.getByRole("button", { name: /^Account:/ }).click()
  await expect(
    page
      .getByRole("dialog", { name: "Account" })
      .getByRole("button", { name: "Feedback", exact: true }),
  ).toHaveCount(0)
  await page.keyboard.press("Escape")
  const topBar = page.locator("header")
  const trigger = topBar.getByRole("button", { name: "Feedback", exact: true })
  expect(
    await topBar
      .getByRole("link", { name: "Documentation", exact: true })
      .evaluate((link) => link.nextElementSibling?.textContent),
  ).toBe("Feedback")
  await trigger.click()
  const dialog = page.getByRole("dialog", { name: "Feedback", exact: true })
  const field = dialog.getByRole("textbox", { name: "Feedback" })
  await expect(field).toBeFocused()
  await field.fill(MESSAGE)
  await field.press("ControlOrMeta+Enter")
  await expect(dialog.getByRole("alert")).toHaveText(
    "That didn’t reach us. Your message is still here; send it again.",
  )
  await expect(field).toHaveValue(MESSAGE)
  await dialog.getByRole("button", { name: /^Send to the Otari team/ }).click()
  const thanks = page.getByRole("dialog", { name: "Thank you!" })
  await expect(thanks).toBeFocused()
  await page.keyboard.press("Escape")
  await expect(thanks).toBeHidden()
  await expect(trigger).toBeFocused()
  expect(payloads).toEqual([{ message: MESSAGE }, { message: MESSAGE }])
})

test.describe("phone", () => {
  test.use({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
  })

  test("the drawer's account menu row opens a full-screen sheet that closes by its own control", async ({
    page,
  }) => {
    let submissions = 0
    await page.route("**/api/v1/feedback", async (route) => {
      submissions++
      await route.fulfill({ status: 204 })
    })
    await enableFeedback(page)
    await page.goto("/")
    await page.locator('input[type="password"]').fill(MASTER_KEY)
    await page.locator('input[type="password"]').press("Enter")
    const openNavigation = page.getByRole("button", {
      name: "Open navigation",
      exact: true,
    })
    await openNavigation.click()
    await page.getByRole("button", { name: /^Account:/ }).click()
    await page.getByRole("button", { name: "Feedback", exact: true }).click()
    const dialog = page.getByRole("dialog", { name: "Feedback", exact: true })
    await expect(openNavigation).toHaveAttribute("aria-expanded", "false")
    await expect.poll(async () => (await dialog.boundingBox())?.width).toBe(390)
    await expect
      .poll(async () => (await dialog.boundingBox())?.height)
      .toBe(844)
    const field = dialog.getByRole("textbox", { name: "Feedback" })
    // No autofocus on touch, so the keyboard does not cover the sheet on open.
    await expect(field).not.toBeFocused()
    await field.fill("A mobile idea")
    await dialog.getByRole("button", { name: "Close" }).click()
    await dialog.getByRole("button", { name: "Keep editing" }).click()
    await expect(field).toHaveValue("A mobile idea")
    await dialog
      .getByRole("button", { name: /^Send to the Otari team/ })
      .click()
    const thanks = page.getByRole("dialog", { name: "Thank you!" })
    await thanks.getByRole("button", { name: "Close" }).click()
    await expect(thanks).toBeHidden()
    await expect(openNavigation).toBeFocused()
    expect(submissions).toBe(1)
  })
})

test("a gateway with feedback off never offers it", async ({ page }) => {
  await login(page)
  await expect(
    page.getByRole("link", { name: "Documentation", exact: true }),
  ).toBeVisible()
  await expect(
    page.getByRole("button", { name: "Feedback", exact: true }),
  ).toHaveCount(0)
})
