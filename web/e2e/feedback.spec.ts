import { expect, type Page, test } from "@playwright/test"
import { login, MASTER_KEY } from "./helpers"

const MESSAGE =
  "We use Otari to track research spending. Comparing groups would help."

// The e2e gateway turns feedback on, so the off case is stubbed in the bootstrap.
// Every submission is intercepted and never reaches the receiver.
async function disableFeedback(page: Page) {
  await page.route("**/api/v1/bootstrap", async (route) => {
    const response = await route.fetch()
    await route.fulfill({
      response,
      json: { ...(await response.json()), feedback_enabled: false },
    })
  })
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("otari.dashboard.theme", "light")
    // The phone spec signs in by hand, past the helper that retires the welcome.
    localStorage.setItem("otari.welcome.answered", "e2e")
  })
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
  await login(page)
  // The account menu's row is the phone's entry, so on a desktop it is hidden.
  // The operator has no email, so no consent line names one.
  await page.getByRole("button", { name: /^Account:/ }).click()
  await expect(
    page
      .getByRole("dialog", { name: "Account" })
      .getByRole("button", { name: "Contact", exact: true }),
  ).toHaveCount(0)
  await page.keyboard.press("Escape")
  const topBar = page.locator("header")
  const trigger = topBar.getByRole("button", { name: "Contact", exact: true })
  expect(
    await topBar
      .getByRole("link", { name: "Documentation", exact: true })
      .evaluate((link) => link.nextElementSibling?.textContent),
  ).toBe("Contact")
  await trigger.click()
  const dialog = page.getByRole("dialog", { name: "Contact", exact: true })
  const field = dialog.getByRole("textbox", { name: "Contact" })
  await expect(field).toBeFocused()
  await expect(dialog.getByText(/^By sending/)).toHaveCount(0)
  await field.fill(MESSAGE)
  await field.press("ControlOrMeta+Enter")
  await expect(dialog.getByRole("alert")).toHaveText(
    "That didn’t reach us. Your message is still here; send it again.",
  )
  await expect(field).toHaveValue(MESSAGE)
  await dialog
    .getByRole("button", { name: /^Send to the Otari team/ })
    .dblclick()
  const thanks = page.getByRole("dialog", { name: "Thank you!" })
  await expect(thanks).toBeFocused()
  await page.keyboard.press("Escape")
  await expect(thanks).toBeHidden()
  await expect(trigger).toBeFocused()
  expect(payloads).toEqual([{ message: MESSAGE }, { message: MESSAGE }])
})

test("desktop: Close sits in the frame's corner clear of the field, and Cancel closes", async ({
  page,
}) => {
  await page.route("**/api/v1/feedback", (route) =>
    route.fulfill({ status: 204 }),
  )
  await login(page)
  const trigger = page
    .locator("header")
    .getByRole("button", { name: "Contact", exact: true })
  await trigger.click()
  const dialog = page.getByRole("dialog", { name: "Contact", exact: true })
  const close = dialog.getByRole("button", { name: "Close" })
  await expect(close).toBeVisible()
  // Measured once the entrance settles, the backdrop's and container's too,
  // and in one read: their transforms move and scale every box, and the
  // page's scrollbar going away re-centers the frame between two reads.
  await page.evaluate(() =>
    Promise.allSettled(
      document.getAnimations().map((motion) => motion.finished),
    ),
  )
  const { frame, corner, field } = await dialog.evaluate((element) => {
    const box = (target: Element | null) => {
      if (!target) throw new Error("The frame did not render")
      return target.getBoundingClientRect().toJSON() as DOMRect
    }
    return {
      frame: box(element),
      corner: box(element.querySelector('button[aria-label="Close"]')),
      field: box(element.querySelector("textarea")),
    }
  })
  expect(frame.right - corner.right).toBeLessThan(16)
  expect(corner.top - frame.top).toBeLessThan(16)
  expect(field.right).toBeLessThanOrEqual(corner.left)
  await dialog.getByRole("button", { name: "Cancel" }).click()
  await expect(dialog).toBeHidden()
  await expect(trigger).toBeFocused()
})

test("an account with an email is told where replies go, and the shortcut sends it", async ({
  page,
}) => {
  // The e2e operator has no email, so the context names one.
  await page.route("**/api/v1/organizations/me", async (route) => {
    const response = await route.fetch()
    const context = await response.json()
    await route.fulfill({
      response,
      json: {
        ...context,
        caller: { ...context.caller, email: "ada.lovelace@example.com" },
      },
    })
  })
  const payloads: unknown[] = []
  await page.route("**/api/v1/feedback", async (route) => {
    payloads.push(route.request().postDataJSON())
    await route.fulfill({ status: 204 })
  })
  await login(page)
  await page
    .locator("header")
    .getByRole("button", { name: "Contact", exact: true })
    .click()
  const dialog = page.getByRole("dialog", { name: "Contact", exact: true })
  await expect(
    dialog.getByText(
      "By sending, you agree we can email ada.lovelace@example.com about this message only.",
    ),
  ).toBeVisible()
  await dialog.getByRole("textbox", { name: "Contact" }).fill(MESSAGE)
  // offsetHeight, which the frame's entrance transform does not scale.
  const height = (frame: typeof dialog) =>
    frame.evaluate((element) => (element as HTMLElement).offsetHeight)
  const composed = await height(dialog)
  await page.keyboard.press("ControlOrMeta+Enter")
  const thanks = page.getByRole("dialog", { name: "Thank you!" })
  await expect(thanks).toContainText(
    "We’ll email you at ada.lovelace@example.com.",
  )
  // The consent line wraps, so the thanks hold the compose's height rather
  // than a fixed one, and the frame does not shrink as they swap in.
  expect(await height(thanks)).toBe(composed)
  expect(payloads).toEqual([
    { message: MESSAGE, share_email: "ada.lovelace@example.com" },
  ])
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
    await page.goto("/")
    await page.locator('input[type="password"]').fill(MASTER_KEY)
    await page.locator('input[type="password"]').press("Enter")
    const openNavigation = page.getByRole("button", {
      name: "Open navigation",
      exact: true,
    })
    await openNavigation.click()
    await page.getByRole("button", { name: /^Account:/ }).click()
    await page.getByRole("button", { name: "Contact", exact: true }).click()
    const dialog = page.getByRole("dialog", { name: "Contact", exact: true })
    await expect(openNavigation).toHaveAttribute("aria-expanded", "false")
    await expect.poll(async () => (await dialog.boundingBox())?.width).toBe(390)
    await expect
      .poll(async () => (await dialog.boundingBox())?.height)
      .toBe(844)
    const field = dialog.getByRole("textbox", { name: "Contact" })
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

test("a gateway with feedback off never offers Contact", async ({ page }) => {
  await disableFeedback(page)
  await login(page)
  await expect(
    page.getByRole("link", { name: "Documentation", exact: true }),
  ).toBeVisible()
  await expect(
    page.getByRole("button", { name: "Contact", exact: true }),
  ).toHaveCount(0)
})
