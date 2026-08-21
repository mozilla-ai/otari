import { expect, type Page } from "@playwright/test"

import { captureScreenshot, test } from "./fixtures"

// Everything a browser can reach without a session. Kept apart from the
// authenticated matrix because these are the screens an operator meets first
// and the ones most likely to be linked to from outside the app, so a
// regression here is the most expensive one to ship.

test("sign-in screen", async ({ page }) => {
  await page.goto("/")
  await expect(page.locator('input[type="password"]')).toBeVisible()
  await captureScreenshot(page, "sign-in")
})

test("sign-in screen with a rejected key", async ({ page }) => {
  await page.goto("/")
  const key = page.locator('input[type="password"]')
  await key.fill("not-the-master-key")
  await key.press("Enter")
  // The error state is a layout of its own: it adds a banner above the form,
  // which is exactly the kind of shift a screenshot catches and a unit test
  // does not.
  await expect(page.getByText(/invalid|unauthor/i).first()).toBeVisible()
  await captureScreenshot(page, "sign-in-rejected")
})

test("welcome page", async ({ page }) => {
  // Served by the gateway itself, not the SPA (src/gateway/dashboard.py), and
  // what "/" degrades to when no bundle was built.
  await page.goto("/welcome")
  await expect(page.getByRole("heading").first()).toBeVisible()
  await captureScreenshot(page, "welcome")
})

// The six auth pages that answer in front of a session (otari#650). They are
// hash paths rendered by `DeploymentRoot` ahead of the router, not routes, so
// they are here rather than in the authenticated matrix.
//
// Four of them need this deployment to be able to send mail, and the E2E
// gateway deliberately cannot: `e2e/otari.yml` configures no transport, and
// `parity.bootstrap.spec.ts` asserts that `mail_ready` is false as part of what
// this fixture *is*. So the bootstrap is stubbed for those four captures rather
// than the fixture being changed underneath a spec that depends on it. It is
// one boolean, on the one response the shell reads before it mounts.
async function withMailReady(page: Page): Promise<void> {
  await page.route("**/v1/bootstrap", async (route) => {
    const response = await route.fetch()
    const bootstrap = await response.json()
    await route.fulfill({
      response,
      json: { ...bootstrap, mail_ready: true },
    })
  })
}

test("signup", async ({ page }) => {
  await withMailReady(page)
  await page.goto("/#/signup")
  await expect(
    page.getByRole("heading", { name: "Claim your account" }),
  ).toBeVisible()
  await captureScreenshot(page, "signup")
})

test("signup on a gateway that cannot send mail", async ({ page }) => {
  // The hidden-rather-than-broken half of the same page: the sign-in screen
  // offers no link to it here, and this is what a bookmark still reaches.
  await page.goto("/#/signup")
  await expect(
    page.getByRole("heading", { name: "Not available on this gateway" }),
  ).toBeVisible()
  await captureScreenshot(page, "signup-mail-unavailable")
})

test("check your email", async ({ page }) => {
  await withMailReady(page)
  await page.goto("/#/check-email?type=signup")
  await expect(
    page.getByRole("heading", { name: "Check your email" }),
  ).toBeVisible()
  await captureScreenshot(page, "check-email")
})

test("resend verification", async ({ page }) => {
  await withMailReady(page)
  await page.goto("/#/resend-verification")
  await expect(
    page.getByRole("heading", { name: "Send a new verification link" }),
  ).toBeVisible()
  await captureScreenshot(page, "resend-verification")
})

test("recover password", async ({ page }) => {
  await withMailReady(page)
  await page.goto("/#/recover-password")
  await expect(
    page.getByRole("heading", { name: "Reset your password" }),
  ).toBeVisible()
  await captureScreenshot(page, "recover-password")
})

test("verify email with a spent link", async ({ page }) => {
  // The gateway's own refusal, not a stub: a token it never issued is exactly
  // what an expired or already-used link looks like to it.
  await page.goto("/#/verify-email?token=not-a-real-token")
  await expect(
    page.getByRole("heading", { name: "Verification failed" }),
  ).toBeVisible()
  await captureScreenshot(page, "verify-email-failed")
})

test("reset password", async ({ page }) => {
  await page.goto("/#/reset-password?token=not-a-real-token")
  await expect(
    page.getByRole("heading", { name: "Set a new password" }),
  ).toBeVisible()
  await captureScreenshot(page, "reset-password")
})
