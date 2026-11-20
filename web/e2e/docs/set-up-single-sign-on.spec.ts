import { expect, type Page, test } from "@playwright/test"

import type { OrganizationSso, OrganizationSsoPreflight } from "@/client"
import { API_ROOT } from "@/shared/api/client"

import { login, nav } from "../helpers"
import { docScreenshot } from "./fixtures"

const OWNER_ID = "00000000-0000-4000-8000-000000000001"

/** A proof verified today, so its expiry reads the same on every capture. */
function inNinetyDays(): string {
  return new Date(Date.now() + 90 * 24 * 60 * 60 * 1000).toISOString()
}

const CONNECTION = {
  id: "00000000-0000-4000-8000-0000000000c1",
  display_name: "Example Corp",
  issuer: "https://login.example.com",
  resolved_issuer: "https://login.example.com",
  client_id: "otari-dashboard",
  client_secret_last4: "x7Qa",
  default_role: "member",
  enabled: true,
  discovered_at: new Date().toISOString(),
  domains: [
    { domain: "example.com", enabled: true, proof_expires_at: inNinetyDays() },
  ],
}

/**
 * Stand in for the organization's single sign-on settings. This gateway has no
 * public address, so it serves no redirect URI, and saving a connection would
 * read an identity provider's discovery document, which nothing here runs.
 * Nothing in this walk is saved.
 */
async function fakeSso(
  page: Page,
  connection: OrganizationSso["connection"],
): Promise<void> {
  const settings: OrganizationSso = {
    connection,
    redirect_uri: "https://otari.example.com/auth/sso/callback",
    enforcement: {
      required: false,
      break_glass_user_id: null,
      break_glass_waived: false,
    },
  }
  const preflight: OrganizationSsoPreflight = {
    affected_count: 2,
    total_members: 6,
    affected_members: [
      { user_id: OWNER_ID, full_name: "Sam Rivera", email: "sam@example.com" },
      {
        user_id: "00000000-0000-4000-8000-000000000002",
        full_name: "Priya Shah",
        email: "priya@example.com",
      },
    ],
    affected_members_truncated: false,
    break_glass_candidates: [
      { user_id: OWNER_ID, full_name: "Sam Rivera", email: "sam@example.com" },
    ],
  }
  await page.route(`**${API_ROOT}/organizations/me/sso`, (route) =>
    route.request().method() === "GET"
      ? route.fulfill({ json: settings })
      : route.abort(),
  )
  await page.route(`**${API_ROOT}/organizations/me/sso/preflight`, (route) =>
    route.fulfill({ json: preflight }),
  )
}

async function openSingleSignOn(page: Page): Promise<void> {
  await login(page)
  await page.getByRole("link", { name: "Organization", exact: true }).click()
  await nav(page).getByRole("link", { name: "Single sign-on" }).click()
}

// The domain is claimed on Email domains, which docs/members.md walks; this
// only checks the page the step sends the reader to is still there.
async function passEmailDomains(page: Page): Promise<void> {
  await nav(page).getByRole("link", { name: "Email domains" }).click()
  await expect(page.getByRole("button", { name: "Claim domain" })).toBeVisible()
  await nav(page).getByRole("link", { name: "Single sign-on" }).click()
}

function section(page: Page, heading: string) {
  return page
    .locator("section")
    .filter({ has: page.getByRole("heading", { name: heading, exact: true }) })
}

// The dialog is captured filled in and never submitted.
test("add a connection", async ({ page }) => {
  // Tall enough for the dialog's last field, which scrolls out of view at 800.
  await page.setViewportSize({ width: 1280, height: 1000 })
  await fakeSso(page, null)
  await openSingleSignOn(page)

  const setup = section(page, "Provider setup")
  await expect(setup.getByLabel("Redirect URI")).toHaveValue(
    "https://otari.example.com/auth/sso/callback",
  )
  await docScreenshot(setup, "sso-provider-setup")

  await page.getByRole("button", { name: "Add connection" }).click()
  const dialog = page.getByRole("dialog")
  await dialog.getByLabel("Display name").fill(CONNECTION.display_name)
  await dialog.getByLabel("Issuer").fill(CONNECTION.issuer)
  await dialog.getByLabel("Client ID").fill(CONNECTION.client_id)
  await dialog.getByLabel("Client secret").fill("example-client-secret")
  await expect(
    dialog.getByRole("button", { name: /New sign-ins join as/ }),
  ).toContainText("Member")
  await expect(
    dialog.getByRole("button", { name: "Add connection" }),
  ).toBeEnabled()
  await docScreenshot(dialog, "sso-add-connection")
})

// Enforcement is chosen and captured, never saved.
test("check the domain and require sign-in", async ({ page }) => {
  await fakeSso(page, CONNECTION)
  await openSingleSignOn(page)
  await passEmailDomains(page)

  const summary = section(page, CONNECTION.display_name)
  await expect(
    summary.getByRole("heading", { name: "Domains that admit sign-in" }),
  ).toBeVisible()
  await expect(summary.getByText("example.com", { exact: true })).toBeVisible()
  await docScreenshot(summary, "sso-connection")

  const enforcement = section(page, "Require sign-in")
  await expect(enforcement.getByText("2 of 6 members")).toBeVisible()
  // Pressed by its words: react-aria hides the input itself, so `check` waits
  // on a control that never becomes visible.
  await enforcement.getByText("Require sign-in through this connection").click()
  await expect(
    enforcement.getByRole("checkbox", {
      name: "Require sign-in through this connection",
    }),
  ).toBeChecked()
  await enforcement
    .getByRole("button", {
      name: /Owner who can still sign in if your provider breaks/,
    })
    .click()
  await page.getByRole("option", { name: /Sam Rivera/ }).click()
  await expect(
    enforcement.getByRole("button", { name: "Save enforcement" }),
  ).toBeEnabled()
  await docScreenshot(enforcement, "sso-require-sign-in")
})

// Captured signed out, with the address typed and never looked up. Only the
// company sign-in is captured: the rest of this gateway's sign-in screen is its
// master key, which a reader's deployment does not show.
test("sign in with your company", async ({ page }) => {
  await page.goto("/")
  await page
    .getByRole("button", { name: "Sign in with your company (SSO)" })
    .click()
  await page.getByLabel("Work email").fill("sam@example.com")
  const form = page
    .locator("form")
    .filter({ has: page.getByLabel("Work email") })
  await expect(form.getByRole("button", { name: "Continue" })).toBeEnabled()
  await page.getByLabel("Work email").blur()
  await docScreenshot(form, "sso-sign-in", { margin: 8 })
})
