import { expect, test } from "@playwright/test"

import { login, nav, tableRows } from "../helpers"
import { PARITY_SESSION } from "../parity-data"
import { docScreenshot } from "./fixtures"

// docs/sessions.md. The session is the one the seed project exports over OTLP,
// so the walk creates nothing: no route deletes a session, and one left behind
// would show in the parity flows.
test("look at a session", async ({ page }) => {
  await login(page)
  await nav(page).getByRole("link", { name: "Overview" }).click()
  await page
    .getByRole("navigation", { name: "Overview views" })
    .getByRole("link", { name: "Sessions" })
    .click()
  const row = tableRows(page, "Sessions").filter({
    hasText: PARITY_SESSION.agent,
  })
  await expect(row).toHaveCount(1)
  await docScreenshot(page, "sessions-list")

  await row.getByRole("rowheader").click()
  const panel = page.getByRole("complementary", {
    name: `Trace ${PARITY_SESSION.agent}`,
  })
  await expect(
    panel.getByRole("button", {
      name: new RegExp(`^Tool, ${PARITY_SESSION.tool},`),
    }),
  ).toBeVisible()
  await docScreenshot(page, "sessions-panel")
})

// "Keep what a session was about". The switch is only looked at: whether it can
// be turned on depends on the deployment.
test("find the content switch", async ({ page }) => {
  await login(page)
  await page.getByRole("link", { name: "Organization", exact: true }).click()
  await nav(page).getByRole("link", { name: "Workspaces" }).click()
  await page
    .getByRole("main")
    .getByRole("link", { name: "Default workspace", exact: true })
    .click()
  await page.getByRole("main").getByRole("button", { name: "General" }).click()
  await expect(
    page.getByRole("switch", { name: "Keep what agent sessions were about" }),
  ).toBeVisible()
})
