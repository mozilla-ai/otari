import { expect, test } from "@playwright/test"

import { login, nav, pageHeading, tableRows } from "./helpers"
import { PARITY_SESSION } from "./parity-data"

// Against the real gateway, so the surface gate, the list and the session read
// are the server's answers rather than fetch mocks.
test.describe("sessions", () => {
  test("Sessions opens from the rail and a seeded session opens into its spans", async ({
    page,
  }) => {
    await login(page)
    await nav(page).getByRole("link", { name: "Sessions" }).click()
    await expect(pageHeading(page, "Sessions")).toBeVisible()

    const row = tableRows(page, "Sessions").filter({
      hasText: PARITY_SESSION.agent,
    })
    await expect(row).toHaveCount(1)
    await row.getByRole("rowheader").click()

    const panel = page.getByRole("complementary", {
      name: `Trace ${PARITY_SESSION.agent}`,
    })
    await expect(
      panel.getByRole("button", {
        name: new RegExp(`^Tool, ${PARITY_SESSION.tool},`),
      }),
    ).toBeVisible()
  })
})
