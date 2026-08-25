import { expect } from "@playwright/test"

import { HYBRID_BASE_URL } from "../hybrid"
import { captureScreenshot, test } from "./fixtures"

test.use({ baseURL: HYBRID_BASE_URL })

test("hybrid landing page", async ({ page }) => {
  await page.goto("/")
  await expect(
    page.getByRole("heading", { name: "Otari gateway" }),
  ).toBeVisible()
  await captureScreenshot(page, "hybrid-landing")
})
