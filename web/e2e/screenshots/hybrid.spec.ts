import { expect } from "@playwright/test"

import { HYBRID_BASE_URL } from "../hybrid"
import { captureScreenshot, test } from "./fixtures"

// The third registry: the one page only a hybrid deployment serves, so it fits
// neither of the two beside it. Both of those run against the standalone gateway
// on :8000, and `HybridLanding` renders on no deployment that has a session to
// be in front of or behind, which is why this file points itself at the hybrid
// gateway playwright.config.ts already boots (otari#732).
test.use({ baseURL: HYBRID_BASE_URL })

test("hybrid landing page", async ({ page }) => {
  await page.goto("/")
  await expect(
    page.getByRole("heading", { name: "Otari gateway" }),
  ).toBeVisible()
  // Awaited past "CHECKING…", so the capture is the settled page: both rows read
  // from one `/health` query, and the heading is painted before it resolves, so
  // anchoring on the heading alone leaves the two states in the shot to whether
  // `waitForStable`'s networkidle wait beat the request. Both words are the real
  // answer this pair of gateways gives, which is what parity.hybrid.spec.ts
  // asserts them for.
  await expect(page.getByText("HEALTHY", { exact: true })).toBeVisible()
  await expect(page.getByText("CONNECTED", { exact: true })).toBeVisible()
  await captureScreenshot(page, "hybrid-landing")
})
