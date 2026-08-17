import { test } from "@playwright/test"

import { seedParityUsage } from "./parity-data"

// Seeding runs as its own Playwright project (see web/playwright.config.ts), not
// as a `beforeAll` inside each parity spec: `beforeAll` can only take
// worker-scoped fixtures, and every parity spec reads the same fixture, so
// re-seeding per file would be work repeated for no isolation gained. Declaring
// it a dependency also makes the order explicit rather than resting on the
// alphabetical order of filenames.
test("seed the behavioural-parity usage fixture", async ({ page }) => {
  await seedParityUsage(page)
})
