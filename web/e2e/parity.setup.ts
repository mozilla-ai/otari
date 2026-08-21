import { test } from "@playwright/test"

import { dismissSetupGuide } from "./helpers"
import { seedParityUsage } from "./parity-data"

// Seeding runs as its own Playwright project (see web/playwright.config.ts), not
// as a `beforeAll` inside each parity spec: `beforeAll` can only take
// worker-scoped fixtures, and every parity spec reads the same fixture, so
// re-seeding per file would be work repeated for no isolation gained. Declaring
// it a dependency also makes the order explicit rather than resting on the
// alphabetical order of filenames.
test("seed the behavioural-parity usage fixture", async ({ page }) => {
  await seedParityUsage(page)
  // The fixture is imported usage, which is deliberately not a call to this
  // gateway, so the first-request setup guide is still on offer and would sit at
  // the top of the Overview the parity specs read. Retiring it here keeps the
  // fixture, not another spec's side effects, in charge of what those pages
  // start from. Idempotent server-side, so this is safe on a warm database.
  await dismissSetupGuide(page)
})
