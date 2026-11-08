import { test as base, expect, type Locator, type Page } from "@playwright/test"

// Shared machinery for the visual-regression suite, ported from
// otari-ai/frontend's `tests/e2e/helpers/page-setup.ts` so both suites are
// stable for the same reasons rather than each discovering its own. The specs
// stay a list of routes; everything that makes a capture reproducible is here.

/** Same key `index.html`'s pre-paint script and `useTheme` both read. */
const THEME_STORAGE_KEY = "otari.dashboard.theme"

/**
 * The instant every capture believes it is.
 *
 * Pinned so anything derived from the current time (a "today" default, a date
 * range, a freshness line) renders the same bytes on every run.
 */
const FROZEN_NOW = new Date("2025-01-15T12:00:00Z")

/**
 * The theme a project captures, taken from its name rather than a typed
 * Playwright option: the matrix in playwright.config.ts already spells it
 * (`screenshots-mobile-dark`), and reading it back here keeps the config plain
 * instead of making it generic over a custom option type.
 */
function themeFor(projectName: string): "light" | "dark" {
  return projectName.endsWith("-dark") ? "dark" : "light"
}

export const test = base.extend({
  page: async ({ page }, use, testInfo) => {
    // `setFixedTime`, not `clock.install`: the latter also fakes setTimeout,
    // setInterval and requestAnimationFrame, which deadlocks anything that
    // schedules deferred work, including TanStack Query's refetch timers and
    // React's own scheduler.
    await page.clock.setFixedTime(FROZEN_NOW)

    await page.addInitScript(
      ([key, value]) => {
        // Seeded before the first navigation so the pre-paint script in
        // index.html applies the theme on the first frame. Setting it after
        // would capture whatever the OS preference resolved to until React
        // mounted.
        window.localStorage.setItem(key, value)

        // Freeze CSS animations and transitions, and take the scrollbar out of
        // the capture: its width and its auto-hide behavior differ between a
        // developer machine and the CI container, and on a full-page capture it
        // runs the whole height of the image.
        const injectStyles = () => {
          const style = document.createElement("style")
          style.textContent = `
            *, *::before, *::after {
              animation-duration: 0s !important;
              animation-delay: 0s !important;
              transition-duration: 0s !important;
              transition-delay: 0s !important;
              caret-color: transparent !important;
            }
            ::-webkit-scrollbar { display: none !important; }
            html { scrollbar-width: none !important; }
          `
          document.head.appendChild(style)
        }
        // An init script can run before <html> is parsed, so defer if it has.
        if (document.head) injectStyles()
        else document.addEventListener("DOMContentLoaded", injectStyles)
      },
      [THEME_STORAGE_KEY, themeFor(testInfo.project.name)] as const,
    )

    await use(page)
  },
})

/**
 * Regions whose pixels are not the thing under test.
 *
 * Recharts animates its series in on mount through JavaScript, which neither
 * the injected CSS freeze nor Playwright's `animations: "disabled"` reaches, so
 * a chart is a coin flip at capture time. Relative timestamps move with the gap
 * between the frozen clock and rows the parity seed created at run time;
 * otari-ai needs no such mask because its fixtures are static, and this one
 * goes away here too on the day these run against mocked responses.
 */
function volatileRegions(page: Page): Locator[] {
  return [
    page.locator(".recharts-responsive-container"),
    page.getByText(/\bago\b/),
  ]
}

/**
 * Wait for the page to stop moving.
 *
 * `networkidle` says the query layer has settled rather than standing in for
 * it, capped because a route that refetches from an effect would otherwise hold
 * the capture until the test times out, and such a page is visually stable
 * anyway. Fonts are self-hosted and swap in late enough to change metrics
 * mid-capture. Scrolling to the top settles anything driven by scroll position
 * or an IntersectionObserver into its above-the-fold state, which a full-page
 * capture would otherwise race. SVG SMIL animations run on a timeline the CSS
 * freeze does not touch, so each one is paused at t=0. The frames at the end
 * let an observer callback and the React commit it causes land before the
 * shutter.
 */
export async function waitForStable(page: Page): Promise<void> {
  await page.waitForLoadState("networkidle", { timeout: 5000 }).catch(() => {})
  await page.evaluate(async () => {
    window.scrollTo(0, 0)
    await document.fonts?.ready
    for (const svg of Array.from(document.querySelectorAll("svg"))) {
      if (typeof svg.pauseAnimations === "function") {
        svg.pauseAnimations()
        svg.setCurrentTime(0)
      }
    }
    const nextFrame = () =>
      new Promise((resolve) => requestAnimationFrame(() => resolve(null)))
    await nextFrame()
    await nextFrame()
    await nextFrame()
  })
}

/**
 * Capture one page state under the project's viewport and theme.
 *
 * `fullPage` because a dashboard page's interesting parts are usually below the
 * fold on the mobile viewport, and a clipped capture would silently stop
 * covering them. The comparison budgets live in playwright.config.ts, so every
 * capture in the matrix is judged by the same rule.
 */
export async function captureScreenshot(
  page: Page,
  name: string,
): Promise<void> {
  await waitForStable(page)
  await expect(page).toHaveScreenshot(`${name}.png`, {
    fullPage: true,
    mask: volatileRegions(page),
    // Bounds the capture-and-compare retry loop on its own, so a page that
    // never visually settles fails here rather than consuming the whole test
    // budget first. otari-ai sets this under `expect.toHaveScreenshot` in its
    // config; this Playwright's types reject `timeout` there, and putting it in
    // `expect.timeout` instead would also loosen every assertion in the
    // behavioral suite, so it lives at the one call site that wants it.
    timeout: 8_000,
  })
}
