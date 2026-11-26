import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import { installLegacyHashRoute } from "./legacyHashRoute"

/** A browser on `href`, with the move installed in it as the page loads. */
function loadPage(href: string, base = "/") {
  const url = new URL(href, "https://otari.example.com")
  const listeners: Record<string, () => void> = {}
  const location = {
    pathname: url.pathname,
    search: url.search,
    hash: url.hash,
  }
  const fakeWindow = {
    location,
    history: {
      replaceState: (_state: unknown, _title: string, next: string) => {
        const moved = new URL(next, "https://otari.example.com")
        location.pathname = moved.pathname
        location.search = moved.search
        location.hash = moved.hash
      },
    },
    addEventListener: (type: string, listener: () => void) => {
      listeners[type] = listener
    },
  }
  installLegacyHashRoute(fakeWindow, base)
  return {
    address: () => `${location.pathname}${location.search}${location.hash}`,
    /** The fragment alone changing, as following a link to `#…` in the tab does. */
    changeHash(hash: string) {
      location.hash = hash
      listeners.hashchange?.()
    },
  }
}

/** Where `href` lands once the script has run, with the app built under `base`. */
function landing(href: string, base = "/"): string {
  return loadPage(href, base).address()
}

describe("installLegacyHashRoute", () => {
  it("moves a hash route to the path it names", () => {
    expect(landing("/#/organization/provider-keys")).toBe(
      "/organization/provider-keys",
    )
    expect(landing("/#/")).toBe("/")
    expect(landing("/#/models/openai:gpt-4.1")).toBe("/models/openai:gpt-4.1")
  })

  it("puts the path under the base the dashboard was built for", () => {
    expect(landing("/dashboard/#/keys", "/dashboard/")).toBe("/dashboard/keys")
  })

  it("carries a route's query into the page's own, after any already there", () => {
    expect(landing("/#/activity?status=error")).toBe("/activity?status=error")
    expect(
      landing(
        "/dashboard/?region=eu#/organization/billing?wallet=ok",
        "/dashboard/",
      ),
    ).toBe("/dashboard/organization/billing?region=eu&wallet=ok")
  })

  it("keeps a mailed link's secret in the fragment, out of any request", () => {
    expect(landing("/#/verify-email?token=abc")).toBe("/verify-email#token=abc")
    expect(landing("/#/reset-password?token=abc")).toBe(
      "/reset-password#token=abc",
    )
    expect(landing("/#/accept-invitation?token=abc")).toBe(
      "/accept-invitation#token=abc",
    )
    expect(landing("/#/setup?key=otari-mk-abc")).toBe("/setup#key=otari-mk-abc")
    expect(landing("/?region=eu#/verify-email?token=abc")).toBe(
      "/verify-email?region=eu#token=abc",
    )
  })

  it("renames an OAuth callback to the route that replaced it", () => {
    expect(landing("/#/auth/google/callback?code=c&state=s")).toBe(
      "/sign-in/google/callback?code=c&state=s",
    )
  })

  it("moves an old link opened in a tab already on the dashboard", () => {
    const page = loadPage("/")
    page.changeHash("#/routing")
    expect(page.address()).toBe("/routing")
    page.changeHash("#/verify-email?token=abc")
    expect(page.address()).toBe("/verify-email#token=abc")
  })

  it("carries the page's own query only from the dashboard's root", () => {
    const page = loadPage("/activity?status=error")
    page.changeHash("#/routing")
    expect(page.address()).toBe("/routing")
  })

  it("leaves an address with no hash route alone", () => {
    expect(landing("/keys")).toBe("/keys")
    expect(landing("/tools#section")).toBe("/tools#section")
  })
})

describe("main.tsx", () => {
  it("imports the move first, so it runs before the router reads the address", () => {
    const main = readFileSync(resolve(__dirname, "../main.tsx"), "utf8")
    expect(/^import\s+(.+)$/m.exec(main)?.[1]).toBe(
      '"@/app/legacyHashRouteOnLoad"',
    )
  })
})
