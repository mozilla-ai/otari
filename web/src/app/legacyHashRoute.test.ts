import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"

// The script lives inline in index.html, so it can run before the bundle does;
// this runs that same text rather than a copy of it.
const html = readFileSync(resolve(__dirname, "../../index.html"), "utf8")
const script = /<script id="legacy-hash-route">([\s\S]*?)<\/script>/.exec(
  html,
)?.[1]

/** Where `href` lands once the script has run, with the app built under `base`. */
function landing(href: string, base = "/"): string {
  const url = new URL(href, "https://otari.example.com")
  let replaced: string | null = null
  const fakeWindow = {
    location: { hash: url.hash, search: url.search },
    history: {
      replaceState: (_state: unknown, _title: string, next: string) => {
        replaced = next
      },
    },
  }
  new Function("window", (script ?? "").replaceAll("%BASE_URL%", base))(
    fakeWindow,
  )
  return replaced ?? `${url.pathname}${url.search}${url.hash}`
}

describe("the legacy hash route script in index.html", () => {
  it("is there to run", () => {
    expect(script).toBeTruthy()
  })

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

  it("leaves an address with no hash route alone", () => {
    expect(landing("/keys")).toBe("/keys")
    expect(landing("/tools#section")).toBe("/tools#section")
  })
})
