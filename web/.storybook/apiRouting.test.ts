import { describe, expect, it } from "vitest"

import { type ApiMocks, pathOf, route } from "./apiRouting"

describe("pathOf", () => {
  it("normalizes every shape fetch accepts to a path and query", () => {
    // The three cases used to be returned as they came, which is why the
    // gateway test below could be fooled: the same path is a relative string
    // from `apiFetch` and an absolute URL on a `Request`.
    expect(pathOf("/v1/organizations/me")).toBe("/v1/organizations/me")
    expect(pathOf("/v1/usage?window=7d")).toBe("/v1/usage?window=7d")
    expect(pathOf(new URL("https://example.com/v1/usage?window=7d"))).toBe(
      "/v1/usage?window=7d",
    )
    expect(pathOf(new Request("https://example.com/v1/organizations/me"))).toBe(
      "/v1/organizations/me",
    )
  })
})

describe("route", () => {
  it("sends Storybook's own requests to the network", () => {
    // The catalog does not load otherwise: the story index, the bundle chunks
    // and the fonts are all fetched, and none of them is the gateway's.
    for (const path of ["/index.json", "/otari/assets/index-abc.js", "/sb-manager/runtime.js"]) {
      expect(route(path, []), path).toEqual({ kind: "network" })
    }
  })

  it("answers a gateway path the story declared", () => {
    const table: ApiMocks = { "/v1/settings/mail": { configured: true } }
    expect(route("/v1/settings/mail", [table])).toEqual({
      kind: "response",
      status: 200,
      body: { configured: true },
    })
  })

  it("reads the $status envelope as a failure rather than a body", () => {
    const table: ApiMocks = {
      "/v1/settings/mail": { $status: 503, $body: { detail: "No transport." } },
    }
    expect(route("/v1/settings/mail", [table])).toEqual({
      kind: "response",
      status: 503,
      body: { detail: "No transport." },
    })
  })

  it("matches a declared pathname when the request carries a query", () => {
    const table: ApiMocks = { "/v1/usage": { total: 1 } }
    expect(route("/v1/usage?window=7d", [table])).toEqual({
      kind: "response",
      status: 200,
      body: { total: 1 },
    })
  })

  it("answers what the decorators mount, with no table at all", () => {
    // The regression this exists for. Every story is wrapped in
    // `SelectedWorkspaceProvider`, which queries the organization, so a story
    // that declared nothing still asks. It used to reach the network and 404.
    const routed = route("/v1/organizations/me", [])
    expect(routed.kind).toBe("response")
    if (routed.kind !== "response") return
    expect(routed.status).toBe(200)
    expect(routed.body).toMatchObject({ organization_member_id: expect.any(String) })
  })

  it("lets a story override the baseline", () => {
    const table: ApiMocks = {
      "/v1/organizations/me": { $status: 403, $body: { detail: "Not a member." } },
    }
    expect(route("/v1/organizations/me", [table])).toEqual({
      kind: "response",
      status: 403,
      body: { detail: "Not a member." },
    })
  })

  it("answers an undeclared gateway path with 501, never the network", () => {
    // 501 rather than 404, because a 404 is a real gateway answer that some
    // components handle gracefully, which would hide the missing path. And
    // never `network`: that is what put a failing request on every story.
    const routed = route("/v1/does/not/exist", [{ "/v1/other": {} }])
    expect(routed.kind).toBe("response")
    if (routed.kind !== "response") return
    expect(routed.status).toBe(501)
    expect(routed.body).toMatchObject({ detail: expect.stringContaining("/v1/does/not/exist") })
  })

  it("takes the first table carrying the path when several are mounted", () => {
    // An autodocs page mounts several stories at once.
    const first: ApiMocks = { "/v1/usage": { total: 1 } }
    const second: ApiMocks = { "/v1/usage": { total: 2 } }
    expect(route("/v1/usage", [first, second])).toEqual({
      kind: "response",
      status: 200,
      body: { total: 1 },
    })
  })
})
