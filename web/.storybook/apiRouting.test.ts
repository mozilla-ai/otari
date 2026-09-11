import { describe, expect, it } from "vitest"

import { API_ROOT } from "@/shared/api/client"
import { type ApiMocks, pathOf, route } from "./apiRouting"

// Every path here is built from `API_ROOT` rather than written out. These cases
// used to spell `/v1/`, which is why they went on passing when the API moved:
// they were checking the mock against the same stale literal the mock had, so
// the suite was self-consistent with the bug and the catalog 404d anyway.

describe("pathOf", () => {
  it("normalizes every shape fetch accepts to a path and query", () => {
    // The three cases used to be returned as they came, which is why the
    // gateway test below could be fooled: the same path is a relative string
    // from `apiFetch` and an absolute URL on a `Request`.
    expect(pathOf(`${API_ROOT}/organizations/me`)).toBe(`${API_ROOT}/organizations/me`)
    expect(pathOf(`${API_ROOT}/usage?window=7d`)).toBe(`${API_ROOT}/usage?window=7d`)
    expect(pathOf(new URL(`https://example.com${API_ROOT}/usage?window=7d`))).toBe(
      `${API_ROOT}/usage?window=7d`,
    )
    expect(pathOf(new Request(`https://example.com${API_ROOT}/organizations/me`))).toBe(
      `${API_ROOT}/organizations/me`,
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
    const table: ApiMocks = { [`${API_ROOT}/settings/mail`]: { configured: true } }
    expect(route(`${API_ROOT}/settings/mail`, [table])).toEqual({
      kind: "response",
      status: 200,
      body: { configured: true },
    })
  })

  it("reads the $status envelope as a failure rather than a body", () => {
    const table: ApiMocks = {
      [`${API_ROOT}/settings/mail`]: { $status: 503, $body: { detail: "No transport." } },
    }
    expect(route(`${API_ROOT}/settings/mail`, [table])).toEqual({
      kind: "response",
      status: 503,
      body: { detail: "No transport." },
    })
  })

  it("matches a declared pathname when the request carries a query", () => {
    const table: ApiMocks = { [`${API_ROOT}/usage`]: { total: 1 } }
    expect(route(`${API_ROOT}/usage?window=7d`, [table])).toEqual({
      kind: "response",
      status: 200,
      body: { total: 1 },
    })
  })

  it("answers what the decorators mount, with no table at all", () => {
    // The regression this exists for. Every story is wrapped in
    // `SelectedWorkspaceProvider`, which queries the organization, so a story
    // that declared nothing still asks. It used to reach the network and 404.
    const routed = route(`${API_ROOT}/organizations/me`, [])
    expect(routed.kind).toBe("response")
    if (routed.kind !== "response") return
    expect(routed.status).toBe(200)
    expect(routed.body).toMatchObject({ organization_member_id: expect.any(String) })
  })

  it("keys on the API root, so a path at the old prefix is not the gateway's", () => {
    // The case that would have caught #1026. The API moved to `/api/v1` while
    // this mock still matched `/v1/`, so every story's request missed it and
    // went to the network, where the static catalog answers 404. Both halves
    // matter: the current root is intercepted, and the old prefix is not, which
    // is what a second literal of the path would break again.
    expect(route(`${API_ROOT}/organizations/me`, []).kind).toBe("response")
    expect(route("/v1/organizations/me", [])).toEqual({ kind: "network" })
  })

  it("lets a story override the baseline", () => {
    const table: ApiMocks = {
      [`${API_ROOT}/organizations/me`]: { $status: 403, $body: { detail: "Not a member." } },
    }
    expect(route(`${API_ROOT}/organizations/me`, [table])).toEqual({
      kind: "response",
      status: 403,
      body: { detail: "Not a member." },
    })
  })

  it("answers an undeclared gateway path with 501, never the network", () => {
    // 501 rather than 404, because a 404 is a real gateway answer that some
    // components handle gracefully, which would hide the missing path. And
    // never `network`: that is what put a failing request on every story.
    const routed = route(`${API_ROOT}/does/not/exist`, [{ [`${API_ROOT}/other`]: {} }])
    expect(routed.kind).toBe("response")
    if (routed.kind !== "response") return
    expect(routed.status).toBe(501)
    expect(routed.body).toMatchObject({ detail: expect.stringContaining(`${API_ROOT}/does/not/exist`) })
  })

  it("takes the first table carrying the path when several are mounted", () => {
    // An autodocs page mounts several stories at once.
    const first: ApiMocks = { [`${API_ROOT}/usage`]: { total: 1 } }
    const second: ApiMocks = { [`${API_ROOT}/usage`]: { total: 2 } }
    expect(route(`${API_ROOT}/usage`, [first, second])).toEqual({
      kind: "response",
      status: 200,
      body: { total: 1 },
    })
  })
})
