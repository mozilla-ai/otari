import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { renderHook, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useOrganizationContext } from "@/shared/api/hooks"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import type { Entitlements } from "@/shared/hooks/useEntitlements"
import { EntitlementProvider } from "@/shared/hooks/useEntitlements"
import { bootstrap, organizationContext } from "@/tests/fixtures"

import type { NavItem } from "./types"
import { useNavVisibility } from "./useNavVisibility"

// The composition is what this covers, so the items are made up rather than
// taken from the registry: the base build gates nothing on a capability, and a
// base entry that did would be permanently hidden. An overlay contributes
// entries like these through `overlaySections.ts`.
const item = (gating: Partial<NavItem>): NavItem =>
  ({ to: "/", label: "Test", icon: null, ...gating }) as NavItem

// The caller axis is a field on the membership context, so it is stubbed at
// fetch and the hook is given a client. Not an operator in the cases about the
// other two axes: that axis gates eight rows, and a case wants the answer it is
// not about to be the quiet one.
//
// `gate` is for the case that needs the in-flight window to be a state it can
// assert in rather than a race it hopes to win: the read does not answer until
// the test resolves it, which is what makes "before the answer" and "after it"
// two separate observations of one render.
function mockCaller(deployment_operator: boolean, gate?: Promise<unknown>) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
    await gate
    return Response.json(organizationContext({ deployment_operator }))
  })
}

// The other thing that read can do, which is what separates the two values of
// the flag now that neither shows a row before the answer arrives.
function mockCallerUnavailable() {
  vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
    Response.json({ detail: "boom" }, { status: 500 }),
  )
}

// A gateway older than this bundle, which answers the context without the field
// at all. Written by deleting the key rather than by passing `undefined`, since
// the fixture fills a whole shape and `undefined` would not survive `Response
// .json`; the distinction matters because this is a *settled* read with no
// answer in it, which is neither the loading case nor the failed one.
function mockCallerWithoutTheField() {
  vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
    const { deployment_operator: _omitted, ...older } = organizationContext()
    return Response.json(older)
  })
}

function visibility(
  surfaces: string[],
  entitlements: Partial<Entitlements> = {},
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const wrapper = ({ children }: { children: ReactNode }) => (
    <DeploymentProvider value={bootstrap({ surfaces })}>
      <EntitlementProvider
        value={{ capabilities: [], isLoading: false, ...entitlements }}
      >
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      </EntitlementProvider>
    </DeploymentProvider>
  )
  // The predicate's own source is rendered beside it so a case can assert
  // *after* the answer settled rather than merely after the request went out. It
  // is load-bearing rather than convenience: a still-loading read and a settled
  // no hide the same rows, so a regression that admitted `false` would pass a
  // test that only waited for the fetch.
  return renderHook(
    () => ({
      isVisible: useNavVisibility(),
      context: useOrganizationContext(),
    }),
    { wrapper },
  )
}

function predicate(
  surfaces: string[],
  entitlements: Partial<Entitlements> = {},
) {
  return visibility(surfaces, entitlements).result.current.isVisible
}

/** Resolve once the membership read has landed, either way. */
async function settled(rendered: ReturnType<typeof visibility>) {
  await waitFor(() =>
    expect(rendered.result.current.context.isPending).toBe(false),
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("useNavVisibility", () => {
  it("shows an entry that declares no gate at all", () => {
    mockCaller(false)
    expect(predicate([])(item({}))).toBe(true)
  })

  it("gates on the surface the deployment hosts", () => {
    mockCaller(false)
    const isVisible = predicate(["usage"])
    expect(isVisible(item({ surface: "usage" }))).toBe(true)
    expect(isVisible(item({ surface: "keys" }))).toBe(false)
  })

  it("gates on the capability the deployment is entitled to", () => {
    mockCaller(false)
    const isVisible = predicate([], { capabilities: ["routing"] })
    expect(isVisible(item({ capability: "routing" }))).toBe(true)
    expect(isVisible(item({ capability: "billing" }))).toBe(false)
  })

  it("composes the axes as AND, so any one of them hides the entry", () => {
    // The point of keeping them separate: an entitlement does not stand in for
    // a surface, nor a surface for an entitlement.
    mockCaller(false)
    const all = { surface: "routing", capability: "routing" }
    expect(
      predicate(["routing"], { capabilities: ["routing"] })(item(all)),
    ).toBe(true)
    expect(predicate([], { capabilities: ["routing"] })(item(all))).toBe(false)
    expect(predicate(["routing"], { capabilities: [] })(item(all))).toBe(false)
  })

  it("hides both kinds of operator-only entry while the answer is in flight", async () => {
    // The bug this shape fixes (#836): neither value may paint a row on the
    // strength of an answer still coming, because taking it back afterwards
    // tells a member a destination was theirs and then that it was not. Held
    // open by the gate, so this is the in-flight window itself and not the race
    // to observe it.
    let answer = (_: unknown) => {}
    mockCaller(true, new Promise((resolve) => (answer = resolve)))
    const inFlight = visibility(["admin"])
    // Named rather than assumed, so the window this asserts in is the loading
    // one and the assertions below cannot be read as a settled answer.
    expect(inFlight.result.current.context.isPending).toBe(true)
    for (const value of ["unlisted", "refused"] as const) {
      expect(
        inFlight.result.current.isVisible(item({ operatorOnly: value })),
      ).toBe(false)
    }
    // And it gates only the rows that declare it: the rest of the rail is drawn
    // now rather than waiting on a question they do not ask.
    expect(inFlight.result.current.isVisible(item({}))).toBe(true)

    // Then the yes lands and both appear. A row that appears late never told
    // anyone it was missing, which is the trade #836 settled.
    answer(undefined)
    await settled(inFlight)
    for (const value of ["unlisted", "refused"] as const) {
      expect(
        inFlight.result.current.isVisible(item({ operatorOnly: value })),
      ).toBe(true)
    }
  })

  it("keeps both kinds hidden once the caller is known not to be an operator", async () => {
    // A settled no reads the same as no answer yet, which is the whole reason
    // the failed-read case below is a separate test: that one is where the two
    // values come apart, and this one is where they agree.
    mockCaller(false)
    const refused = visibility(["admin"])
    // Waiting on the *query* and not on the fetch: a still-loading read hides
    // these rows too, so a regression that let `false` through would pass a
    // test that only waited for the request to go out.
    await settled(refused)
    expect(refused.result.current.context.data?.deployment_operator).toBe(false)
    for (const value of ["unlisted", "refused"] as const) {
      expect(
        refused.result.current.isVisible(item({ operatorOnly: value })),
      ).toBe(false)
    }
  })

  it("splits the two values on what a failed read means, not on when it shows", async () => {
    // The whole of the difference between them, and the reason the flag carries
    // a value at all. An "unlisted" destination is one the server 404s, so with
    // no answer the rail may not admit it exists either; a "refused" one is
    // 403ed, its existence is no secret, and hiding seven deployment-wide
    // destinations because one query failed strands them, which is the same
    // direction `AppShell` fails `managesOrganization` open in.
    mockCallerUnavailable()
    const unknown = visibility(["admin"])
    await settled(unknown)
    expect(unknown.result.current.context.isError).toBe(true)
    expect(
      unknown.result.current.isVisible(item({ operatorOnly: "refused" })),
    ).toBe(true)
    expect(
      unknown.result.current.isVisible(item({ operatorOnly: "unlisted" })),
    ).toBe(false)
  })

  it("hides both kinds from a gateway whose context omits the field", async () => {
    // A gateway older than this bundle answers the context without
    // `deployment_operator`, so the read settles with no answer in it: not
    // loading, not an error, and not a no. `isDeploymentOperator` requires an
    // explicit `true`, so both kinds stay hidden, which is the safe direction and
    // the one a real operator pays for by upgrading rather than by being shown a
    // row the server will refuse.
    mockCallerWithoutTheField()
    const older = visibility(["admin"])
    await settled(older)
    expect(older.result.current.context.isSuccess).toBe(true)
    expect(
      older.result.current.context.data?.deployment_operator,
    ).toBeUndefined()
    for (const value of ["unlisted", "refused"] as const) {
      expect(
        older.result.current.isVisible(item({ operatorOnly: value })),
      ).toBe(false)
    }
  })

  it("asks the caller axis without a second request of its own", async () => {
    // It rides on the membership context the shell reads anyway, so there is no
    // `/v1/admin/access` from the rail to gate on a surface: that request had to
    // be withheld from a gateway not hosting `/v1/admin` so its 404 would not
    // become a second reading of `surfaces`, and this read is not that request.
    // The row's own `surface` gate is still what decides it on such a gateway.
    mockCaller(true)
    const fetchSpy = vi.mocked(globalThis.fetch)
    const hidden = visibility([])
    await settled(hidden)
    expect(
      hidden.result.current.isVisible(item({ operatorOnly: "unlisted" })),
    ).toBe(true)
    expect(
      fetchSpy.mock.calls.some((call) =>
        String(call[0]).includes("/v1/admin/access"),
      ),
    ).toBe(false)
  })
})
