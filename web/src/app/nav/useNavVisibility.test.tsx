import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { renderHook, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import type { Entitlements } from "@/shared/hooks/useEntitlements"
import { EntitlementProvider } from "@/shared/hooks/useEntitlements"
import { bootstrap } from "@/tests/fixtures"

import type { NavItem } from "./types"
import { useNavVisibility } from "./useNavVisibility"

// The composition is what this covers, so the items are made up rather than
// taken from the registry: the base build gates nothing on a capability, and a
// base entry that did would be permanently hidden. An overlay contributes
// entries like these through `overlaySections.ts`.
const item = (gating: Partial<NavItem>): NavItem =>
  ({ to: "/", label: "Test", icon: null, ...gating }) as NavItem

// The third axis is a query rather than a provider, so it is stubbed at fetch
// and the hook is given a client. `false` by default: the operator axis gates
// exactly one row, and every case below is about the other two. The query only
// runs on a deployment that hosts the surface serving it, so a case about this
// axis passes `admin` in `surfaces` as well as stubbing the answer.
function mockOperator(granted: boolean) {
  vi.spyOn(globalThis, "fetch").mockImplementation(
    async () =>
      new Response(JSON.stringify({ granted }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  )
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
  return renderHook(() => useNavVisibility(), { wrapper })
}

function predicate(
  surfaces: string[],
  entitlements: Partial<Entitlements> = {},
) {
  return visibility(surfaces, entitlements).result.current
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("useNavVisibility", () => {
  it("shows an entry that declares no gate at all", () => {
    mockOperator(false)
    expect(predicate([])(item({}))).toBe(true)
  })

  it("gates on the surface the deployment hosts", () => {
    mockOperator(false)
    const isVisible = predicate(["usage"])
    expect(isVisible(item({ surface: "usage" }))).toBe(true)
    expect(isVisible(item({ surface: "keys" }))).toBe(false)
  })

  it("gates on the capability the deployment is entitled to", () => {
    mockOperator(false)
    const isVisible = predicate([], { capabilities: ["routing"] })
    expect(isVisible(item({ capability: "routing" }))).toBe(true)
    expect(isVisible(item({ capability: "billing" }))).toBe(false)
  })

  it("composes the axes as AND, so any one of them hides the entry", () => {
    // The point of keeping them separate: an entitlement does not stand in for
    // a surface, nor a surface for an entitlement.
    mockOperator(false)
    const all = { surface: "routing", capability: "routing" }
    expect(
      predicate(["routing"], { capabilities: ["routing"] })(item(all)),
    ).toBe(true)
    expect(predicate([], { capabilities: ["routing"] })(item(all))).toBe(false)
    expect(predicate(["routing"], { capabilities: [] })(item(all))).toBe(false)
  })

  it("hides an operator-only entry until the caller is known to be one", async () => {
    // Absent while the answer is still coming, which is the safe direction: the
    // page behind it renders its own refusal, and a row that appears and then
    // disappears reads as a bug.
    mockOperator(false)
    const refused = visibility(["admin"])
    expect(refused.result.current(item({ operatorOnly: true }))).toBe(false)
    await waitFor(() =>
      expect(refused.result.current(item({ operatorOnly: true }))).toBe(false),
    )

    mockOperator(true)
    const allowed = visibility(["admin"])
    await waitFor(() =>
      expect(allowed.result.current(item({ operatorOnly: true }))).toBe(true),
    )
    // And it gates only the rows that declare it.
    expect(refused.result.current(item({}))).toBe(true)
  })

  it("never asks the caller axis on a deployment without the surface", async () => {
    // The deployment axis answers first and settles it: no surface, no operator
    // question, and in particular no request whose 404 would be a second way to
    // learn what `surfaces` already said.
    mockOperator(true)
    const fetchSpy = vi.mocked(globalThis.fetch)
    const hidden = visibility([])
    await waitFor(() =>
      expect(hidden.result.current(item({ operatorOnly: true }))).toBe(false),
    )
    expect(
      fetchSpy.mock.calls.some((call) =>
        String(call[0]).includes("/v1/admin/access"),
      ),
    ).toBe(false)
  })
})
