import { renderHook } from "@testing-library/react"
import type { ReactNode } from "react"
import { describe, expect, it } from "vitest"

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

function visibility(
  surfaces: string[],
  entitlements: Partial<Entitlements> = {},
) {
  const wrapper = ({ children }: { children: ReactNode }) => (
    <DeploymentProvider value={bootstrap({ surfaces })}>
      <EntitlementProvider
        value={{ capabilities: [], isLoading: false, ...entitlements }}
      >
        {children}
      </EntitlementProvider>
    </DeploymentProvider>
  )
  return renderHook(() => useNavVisibility(), { wrapper }).result.current
}

describe("useNavVisibility", () => {
  it("shows an entry that declares no gate at all", () => {
    expect(visibility([])(item({}))).toBe(true)
  })

  it("gates on the surface the deployment hosts", () => {
    const isVisible = visibility(["usage"])
    expect(isVisible(item({ surface: "usage" }))).toBe(true)
    expect(isVisible(item({ surface: "keys" }))).toBe(false)
  })

  it("gates on the capability the deployment is entitled to", () => {
    const isVisible = visibility([], { capabilities: ["routing"] })
    expect(isVisible(item({ capability: "routing" }))).toBe(true)
    expect(isVisible(item({ capability: "billing" }))).toBe(false)
  })

  it("composes the two axes as AND, so either one hides the entry", () => {
    // The point of keeping them separate: an entitlement does not stand in for
    // a surface, nor a surface for an entitlement.
    const all = { surface: "routing", capability: "routing" }
    expect(
      visibility(["routing"], { capabilities: ["routing"] })(item(all)),
    ).toBe(true)
    expect(visibility([], { capabilities: ["routing"] })(item(all))).toBe(false)
    expect(visibility(["routing"], { capabilities: [] })(item(all))).toBe(false)
  })
})
