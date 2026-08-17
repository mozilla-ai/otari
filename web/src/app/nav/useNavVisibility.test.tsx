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
// taken from the registry: the base build gates nothing on a flag, and a base
// entry that did would be permanently hidden. An overlay contributes entries
// like these through `overlaySections.ts`.
const item = (gating: Partial<NavItem>): NavItem =>
  ({ to: "/", label: "Test", icon: null, ...gating }) as NavItem

function visibility(
  surfaces: string[],
  entitlements: Partial<Entitlements> = {},
) {
  const wrapper = ({ children }: { children: ReactNode }) => (
    <DeploymentProvider value={bootstrap({ surfaces })}>
      <EntitlementProvider
        value={{
          capabilities: [],
          flags: {},
          isLoading: false,
          ...entitlements,
        }}
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

  it("gates on the feature flag beneath the capability", () => {
    const isVisible = visibility([], {
      capabilities: ["routing"],
      flags: { "smart-selection": true, retired: false },
    })
    expect(
      isVisible(item({ capability: "routing", flag: "smart-selection" })),
    ).toBe(true)
    expect(isVisible(item({ capability: "routing", flag: "retired" }))).toBe(
      false,
    )
    // An unknown key is off, so a flag removed from the resolver hides its
    // surface rather than silently opening it.
    expect(isVisible(item({ capability: "routing", flag: "unknown" }))).toBe(
      false,
    )
  })

  it("composes the three axes as AND, so any one of them hides the entry", () => {
    // The point of keeping them separate: an entitlement does not stand in for
    // a surface, and a flag does not stand in for an entitlement.
    const all = { surface: "routing", capability: "routing", flag: "v2" }
    expect(
      visibility(["routing"], {
        capabilities: ["routing"],
        flags: { v2: true },
      })(item(all)),
    ).toBe(true)
    expect(
      visibility([], { capabilities: ["routing"], flags: { v2: true } })(
        item(all),
      ),
    ).toBe(false)
    expect(
      visibility(["routing"], { capabilities: [], flags: { v2: true } })(
        item(all),
      ),
    ).toBe(false)
    expect(
      visibility(["routing"], {
        capabilities: ["routing"],
        flags: { v2: false },
      })(item(all)),
    ).toBe(false)
  })
})
