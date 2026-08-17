import { renderHook } from "@testing-library/react"
import type { ReactNode } from "react"
import { describe, expect, it } from "vitest"

import {
  BASE_CAPABILITIES,
  EntitlementProvider,
  useEntitlement,
  useEntitlements,
  useFeatureFlag,
} from "@/shared/hooks/useEntitlements"

const withProvider =
  (capabilities: string[], flags: Record<string, boolean> = {}) =>
  ({ children }: { children: ReactNode }) => (
    <EntitlementProvider value={{ capabilities, flags, isLoading: false }}>
      {children}
    </EntitlementProvider>
  )

describe("useEntitlements without a provider", () => {
  it("answers from the base constant, which is the seam's fallback", () => {
    // The base build renders no EntitlementProvider at all, so the context
    // default is the code path a plain `npm run build` takes. Asserting against
    // the constant rather than a literal keeps this true when the base grows a
    // capability, instead of turning into the test that has to be edited.
    const { result } = renderHook(() => useEntitlements())
    expect(result.current.capabilities).toEqual(BASE_CAPABILITIES)
    expect(result.current.flags).toEqual({})
    // Never pending: the base answer is a constant, so a gate resolves on the
    // first render rather than flashing a loading state.
    expect(result.current.isLoading).toBe(false)
  })

  it("denies a capability the base build does not ship", () => {
    const { result } = renderHook(() => useEntitlement("billing"))
    expect(result.current.entitled).toBe(false)
  })

  it("reports every flag off", () => {
    const { result } = renderHook(() => useFeatureFlag("anything"))
    expect(result.current.enabled).toBe(false)
  })
})

describe("useEntitlements with a provider", () => {
  it("takes the provider's answer over the base constant", () => {
    // How an overlay supplies a real resolver: one component above the shell,
    // and nothing that reads a capability changes.
    const { result } = renderHook(() => useEntitlement("billing"), {
      wrapper: withProvider(["billing"]),
    })
    expect(result.current.entitled).toBe(true)
  })

  it("does not add the base constant to what the provider grants", () => {
    // The provider replaces the answer rather than extending it, so an overlay
    // that means to withhold something can. A union here would make the base
    // set impossible to revoke.
    const { result } = renderHook(() => useEntitlements(), {
      wrapper: withProvider(["billing"]),
    })
    expect(result.current.capabilities).toEqual(["billing"])
  })

  it("reads a flag the provider evaluated", () => {
    const { result } = renderHook(() => useFeatureFlag("rollout"), {
      wrapper: withProvider([], { rollout: true }),
    })
    expect(result.current.enabled).toBe(true)
  })

  it("treats a flag the provider did not evaluate as off", () => {
    const { result } = renderHook(() => useFeatureFlag("unknown"), {
      wrapper: withProvider([], { rollout: true }),
    })
    expect(result.current.enabled).toBe(false)
  })
})
