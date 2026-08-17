import { renderHook } from "@testing-library/react"
import type { ReactNode } from "react"
import { describe, expect, it, vi } from "vitest"

import {
  DeploymentProvider,
  useDeployment,
  useSurfaces,
} from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"

function wrapper(value = bootstrap()) {
  return ({ children }: { children: ReactNode }) => (
    <DeploymentProvider value={value}>{children}</DeploymentProvider>
  )
}

describe("useDeployment", () => {
  it("hands back the bootstrap the page was served with", () => {
    const served = bootstrap({ deployment_type: "hybrid" })
    const { result } = renderHook(() => useDeployment(), {
      wrapper: wrapper(served),
    })

    expect(result.current).toEqual(served)
  })

  it("throws outside a provider rather than guessing a deployment", () => {
    // A tree with no bootstrap has not been told what it is running against.
    // Defaulting to standalone here would put the failure somewhere else.
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})

    expect(() => renderHook(() => useDeployment())).toThrow(
      /within a DeploymentProvider/,
    )

    consoleError.mockRestore()
  })
})

describe("useSurfaces", () => {
  it("reports only what the deployment hosts", () => {
    const { result } = renderHook(() => useSurfaces(), {
      wrapper: wrapper(bootstrap({ surfaces: ["usage", "models"] })),
    })

    expect(result.current("usage")).toBe(true)
    expect(result.current("models")).toBe(true)
    expect(result.current("budgets")).toBe(false)
  })

  it("hosts nothing on a hybrid gateway", () => {
    const { result } = renderHook(() => useSurfaces(), {
      wrapper: wrapper(bootstrap({ deployment_type: "hybrid", surfaces: [] })),
    })

    expect(result.current("usage")).toBe(false)
  })
})
