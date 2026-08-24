import { act, renderHook } from "@testing-library/react"
import type { ReactNode } from "react"
import { describe, expect, it, vi } from "vitest"

import {
  DeploymentProvider,
  useDeployment,
  useOfferPasskeySignIn,
  useRetireMasterKeySignIn,
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

describe("the two corrections this app may make to sign_in_methods", () => {
  it("keeps a registered passkey when claiming retires the master key", () => {
    // The interaction worth pinning: both corrections touch the same field, and
    // replacing the list on a claim rather than swapping the one member would
    // make an operator who registered a passkey first watch its sign-in button
    // vanish the moment they set a password.
    const { result } = renderHook(
      () => ({
        deployment: useDeployment(),
        retire: useRetireMasterKeySignIn(),
      }),
      {
        wrapper: wrapper(
          bootstrap({ sign_in_methods: ["master_key", "passkey"] }),
        ),
      },
    )

    act(() => result.current.retire())

    expect(result.current.deployment.sign_in_methods).toEqual([
      "passkey",
      "password",
    ])
  })

  it("adds and removes passkey without disturbing the typed credential", () => {
    const { result } = renderHook(
      () => ({
        deployment: useDeployment(),
        offer: useOfferPasskeySignIn(),
      }),
      { wrapper: wrapper(bootstrap({ sign_in_methods: ["password"] })) },
    )

    // Registering the first one.
    act(() => result.current.offer(true))
    expect(result.current.deployment.sign_in_methods).toEqual([
      "passkey",
      "password",
    ])

    // Deleting the last one. Reversible, unlike the claim.
    act(() => result.current.offer(false))
    expect(result.current.deployment.sign_in_methods).toEqual(["password"])
  })

  it("sorts a corrected list, so it is indistinguishable from a fetched one", () => {
    const { result } = renderHook(
      () => ({
        deployment: useDeployment(),
        offer: useOfferPasskeySignIn(),
      }),
      { wrapper: wrapper(bootstrap({ sign_in_methods: ["master_key"] })) },
    )

    act(() => result.current.offer(true))
    // The gateway sorts this field, so a consumer comparing the array rather
    // than probing it with includes() must not be able to tell the difference.
    expect(result.current.deployment.sign_in_methods).toEqual([
      "master_key",
      "passkey",
    ])
  })
})
