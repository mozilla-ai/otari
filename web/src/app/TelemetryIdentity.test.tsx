import { render, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { Provider } from "@/app/provider"
import { TelemetryIdentity } from "@/app/TelemetryIdentity"
import type { DeploymentBootstrap } from "@/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import {
  identify,
  resetTelemetrySpy,
  setTelemetryConsent,
} from "@/tests/telemetry"

// By the `@/…` specifier, which is the one a superset build's alias replaces,
// so this exercises the resolution that build performs.
vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
  const { telemetrySpy } = await import("@/tests/telemetry")
  return { useTelemetry: () => telemetrySpy }
})

function renderIdentity(
  deployment: DeploymentBootstrap = bootstrap(),
  context: Parameters<typeof organizationContext>[0] = {},
) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
    Response.json(organizationContext(context)),
  )
  return render(
    <Provider>
      <DeploymentProvider value={deployment}>
        <TelemetryIdentity />
      </DeploymentProvider>
    </Provider>,
  )
}

describe("TelemetryIdentity", () => {
  beforeEach(() => {
    resetTelemetrySpy()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("names the actor and the organization together", async () => {
    setTelemetryConsent("granted")

    renderIdentity(bootstrap({ session_type: "local_operator" }), {
      role: "owner",
    })

    await waitFor(() => {
      expect(identify).toHaveBeenCalledWith({
        actorId: "22222222-2222-2222-2222-222222222222",
        sessionType: "local_operator",
        organizationId: "11111111-1111-1111-1111-111111111111",
        organizationName: "Default Organization",
        role: "owner",
      })
    })
  })

  it("withholds the identity while no consent decision is stored", async () => {
    // The base build's answer, and the reason it emits nothing: an undecided
    // browser is not a consenting one, so the one call that hands a tracker
    // something durable about a person is never made.
    setTelemetryConsent("unknown")

    renderIdentity()

    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalled()
    })
    expect(identify).not.toHaveBeenCalled()
  })

  it("withholds it on a refusal too", async () => {
    setTelemetryConsent("denied")

    renderIdentity()

    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalled()
    })
    expect(identify).not.toHaveBeenCalled()
  })

  it("sends nothing until the organization context resolves", async () => {
    setTelemetryConsent("granted")
    // A context that never answers: there is no actor to name yet, and naming
    // one without its organization is what this seam exists to avoid.
    vi.spyOn(globalThis, "fetch").mockImplementation(
      () => new Promise(() => undefined),
    )

    render(
      <Provider>
        <DeploymentProvider value={bootstrap()}>
          <TelemetryIdentity />
        </DeploymentProvider>
      </Provider>,
    )

    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalled()
    })
    expect(identify).not.toHaveBeenCalled()
  })

  it("identifies once, not on every render", async () => {
    setTelemetryConsent("granted")

    const { rerender } = renderIdentity()
    await waitFor(() => {
      expect(identify).toHaveBeenCalledTimes(1)
    })

    rerender(
      <Provider>
        <DeploymentProvider value={bootstrap()}>
          <TelemetryIdentity />
        </DeploymentProvider>
      </Provider>,
    )

    // The seam promises a stable tracker and this reads its fields out as
    // values rather than assembling an object in the render body; either half
    // going wrong turns this into an effect that re-identifies forever.
    expect(identify).toHaveBeenCalledTimes(1)
  })

  it("renders nothing", async () => {
    setTelemetryConsent("granted")

    const { container } = renderIdentity()

    await waitFor(() => {
      expect(identify).toHaveBeenCalled()
    })
    expect(container).toBeEmptyDOMElement()
  })
})
