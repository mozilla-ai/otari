import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { EntitlementGate } from "@/shared/components/EntitlementGate"
import type { Entitlements } from "@/shared/hooks/useEntitlements"
import {
  BASE_CAPABILITIES,
  EntitlementProvider,
} from "@/shared/hooks/useEntitlements"

function renderGate(
  gate: React.ReactElement,
  entitlements?: Partial<Entitlements>,
) {
  if (!entitlements) {
    // No provider: the context default is the base build's answer, which is the
    // path a plain `npm run build` actually takes.
    return render(gate)
  }
  return render(
    <EntitlementProvider
      value={{
        capabilities: [],
        flags: {},
        isLoading: false,
        ...entitlements,
      }}
    >
      {gate}
    </EntitlementProvider>,
  )
}

describe("EntitlementGate", () => {
  it("renders an entitled capability", () => {
    renderGate(
      <EntitlementGate capability="routing">
        <p>GATED</p>
      </EntitlementGate>,
      { capabilities: ["routing"] },
    )
    expect(screen.getByText("GATED")).toBeInTheDocument()
  })

  it("hides an unentitled capability, rendering nothing by default", () => {
    const { container } = renderGate(
      <EntitlementGate capability="billing">
        <p>GATED</p>
      </EntitlementGate>,
      { capabilities: ["routing"] },
    )
    expect(screen.queryByText("GATED")).toBeNull()
    // Hides rather than substituting: a gate with no fallback leaves no trace of
    // the surface it withheld.
    expect(container).toBeEmptyDOMElement()
  })

  it("renders the fallback in place of an unentitled capability", () => {
    renderGate(
      <EntitlementGate capability="billing" fallback={<p>NOT AVAILABLE</p>}>
        <p>GATED</p>
      </EntitlementGate>,
      { capabilities: [] },
    )
    expect(screen.getByText("NOT AVAILABLE")).toBeInTheDocument()
    expect(screen.queryByText("GATED")).toBeNull()
  })

  it("requires the flag as well as the entitlement", () => {
    // The two axes compose as AND and are never merged: holding the capability
    // is not the same as having its rollout turned on.
    renderGate(
      <EntitlementGate capability="routing" flag="smart-selection">
        <p>GATED</p>
      </EntitlementGate>,
      { capabilities: ["routing"], flags: { "smart-selection": false } },
    )
    expect(screen.queryByText("GATED")).toBeNull()
  })

  it("renders once both the entitlement and the flag are satisfied", () => {
    renderGate(
      <EntitlementGate capability="routing" flag="smart-selection">
        <p>GATED</p>
      </EntitlementGate>,
      { capabilities: ["routing"], flags: { "smart-selection": true } },
    )
    expect(screen.getByText("GATED")).toBeInTheDocument()
  })

  it("keeps a flag from standing in for an entitlement", () => {
    renderGate(
      <EntitlementGate capability="billing" flag="smart-selection">
        <p>GATED</p>
      </EntitlementGate>,
      { capabilities: [], flags: { "smart-selection": true } },
    )
    expect(screen.queryByText("GATED")).toBeNull()
  })

  it("treats an unknown flag as off", () => {
    renderGate(
      <EntitlementGate capability="routing" flag="never-declared">
        <p>GATED</p>
      </EntitlementGate>,
      { capabilities: ["routing"] },
    )
    expect(screen.queryByText("GATED")).toBeNull()
  })

  it("shows the loading state instead of a false negative while resolving", () => {
    // The base build never reaches this, but an overlay resolving from the
    // server does, and an entitled user must not be told "not available" during
    // a cold load.
    renderGate(
      <EntitlementGate
        capability="routing"
        fallback={<p>NOT AVAILABLE</p>}
        loading={<p>CHECKING</p>}
      >
        <p>GATED</p>
      </EntitlementGate>,
      { capabilities: ["routing"], isLoading: true },
    )
    expect(screen.getByText("CHECKING")).toBeInTheDocument()
    expect(screen.queryByText("NOT AVAILABLE")).toBeNull()
    expect(screen.queryByText("GATED")).toBeNull()
  })

  it("falls back to the fallback while resolving when given no loading state", () => {
    renderGate(
      <EntitlementGate capability="routing" fallback={<p>NOT AVAILABLE</p>}>
        <p>GATED</p>
      </EntitlementGate>,
      { capabilities: ["routing"], isLoading: true },
    )
    expect(screen.getByText("NOT AVAILABLE")).toBeInTheDocument()
  })

  it("answers from the base constant with no provider above it", () => {
    // The default context value is the seam's fallback, so the base build wires
    // nothing up and still gets an answer. `BASE_CAPABILITIES` is empty today,
    // which makes the first half of this vacuous and the second half the real
    // assertion; both survive a name being added to the constant later.
    for (const capability of BASE_CAPABILITIES) {
      const { unmount } = renderGate(
        <EntitlementGate capability={capability}>
          <p>GATED</p>
        </EntitlementGate>,
      )
      expect(screen.getByText("GATED")).toBeInTheDocument()
      unmount()
    }

    renderGate(
      <EntitlementGate capability="billing">
        <p>GATED</p>
      </EntitlementGate>,
    )
    // Absent from the base constant, so the base build hides it. This is the
    // fail-closed default: an unknown capability is not entitled.
    expect(screen.queryByText("GATED")).toBeNull()
  })
})
