import { render, screen } from "@testing-library/react"
import type { ReactElement, ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { AppShell } from "@/app/AppShell"
import { Provider } from "@/app/provider"
import { EntitlementGate } from "@/shared/components/EntitlementGate"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import type { Entitlements } from "@/shared/hooks/useEntitlements"
import {
  BASE_CAPABILITIES,
  EntitlementProvider,
  useEntitlements,
} from "@/shared/hooks/useEntitlements"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

// The capability under test. Overlay-only by ARCHITECTURE.md's table, so no
// build of this repository grants it and nothing else in the suite can be
// supplying it.
const OVERLAY_CAPABILITY = "billing"

// The seam, replaced the way a superset build's alias replaces it: a resolver
// that answers from somewhere this build cannot reach. Mocked by its `@/…`
// specifier, which is the resolution a superset build performs, and not by a
// relative path.
vi.mock("@/app/overlayEntitlementResolver", async () => {
  const { EntitlementProvider: Provide } = await import(
    "@/shared/hooks/useEntitlements"
  )
  return {
    EntitlementResolver: ({ children }: { children: ReactNode }) => (
      <Provide value={{ capabilities: ["billing"], isLoading: false }}>
        {children}
      </Provide>
    ),
  }
})

// The row such a build contributes, gated on the capability its resolver grants.
// Together the two mocks are the whole superset shape, which is what makes this
// the regression test for otari#758: before the seam existed the row was hidden
// in every build, the entitled one included.
vi.mock("@/app/nav/overlayNavItems", async () => {
  const { FiCreditCard } = await import("react-icons/fi")
  return {
    OVERLAY_NAV_ITEMS: [
      {
        sectionId: "access",
        items: [
          {
            // Any real route the registry does not already declare: what is
            // under test is the gate, not where the row points.
            to: "/aliases",
            label: "Billing",
            icon: FiCreditCard,
            capability: "billing",
          },
        ],
      },
    ],
  }
})

// The base default, reached past the mock above, because this file has to hold
// both halves of the seam: the module a replacement replaces, and the shell that
// mounts whichever of the two is present.
const { EntitlementResolver } = await vi.importActual<
  typeof import("@/app/overlayEntitlementResolver")
>("@/app/overlayEntitlementResolver")

/** Reports what the entitlement axis resolved to where it is rendered. */
function Probe() {
  const { capabilities, isLoading } = useEntitlements()
  return (
    <p>
      {`resolved:${capabilities.join(",") || "none"}:${isLoading ? "loading" : "settled"}`}
    </p>
  )
}

describe("the base entitlement resolver", () => {
  it("renders its children unchanged", () => {
    render(
      <EntitlementResolver>
        <p>CHILD</p>
      </EntitlementResolver>,
    )

    expect(screen.getByText("CHILD")).toBeInTheDocument()
  })

  it("mounts no provider, so consumers keep the base answer", () => {
    render(
      <EntitlementResolver>
        <Probe />
      </EntitlementResolver>,
    )

    // `BASE_CAPABILITIES` is empty today, which is what makes the "none" here
    // the base answer rather than a coincidence; the assertion is written
    // against the constant so it stays true if the base grows a capability.
    expect(screen.getByText(/^resolved:/)).toHaveTextContent(
      `resolved:${BASE_CAPABILITIES.join(",") || "none"}:settled`,
    )
  })

  it("does not shadow an answer supplied above it", () => {
    // Why the base default is a passthrough rather than an
    // `EntitlementProvider` carrying `BASE_CAPABILITIES`, which would look
    // equivalent: a provider shadows whatever is above it, so the base default
    // would overwrite a superset build's own answer, and every test that wraps
    // the shell in a provider of its own would silently get the empty base one.
    const supplied: Entitlements = {
      capabilities: [OVERLAY_CAPABILITY],
      isLoading: false,
    }
    render(
      <EntitlementProvider value={supplied}>
        <EntitlementResolver>
          <Probe />
        </EntitlementResolver>
      </EntitlementProvider>,
    )

    expect(screen.getByText(/^resolved:/)).toHaveTextContent(
      `resolved:${OVERLAY_CAPABILITY}:settled`,
    )
  })
})

describe("the shell's mount point", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  /**
   * The shell with a page inside it, as `__root.tsx` mounts it.
   *
   * No `EntitlementProvider` anywhere in the tree: the whole point is that the
   * only answer available is the one the seam resolves, so a provider here would
   * be the test supplying what it is meant to be observing.
   */
  async function renderShell(page: ReactElement) {
    // The shell reads the organization context for its switcher and for the way
    // into the organization rail.
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      Response.json(organizationContext()),
    )
    return renderWithRouter(page, {
      shell: (
        <Provider>
          <DeploymentProvider value={bootstrap()}>
            <SelectedWorkspaceProvider>
              <AppShell />
            </SelectedWorkspaceProvider>
          </DeploymentProvider>
        </Provider>
      ),
    })
  }

  it("puts the resolved answer above the navigation", async () => {
    await renderShell(<p>PAGE</p>)

    // The bug otari#758 describes: a contributed row gated on a capability was
    // resolved from `BASE_CAPABILITIES`, which grants no overlay capability, so
    // it was absent from every build including the entitled one.
    expect(
      await screen.findByRole("link", { name: "Billing" }),
    ).toBeInTheDocument()
  })

  it("puts it above the routes as well", async () => {
    // The other half a page gates on, and the reason the resolver wraps the
    // shell rather than only the sidebar: `EntitlementGate` is what a whole page
    // is wrapped in, and pages render through the shell's `<Outlet>`.
    await renderShell(
      <EntitlementGate
        capability={OVERLAY_CAPABILITY}
        fallback={<p>NOT AVAILABLE</p>}
      >
        <p>BILLING PAGE</p>
      </EntitlementGate>,
    )

    expect(await screen.findByText("BILLING PAGE")).toBeInTheDocument()
    expect(screen.queryByText("NOT AVAILABLE")).not.toBeInTheDocument()
  })
})
