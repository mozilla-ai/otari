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

// Through `vi.hoisted`, because a `vi.mock` factory is hoisted above every
// ordinary declaration in the file and would see a plain `const` in its temporal
// dead zone. That is also what lets a test vary the answer: the factory reads
// `.value` when it renders rather than closing over what it was at import.
const { OVERLAY_CAPABILITY, ENTITLED, resolved } = vi.hoisted(() => {
  // Overlay-only by ARCHITECTURE.md's table, so no build of this repository
  // grants it and nothing else in the suite can be supplying it.
  const capability = "billing"
  const entitled = { capabilities: [capability], isLoading: false }
  return {
    OVERLAY_CAPABILITY: capability,
    ENTITLED: entitled,
    resolved: {
      value: entitled as { capabilities: string[]; isLoading: boolean },
    },
  }
})

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
      <Provide value={resolved.value}>{children}</Provide>
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
            capability: OVERLAY_CAPABILITY,
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
    resolved.value = ENTITLED
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
  async function renderShell(page: ReactElement, url?: string) {
    // The shell reads the organization context for its switcher and for the way
    // into the organization rail.
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      Response.json(organizationContext()),
    )
    return renderWithRouter(page, {
      url,
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

  it("does not claim a route is unserved while the answer is resolving", async () => {
    // The base resolves synchronously, so this is the branch only a superset
    // build's asynchronous resolver reaches, and it is the one place the axis
    // going async is visible to a person rather than only to a rail: the route
    // gate is a predicate with no `loading` prop to pass, unlike
    // `EntitlementGate`, so before this it answered a still-unknown entitlement
    // by asserting the deployment does not serve the page. An entitled visitor
    // deep-linking to an overlay page was told it did not exist here, and then
    // shown it.
    resolved.value = { capabilities: [], isLoading: true }
    await renderShell(<p>BILLING PAGE</p>, "/aliases")

    expect(screen.queryByText(/is not available here/)).not.toBeInTheDocument()
    expect(await screen.findByRole("status")).toBeInTheDocument()
  })

  it("still answers a surface-gated route while the answer is resolving", async () => {
    // The other axis, and it is not waiting on anything: the bootstrap settled
    // the surfaces before the page rendered, so a destination this deployment
    // does not host is answerable now and no entitlement query can change it.
    // Waiting on one would hold back a panel that is already correct.
    // `/organization/provider-keys` is gated on `organization_providers`, which
    // `STANDALONE_SURFACES` does not report, so the base registry gates it off
    // in this build with no overlay contribution involved.
    resolved.value = { capabilities: [], isLoading: true }
    await renderShell(<p>PROVIDER KEYS</p>, "/organization/provider-keys")

    expect(await screen.findByText(/is not available here/)).toBeInTheDocument()
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })
})
