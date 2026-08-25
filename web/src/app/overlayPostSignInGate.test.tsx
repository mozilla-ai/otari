import { render, screen } from "@testing-library/react"
import type { ReactElement, ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { AppShell } from "@/app/AppShell"
import { Provider } from "@/app/provider"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { useEntitlements } from "@/shared/hooks/useEntitlements"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

// Through `vi.hoisted`, because a `vi.mock` factory is hoisted above every
// ordinary declaration in the file and would see a plain `const` in its temporal
// dead zone. That is also what lets each test choose what the replacement does:
// the factory reads `.current` when it renders rather than closing over what it
// was at import.
const { OVERLAY_CAPABILITY, capabilities, replacement } = vi.hoisted(() => ({
  // Overlay-only by ARCHITECTURE.md's table, so no build of this repository
  // grants it and nothing else in the suite can be supplying it.
  OVERLAY_CAPABILITY: "billing",
  // What the mocked resolver answers, and empty unless a test says otherwise:
  // the mock below is file-wide, so a granted capability would be granted to the
  // test that pins the OSS baseline too. Inert today, since no base nav entry
  // gates on a capability, and that is the point at which it would stop being.
  capabilities: { current: [] as string[] },
  replacement: {
    current: null as ((children: ReactNode) => ReactNode) | null,
  },
}))

// The seam, replaced the way a superset build's alias replaces it. Mocked by its
// `@/…` specifier, which is the resolution a superset build performs, and not by
// a relative path.
vi.mock("@/app/overlayPostSignInGate", () => ({
  PostSignInGate: ({ children }: { children: ReactNode }) =>
    replacement.current ? replacement.current(children) : children,
}))

// The other half of the position under test: a resolver granting a capability
// this build never grants. `AppShell` mounts the gate inside this, so a
// replacement that reads the axis gets the resolved answer rather than the
// context default, which is what lets it gate its step on a capability.
vi.mock("@/app/overlayEntitlementResolver", async () => {
  const { EntitlementProvider } = await import("@/shared/hooks/useEntitlements")
  return {
    EntitlementResolver: ({ children }: { children: ReactNode }) => (
      <EntitlementProvider
        value={{ capabilities: capabilities.current, isLoading: false }}
      >
        {children}
      </EntitlementProvider>
    ),
  }
})

// The base default, reached past the mock above, because this file has to hold
// both halves of the seam: the module a replacement replaces, and the shell that
// mounts whichever of the two is present.
const { PostSignInGate } = await vi.importActual<
  typeof import("@/app/overlayPostSignInGate")
>("@/app/overlayPostSignInGate")

describe("the base post-sign-in gate", () => {
  it("renders its children unchanged", () => {
    render(
      <PostSignInGate>
        <p>CHILD</p>
      </PostSignInGate>,
    )

    expect(screen.getByText("CHILD")).toBeInTheDocument()
  })
})

describe("the shell's mount point", () => {
  afterEach(() => {
    replacement.current = null
    capabilities.current = []
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  /** The shell with a page inside it, as `__root.tsx` mounts it. */
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

  it("leaves the shell as it was when the gate passes its children through", async () => {
    // The base default, and the state every OSS deployment is in: no step, the
    // rails and the page exactly as they were before the seam existed.
    await renderShell(<p>PAGE</p>)

    expect(await screen.findByText("PAGE")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Overview" })).toBeInTheDocument()
  })

  it("is above the chrome, so a step replaces the rails rather than covering them", async () => {
    // The reason the seam is mounted where it is: a replacement rendering a step
    // instead of its children takes the whole app, so the person answering the
    // questions is not looking at a sidebar they cannot use or a page behind it.
    replacement.current = () => <p>STEP</p>
    await renderShell(<p>PAGE</p>)

    expect(await screen.findByText("STEP")).toBeInTheDocument()
    expect(
      screen.queryByRole("link", { name: "Overview" }),
    ).not.toBeInTheDocument()
    expect(screen.queryByText("PAGE")).not.toBeInTheDocument()
  })

  it("is inside the entitlement resolver, so a step may gate on a capability", async () => {
    // The other boundary, and it is not observable from the base build alone:
    // both halves are mocked here because both are the superset build's, and
    // together they are what a replacement deciding whether to ask anything at
    // all can rely on.
    function CapabilityProbe() {
      const { capabilities } = useEntitlements()
      return <p>{`gate sees:${capabilities.join(",") || "none"}`}</p>
    }
    capabilities.current = [OVERLAY_CAPABILITY]
    replacement.current = () => <CapabilityProbe />
    await renderShell(<p>PAGE</p>)

    expect(await screen.findByText(/^gate sees:/)).toHaveTextContent(
      `gate sees:${OVERLAY_CAPABILITY}`,
    )
  })

  it("still renders the shell when the gate renders children and a step at once", async () => {
    // The contract a replacement owes while it is deciding: children keep
    // rendering, so the dashboard does not blank on every load for the people
    // who answered months ago. Asserted through a replacement that does it, since
    // the base has nothing to decide.
    replacement.current = (children) => (
      <>
        <p>ASKING</p>
        {children}
      </>
    )
    await renderShell(<p>PAGE</p>)

    expect(await screen.findByText("ASKING")).toBeInTheDocument()
    expect(screen.getByText("PAGE")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Overview" })).toBeInTheDocument()
  })
})
