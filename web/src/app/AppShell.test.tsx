import { act, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { AppShell } from "@/app/AppShell"
import { Provider } from "@/app/provider"
import type { DeploymentBootstrap } from "@/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import type { Entitlements } from "@/shared/hooks/useEntitlements"
import {
  BASE_CAPABILITIES,
  EntitlementProvider,
} from "@/shared/hooks/useEntitlements"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

// jsdom has no layout engine, so `md:hidden` / responsive classes never take
// effect. The mobile-vs-desktop branch keys off window.matchMedia instead, which
// jsdom also does not implement, so tests drive it through this stub. The stub
// captures listeners so a viewport change can be simulated.
function mockMatchMedia(matches: boolean, options: { legacy?: boolean } = {}) {
  const listeners = new Set<(event: MediaQueryListEvent) => void>()
  const mql: Record<string, unknown> = {
    matches,
    media: "",
    onchange: null,
    // Deprecated Safari < 14 API; always present.
    addListener: (cb: (event: MediaQueryListEvent) => void) =>
      listeners.add(cb),
    removeListener: (cb: (event: MediaQueryListEvent) => void) =>
      listeners.delete(cb),
    dispatchEvent: () => true,
  }
  // `legacy` omits the modern API so the component must fall back to addListener.
  if (!options.legacy) {
    mql.addEventListener = (
      _type: string,
      cb: (event: MediaQueryListEvent) => void,
    ) => listeners.add(cb)
    mql.removeEventListener = (
      _type: string,
      cb: (event: MediaQueryListEvent) => void,
    ) => listeners.delete(cb)
  }
  vi.stubGlobal("matchMedia", vi.fn().mockReturnValue(mql))
  return { mql, listeners }
}

// AppShell is the root route's component in the real tree, so it renders its
// page through an <Outlet>. Here it is the component under test and the pages
// are the router's children, which is what makes clicking a nav link swap them.
function renderShell(
  deployment: DeploymentBootstrap = bootstrap(),
  options: { entitlements?: Partial<Entitlements>; url?: string } = {},
) {
  const entitlements: Entitlements = {
    capabilities: BASE_CAPABILITIES,
    flags: {},
    isLoading: false,
    ...options.entitlements,
  }
  const url = options.url ?? "/"
  // The shell reads the organization context to decide whether to offer the way
  // into that rail, and the switcher reads it for the names it shows. Stubbed
  // here so the sidebar behaves as it does in front of a real gateway.
  vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
    Response.json(organizationContext()),
  )
  return renderWithRouter(<div>PAGE CONTENT</div>, {
    url,
    shell: (
      <Provider>
        <DeploymentProvider value={deployment}>
          <EntitlementProvider value={entitlements}>
            {/* The shell reads the selected workspace for its switcher and its
                way back out of the organization rail. */}
            <SelectedWorkspaceProvider>
              <AppShell />
            </SelectedWorkspaceProvider>
          </EntitlementProvider>
        </DeploymentProvider>
      </Provider>
    ),
    // The harness already mounts the component under test at `url`, so a probe
    // for that same path would be a duplicate route.
    routes: [{ path: "/providers", element: <div>PROVIDERS PAGE</div> }].filter(
      (route) => route.path !== url,
    ),
  })
}

describe("AppShell responsive layout", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    window.localStorage.clear()
  })

  it("keeps the sidebar an off-canvas drawer on mobile, toggled from the header", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    const { container } = await renderShell()

    const aside = container.querySelector("aside")
    // Off-canvas by default so it does not squash the page's content.
    expect(aside?.className).toContain("fixed")
    expect(aside?.className).toContain("-translate-x-full")

    const toggle = screen.getByRole("button", { name: "Open navigation" })
    expect(toggle).toHaveAttribute("aria-expanded", "false")

    await user.click(toggle)

    expect(
      screen.getByRole("button", { name: "Close navigation" }),
    ).toHaveAttribute("aria-expanded", "true")
    expect(aside?.className).toContain("translate-x-0")
  })

  it("dismisses the mobile drawer after navigating to a destination", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell()

    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    await user.click(screen.getByRole("link", { name: "Provider credentials" }))

    expect(await screen.findByText("PROVIDERS PAGE")).toBeInTheDocument()
    // Navigating closes the drawer so the page it landed on is not hidden behind it.
    expect(
      screen.getByRole("button", { name: "Open navigation" }),
    ).toHaveAttribute("aria-expanded", "false")
  })

  it("marks only the current page's nav link as active", async () => {
    // Worth pinning because the router decides this, not the shell: a link is
    // active when its route is in the current match chain, so "/" lights only on
    // the index rather than on every page under it. Nothing in the shell opts
    // into that, which is precisely why a well-meant `activeOptions` could take
    // it away without looking wrong.
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell()

    const overview = screen.getByRole("link", { name: "Overview" })
    const providers = screen.getByRole("link", { name: "Provider credentials" })
    expect(overview).toHaveAttribute("aria-current", "page")
    expect(providers).not.toHaveAttribute("aria-current")

    await user.click(providers)

    expect(await screen.findByText("PROVIDERS PAGE")).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Provider credentials" }),
    ).toHaveAttribute("aria-current", "page")
    expect(screen.getByRole("link", { name: "Overview" })).not.toHaveAttribute(
      "aria-current",
    )
  })

  it("renders the resizable rail (not a drawer) on desktop", async () => {
    mockMatchMedia(false)
    const { container } = await renderShell()

    const aside = container.querySelector("aside")
    // Desktop keeps the in-flow, inline-width-driven rail rather than a fixed overlay.
    expect(aside?.className).not.toContain("fixed")
    expect(aside?.getAttribute("style")).toContain("width")
  })

  it("closes the drawer when the viewport grows past the mobile breakpoint", async () => {
    const { listeners } = mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell()

    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    expect(
      screen.getByRole("button", { name: "Close navigation" }),
    ).toBeInTheDocument()

    // Simulate crossing to a desktop viewport.
    act(() => {
      listeners.forEach((cb) => {
        cb({ matches: false } as MediaQueryListEvent)
      })
    })

    expect(
      screen.getByRole("button", { name: "Open navigation" }),
    ).toHaveAttribute("aria-expanded", "false")
  })

  it("closes the drawer on Escape and restores focus to the toggle", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell()

    const toggle = screen.getByRole("button", { name: "Open navigation" })
    await user.click(toggle)
    expect(
      screen.getByRole("button", { name: "Close navigation" }),
    ).toHaveAttribute("aria-expanded", "true")

    await user.keyboard("{Escape}")

    expect(
      screen.getByRole("button", { name: "Open navigation" }),
    ).toHaveAttribute("aria-expanded", "false")
    // Focus returns to the trigger so a keyboard user is not dropped to the top.
    expect(
      screen.getByRole("button", { name: "Open navigation" }),
    ).toHaveFocus()
  })

  it("closes the drawer when the backdrop is clicked", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    const { container } = await renderShell()

    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    const backdrop = container.querySelector(".fixed.inset-0")!
    expect(backdrop).toBeInTheDocument()

    await user.click(backdrop)

    expect(
      screen.getByRole("button", { name: "Open navigation" }),
    ).toHaveAttribute("aria-expanded", "false")
  })

  it("marks the drawer inert while closed so its links leave the tab order", async () => {
    mockMatchMedia(true)
    const { container } = await renderShell()

    // Off-canvas and inert by default: the nav is not reachable until opened.
    expect(container.querySelector("aside")).toHaveAttribute("inert")
    expect(container.querySelector("aside")).toHaveAttribute(
      "aria-label",
      "Navigation",
    )
  })

  it("makes the background (header + main) inert while the drawer is open", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    const { container } = await renderShell()

    const header = container.querySelector("header")!
    const main = container.querySelector("main")!
    // Background is interactive until the modal drawer opens.
    expect(header).not.toHaveAttribute("inert")
    expect(main).not.toHaveAttribute("inert")

    await user.click(screen.getByRole("button", { name: "Open navigation" }))

    // aria-modal isn't universally honored, so inert is what actually keeps the
    // obscured page out of the tab order and the accessibility tree.
    expect(header).toHaveAttribute("inert")
    expect(main).toHaveAttribute("inert")
  })

  it("moves focus to the main region via the skip link without changing the route", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    const { container } = await renderShell()

    // The skip link is the first tab stop so keyboard users reach the page body
    // without traversing the whole nav.
    await user.tab()
    const skip = screen.getByRole("button", { name: "Skip to main content" })
    expect(skip).toHaveFocus()

    await user.keyboard("{Enter}")

    const main = container.querySelector("main")!
    expect(main).toHaveFocus()
    // Focus moved without navigating away: the index route is still rendered.
    expect(screen.getByText("PAGE CONTENT")).toBeInTheDocument()
  })

  it("makes the skip link inert while the drawer is open, matching its target", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell()

    const skip = screen.getByRole("button", { name: "Skip to main content" })
    // Live before the modal opens: it is the keyboard user's fast path to content.
    expect(skip).not.toHaveAttribute("inert")

    await user.click(screen.getByRole("button", { name: "Open navigation" }))

    // With the drawer open, main is inert, so focusing it would no-op. The skip
    // link goes inert too rather than sitting live in front of the modal as a
    // dead control an AT cursor reaches first.
    expect(skip).toHaveAttribute("inert")
  })

  it("hides the decorative nav icons from assistive tech", async () => {
    mockMatchMedia(false)
    const { container } = await renderShell()

    // Each nav link carries a visible text label, so its leading glyph is
    // decorative and must not be announced twice. Every SVG in the shell is marked
    // aria-hidden to match the rest of the codebase's convention.
    const icons = container.querySelectorAll("svg")
    expect(icons.length).toBeGreaterThan(0)
    icons.forEach((icon) => {
      expect(icon).toHaveAttribute("aria-hidden")
    })
  })

  it("links to the bundled user guide from the sidebar footer", async () => {
    mockMatchMedia(false)
    await renderShell()

    // A footer link points operators at the guide bundled with this dashboard,
    // discoverable without hunting for a separate docs site.
    const guideLink = screen.getByRole("link", { name: "User guide" })
    expect(guideLink).toHaveAttribute("href", "/docs")
  })

  it("subscribes via the legacy matchMedia API when addEventListener is absent", async () => {
    // Safari < 14 exposes only addListener/removeListener; the shell must still
    // react to breakpoint changes rather than throwing on a missing method.
    const { listeners } = mockMatchMedia(true, { legacy: true })
    await renderShell()

    // The component registered through addListener, so the captured set is live.
    expect(listeners.size).toBeGreaterThan(0)
    expect(
      screen.getByRole("button", { name: "Open navigation" }),
    ).toBeInTheDocument()

    act(() => {
      listeners.forEach((cb) => {
        cb({ matches: false } as MediaQueryListEvent)
      })
    })

    // A desktop-width change still flips the layout off the mobile drawer.
    expect(
      screen.getByRole("button", { name: "Open navigation" }),
    ).toHaveAttribute("aria-expanded", "false")
  })
})

describe("AppShell surface gating", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    window.localStorage.clear()
  })

  it("renders every destination the deployment serves", async () => {
    mockMatchMedia(false)
    await renderShell()

    // Every label, not a sample: a surface misspelled on a NAV entry hides that
    // destination in every deployment, and only the full list catches it. Kept
    // as an exact comparison rather than a loop of presence checks, because a
    // subset check keeps passing while the list quietly stops being every label
    // (which is what happened when the tenancy section landed).
    expect(
      within(screen.getByRole("navigation"))
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual([
      "Overview",
      "Activity",
      "Usage",
      "Models",
      "API keys",
      "Provider credentials",
      "Members",
    ])
    // Routing and Tools nest destinations, so they expand rather than
    // navigate; their children are links once the group is open.
    expect(
      within(screen.getByRole("navigation"))
        .getAllByRole("button")
        .map((button) => button.textContent),
    ).toEqual(["Routing", "Tools"])
  })

  it("renders every organization destination on that rail", async () => {
    mockMatchMedia(false)
    // Same reasoning as above, for the other context: the organization rail is
    // its own registry, and nothing else compares it against a full list.
    await renderShell(bootstrap(), { url: "/organization/members" })

    expect(
      within(screen.getByRole("navigation"))
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual([
      "Members & roles",
      "Workspaces",
      "Spend & budgets",
      "Users",
      "Organization",
      "Settings",
    ])
  })

  it("hides a destination whose surface the deployment does not host", async () => {
    mockMatchMedia(false)
    // A surface the bootstrap omits takes its link with it. Both observability
    // pages read /v1/usage, so both go.
    await renderShell(
      bootstrap({ surfaces: ["models", "providers", "settings"] }),
    )

    expect(screen.queryByRole("link", { name: "Activity" })).toBeNull()
    expect(screen.queryByRole("link", { name: "Usage" })).toBeNull()
    expect(screen.queryByRole("link", { name: "API keys" })).toBeNull()
    // Ungated and still present: the index is the deployment's front page.
    expect(screen.getByRole("link", { name: "Overview" })).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Provider credentials" }),
    ).toBeInTheDocument()
  })

  it("drops a section header once its whole group is gated away", async () => {
    mockMatchMedia(false)
    await renderShell(bootstrap({ surfaces: ["models"] }))

    // "Observe" labels Activity and Usage; with neither served, an empty
    // heading over nothing is worse than no heading.
    expect(screen.queryByText("Observe")).toBeNull()
    expect(screen.getByText("Gateway")).toBeInTheDocument()
  })
})

describe("AppShell entitlement and flag gating", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    window.localStorage.clear()
  })

  it("gates no shipping destination on a capability, so the sidebar is whole", async () => {
    mockMatchMedia(false)
    // Withholding every capability changes nothing today: no base entry is
    // tagged with one, because the routing split ARCHITECTURE.md calls
    // provisional has not been decided. This is what starts failing the day
    // someone adds a tag without adding its name to BASE_CAPABILITIES, which
    // would silently drop a page from the sidebar of every gateway.
    await renderShell(bootstrap(), { entitlements: { capabilities: [] } })

    // Routing nests destinations, so it is the group's expander.
    expect(screen.getByRole("button", { name: "Routing" })).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Models" })).toBeInTheDocument()
    expect(screen.getByText("Gateway")).toBeInTheDocument()
  })

  it("answers a gated-off destination with a panel, not the page", async () => {
    mockMatchMedia(false)
    // A bookmark or a shared URL still lands on the route after its link is
    // gone. Rendering the page anyway would fire requests the server refuses.
    // Driven from the surface axis, the only one a shipping entry declares;
    // useNavVisibility.test.tsx covers the other two on synthetic entries,
    // which is where they have users today.
    await renderShell(bootstrap({ surfaces: ["models"] }), {
      url: "/providers",
    })

    expect(
      await screen.findByText("Provider credentials is not available here"),
    ).toBeInTheDocument()
    expect(screen.queryByText("PAGE CONTENT")).toBeNull()
  })

  it("highlights one link on a nested route, and names it correctly", async () => {
    mockMatchMedia(false)
    // /organization/members is a child route of /organization in the generated
    // tree, and both are entries on the organization rail. TanStack's own
    // `activeProps` matches a parent as active, so the sidebar would light up
    // both; the shell drives the highlight from navItemForPath instead, which
    // prefers the exact entry.
    await renderShell(bootstrap(), { url: "/organization/members" })

    const members = await screen.findByRole("link", { name: "Members & roles" })
    const parent = screen.getByRole("link", { name: "Organization" })
    expect(members.className).toContain("bg-primary-subtle")
    expect(parent.className).not.toContain("bg-primary-subtle")
  })

  it("names a gated-off child route after the child, not its parent", async () => {
    mockMatchMedia(false)
    // Same prefix collision, seen from the panel: resolving the parent would
    // tell someone who followed a /organization/members link that
    // "Organization" is not available here.
    await renderShell(bootstrap({ surfaces: ["models"] }), {
      url: "/organization/members",
    })

    expect(
      await screen.findByText("Members & roles is not available here"),
    ).toBeInTheDocument()
  })

  it("still renders a destination that passes every axis", async () => {
    mockMatchMedia(false)
    await renderShell(bootstrap(), { url: "/providers" })

    expect(await screen.findByText("PAGE CONTENT")).toBeInTheDocument()
  })

  it("leaves a path the registry does not declare alone", async () => {
    mockMatchMedia(false)
    // The bundled guide is not a registered destination, so the registry has no
    // opinion on it and must not gate it away with the rest.
    await renderShell(bootstrap({ surfaces: [] }), {
      entitlements: { capabilities: [] },
      url: "/docs",
    })

    expect(await screen.findByText("PAGE CONTENT")).toBeInTheDocument()
  })

  it("swaps the whole rail when the route is an organization destination", async () => {
    mockMatchMedia(false)
    await renderShell(bootstrap(), { url: "/organization/members" })

    // The two rails never render together: entering the organization context
    // replaces the workspace nav rather than expanding a section inside it.
    expect(
      await screen.findByRole("link", { name: "Members & roles" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: "API keys" })).toBeNull()
    expect(screen.queryByRole("link", { name: "Activity" })).toBeNull()
  })

  it("offers a way into the organization rail, and a way back out", async () => {
    mockMatchMedia(false)
    await renderShell()

    // In: a footer entry, not a nav section.
    const enter = await screen.findByRole("link", { name: "Organization" })
    expect(enter).toHaveAttribute("href", "/organization/members")
    // Out only exists on the other rail.
    expect(screen.queryByText(/^Back to /)).toBeNull()
  })

  it("leaves the organization rail by its own way back", async () => {
    mockMatchMedia(false)
    await renderShell(bootstrap(), { url: "/organization/members" })

    // The organization rail has no switcher; it has the one link back to where
    // the shell opened. ("Organization" is a destination *on* this rail, so its
    // absence is not what distinguishes the two.)
    expect(await screen.findByText(/^Back to /)).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Switch workspace" }),
    ).toBeNull()
  })

  it("expands a nav group to reveal its nested destinations", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell()

    // Collapsed to start: the children are not in the tree at all, so they are
    // not reachable by tab or by a screen reader while the group is shut.
    expect(screen.queryByRole("link", { name: "Web search" })).toBeNull()

    await user.click(screen.getByRole("button", { name: "Tools" }))
    expect(screen.getByRole("link", { name: "Web search" })).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Code execution" }),
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Guardrails" })).toBeInTheDocument()
  })

  it("opens the group that holds the current route, without a click", async () => {
    mockMatchMedia(false)
    // Arriving by bookmark or shared URL should show where you are rather than
    // a shut group that hides it.
    await renderShell(bootstrap(), { url: "/tools/guardrails" })

    expect(
      await screen.findByRole("button", { name: "Tools", expanded: true }),
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Guardrails" })).toBeInTheDocument()
  })

  it("signs out from the account menu rather than the page header", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell()

    // The header carried this until the sidebar grew an account block.
    expect(screen.queryByRole("button", { name: "Sign out" })).toBeNull()
    await user.click(screen.getByRole("button", { name: "Account" }))
    expect(
      await screen.findByRole("button", { name: "Sign out" }),
    ).toBeInTheDocument()
  })
})
