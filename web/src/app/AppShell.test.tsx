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
  options: {
    entitlements?: Partial<Entitlements>
    url?: string
    /** The caller's membership, for the controls that gate on their role. */
    context?: Parameters<typeof organizationContext>[0]
  } = {},
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
    Response.json(organizationContext(options.context)),
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
    expect(aside?.className).toContain("-translate-x-full")
    // Absolute within the row the header leads, not fixed to the viewport: the
    // banners above that row are in flow, so a viewport offset left the drawer
    // over the top of the header by however tall they were, covering the one
    // control that closes it.
    expect(aside?.className).toContain("absolute")
    expect(aside?.className).not.toContain("fixed")
    expect(aside?.parentElement?.className).toContain("relative")

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
    await user.click(screen.getByRole("link", { name: "Providers" }))

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
    const providers = screen.getByRole("link", { name: "Providers" })
    expect(overview).toHaveAttribute("aria-current", "page")
    expect(providers).not.toHaveAttribute("aria-current")

    await user.click(providers)

    expect(await screen.findByText("PROVIDERS PAGE")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Providers" })).toHaveAttribute(
      "aria-current",
      "page",
    )
    expect(screen.getByRole("link", { name: "Overview" })).not.toHaveAttribute(
      "aria-current",
    )
  })

  it("renders the rail at one fixed width (not a drawer) on desktop", async () => {
    mockMatchMedia(false)
    const { container } = await renderShell()

    const aside = container.querySelector("aside")
    // Desktop keeps the in-flow rail rather than an overlay, and it is one
    // width: no inline style, because nothing drags it any more.
    expect(aside?.className).not.toContain("absolute")
    expect(aside?.className).toContain("w-[16.5rem]")
    expect(aside?.getAttribute("style")).toBeNull()
    expect(
      screen.queryByRole("separator", { name: "Resize sidebar" }),
    ).toBeNull()
  })

  it("collapses to the narrow rail rather than to a remembered width", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    const { container } = await renderShell()

    await user.click(screen.getByRole("button", { name: "Collapse sidebar" }))
    expect(container.querySelector("aside")?.className).toContain("w-[4.5rem]")
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

  it("fills the viewport below the top bar rather than floating over the page", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    const { container } = await renderShell()

    // The design's drawer is the width of the phone, starting under the top bar.
    // There is therefore nothing behind it to dim, which is why the backdrop that
    // used to dismiss it is gone: the control in that bar is what closes it.
    await user.click(screen.getByRole("button", { name: "Open navigation" }))

    const aside = container.querySelector("aside")!
    expect(aside.className).toContain("w-full")
    expect(aside.className).toContain("top-14")
    expect(container.querySelector(".fixed.inset-0")).toBeNull()
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

  it("makes the page inert while the drawer is open, and leaves the top bar live", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    const { container } = await renderShell()

    const header = container.querySelector("header")!
    const main = container.querySelector("main")!
    expect(main).not.toHaveAttribute("inert")

    await user.click(screen.getByRole("button", { name: "Open navigation" }))

    // The page behind the drawer goes inert, which is what keeps controls nobody
    // can see out of the tab order and the accessibility tree. The top bar does
    // not: the control that closes the drawer is in it, and inerting it would
    // strand a keyboard user with only Escape.
    expect(main).toHaveAttribute("inert")
    expect(header).not.toHaveAttribute("inert")
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

  it("links to the bundled user guide from the top bar", async () => {
    mockMatchMedia(false)
    await renderShell()

    // In the top bar, where the design puts it, rather than inside the account
    // menu: the guide is read alongside a page rather than instead of one.
    const guideLink = await screen.findByRole("link", { name: "Documentation" })
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
      within(screen.getByRole("navigation", { name: "Sidebar" }))
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual([
      "Overview",
      "Activity",
      "Usage",
      "Models",
      "API keys",
      "Providers",
      "Members",
      "Budget defaults",
    ])
    // Routing and Tools nest destinations, so they expand rather than
    // navigate; their children are links once the group is open.
    expect(
      within(screen.getByRole("navigation", { name: "Sidebar" }))
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
      within(screen.getByRole("navigation", { name: "Sidebar" }))
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual([
      "Workspaces",
      "Members & roles",
      "Users",
      "Spend & budgets",
      "Model pricing",
      "Org settings",
      "Settings",
    ])
    // The design's rail has four more rows (the organization's own Providers,
    // Billing, Guardrails and Gateways), and each is gated on a surface a
    // standalone gateway does not report, so none of them is here. The Gateway
    // group is all four's worst case: both its rows are gated, so the heading
    // goes with them.
    expect(screen.queryByText("Gateway")).toBeNull()
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
    expect(screen.getByRole("link", { name: "Providers" })).toBeInTheDocument()
  })

  it("drops a section header once its whole group is gated away", async () => {
    mockMatchMedia(false)
    await renderShell(bootstrap({ surfaces: ["models"] }))

    // "Access" labels keys, providers and the workspace roster; with none of
    // them served, an empty heading over nothing is worse than no heading.
    expect(screen.queryByText("Access")).toBeNull()
    expect(screen.getByText("Gateway")).toBeInTheDocument()
    // "Observe" goes with them. It labels the request log and the usage
    // rollups, both gated on the usage surface, and the index that used to keep
    // the heading alive now sits in its own group above it. The front page is
    // still there, which is the part that matters: a deployment with no
    // management surface at all is left with a landing row and no headings it
    // cannot fill.
    expect(screen.queryByText("Observe")).toBeNull()
    expect(screen.getByRole("link", { name: "Overview" })).toBeInTheDocument()
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
      await screen.findByText("Providers is not available here"),
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
    const parent = screen.getByRole("link", { name: "Org settings" })
    // The selected fill, which the navigation design draws as a lifted chip
    // (`--color-surface-muted`, reached through `bg-surface-alt`) rather than the
    // tinted `bg-primary-subtle` this rail used to wear. Asserted as the class
    // because it is what a reader of the rail actually sees; `aria-current` is
    // covered separately.
    expect(members.className).toContain("bg-surface-alt")
    expect(parent.className).not.toContain("bg-surface-alt")
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
    // A link on the desk, where the rail it swaps to stays on screen to read.
    // The drawer's submenu trigger is the mobile shape of this row and must not
    // turn up beside it.
    expect(screen.queryByRole("button", { name: "Organization" })).toBeNull()
    // Out only exists on the other rail.
    expect(screen.queryByText(/^Back to /)).toBeNull()
  })

  it("opens the organization rail inside the mobile drawer rather than navigating", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell()

    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    await user.click(
      await screen.findByRole("button", { name: "Organization" }),
    )

    // The rail swapped, one level down inside the drawer: the organization's
    // destinations are all there to choose from, and the workspace's are not.
    expect(
      await screen.findByRole("link", { name: "Members & roles" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: "API keys" })).toBeNull()
    // Nothing navigated, so the page behind the drawer is the one the menu was
    // opened from, and the drawer is still over it.
    expect(screen.getByText("PAGE CONTENT")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Close navigation" }),
    ).toHaveAttribute("aria-expanded", "true")
    // Focus follows the level, onto the control that leaves it again: the row
    // that opened it is gone, so without this a keyboard or AT cursor would be
    // dropped to the top of the document.
    expect(screen.getByRole("button", { name: /^Back to / })).toHaveFocus()
  })

  it("dismisses the drawer once a destination on that submenu is chosen", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell()

    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    await user.click(
      await screen.findByRole("button", { name: "Organization" }),
    )
    await user.click(
      await screen.findByRole("link", { name: "Members & roles" }),
    )

    // This is the tap that navigates, so this is the tap that closes.
    expect(
      screen.getByRole("button", { name: "Open navigation" }),
    ).toHaveAttribute("aria-expanded", "false")
    expect(screen.queryByText("PAGE CONTENT")).toBeNull()
    // And the rail is now the organization one because the route says so, which
    // is what the way back out belongs to.
    expect(
      await screen.findByRole("link", { name: /^Back to / }),
    ).toBeInTheDocument()
  })

  it("returns to the workspace menu from the submenu, without navigating", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell()

    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    await user.click(
      await screen.findByRole("button", { name: "Organization" }),
    )
    // A button, not the link the organization rail's own head is: the page
    // under the drawer never left the workspace, so there is nowhere to go.
    await user.click(screen.getByRole("button", { name: /^Back to / }))

    expect(
      await screen.findByRole("link", { name: "API keys" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: "Members & roles" })).toBeNull()
    expect(screen.getByText("PAGE CONTENT")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Close navigation" }),
    ).toHaveAttribute("aria-expanded", "true")
    // Focus comes back to the row that opened the level.
    expect(screen.getByRole("button", { name: "Organization" })).toHaveFocus()
  })

  it("unwinds one level at a time on Escape", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell()

    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    await user.click(
      await screen.findByRole("button", { name: "Organization" }),
    )

    // The submenu first: taking the whole drawer would be the wrong amount of
    // dismissal when the row you were after is the level above.
    await user.keyboard("{Escape}")
    expect(
      await screen.findByRole("link", { name: "API keys" }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Close navigation" }),
    ).toHaveAttribute("aria-expanded", "true")

    await user.keyboard("{Escape}")
    expect(
      screen.getByRole("button", { name: "Open navigation" }),
    ).toHaveAttribute("aria-expanded", "false")
  })

  it("reopens the drawer on the workspace menu, not on the level it was left in", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell()

    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    await user.click(
      await screen.findByRole("button", { name: "Organization" }),
    )
    await user.click(screen.getByRole("button", { name: "Close navigation" }))
    await user.click(screen.getByRole("button", { name: "Open navigation" }))

    // The submenu belongs to the drawer that framed it. Surviving the close
    // would show the organization's rows to someone reopening the menu from a
    // workspace page, with no tap of theirs to explain it.
    expect(
      await screen.findByRole("link", { name: "API keys" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: "Members & roles" })).toBeNull()
  })

  it("offers Create workspace to a role the server would let create one", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell()

    await user.click(
      await screen.findByRole("button", { name: /^Switch workspace/ }),
    )
    const menu = await screen.findByRole("dialog")
    expect(
      within(menu).getByRole("button", { name: "Create workspace" }),
    ).toBeInTheDocument()
  })

  it("withholds Create workspace from a role the server would refuse", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    // `POST /v1/workspaces` is owners and admins only, and the Workspaces page
    // gates its own create control on the same predicate. Offering it here would
    // hand a member the whole form and report the refusal as a 403 after they
    // had typed a name.
    await renderShell(bootstrap(), { context: { role: "member" } })

    await user.click(
      await screen.findByRole("button", { name: /^Switch workspace/ }),
    )
    const menu = await screen.findByRole("dialog")
    expect(
      within(menu).queryByRole("button", { name: "Create workspace" }),
    ).toBeNull()
    // The organization row is still there, so this is the create row missing
    // rather than the menu failing to open.
    expect(within(menu).getByText("Default Organization")).toBeInTheDocument()
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

  it("returns you to where you were on each rail rather than to its landing", async () => {
    mockMatchMedia(false)
    // The design's A ⇄ B note: crossing a rail should resume it, not reset it.
    // Seeded through the memory rather than by navigating twice, because the
    // controls under test are what read it.
    window.localStorage.setItem(
      "otari.dashboard.lastOrganizationLocation",
      "/budgets",
    )
    await renderShell()

    expect(
      await screen.findByRole("link", { name: "Organization" }),
    ).toHaveAttribute("href", "/budgets")
  })

  it("lands on the rail's own first page when there is nothing to resume", async () => {
    mockMatchMedia(false)
    // The solo-operator path: nothing remembered, so the way in must still be one
    // click to somewhere useful rather than a dead control.
    await renderShell()

    expect(
      await screen.findByRole("link", { name: "Organization" }),
    ).toHaveAttribute("href", "/organization/members")
  })

  it("does not resume onto a destination this deployment gates off", async () => {
    mockMatchMedia(false)
    // Registered and still an organization destination, so the registry check
    // passes it; the gateway was restarted against a config that no longer
    // reports the surface. The only way onto that rail must not land on the
    // "not available here" panel.
    window.localStorage.setItem(
      "otari.dashboard.lastOrganizationLocation",
      "/settings",
    )
    await renderShell(bootstrap({ surfaces: ["organizations", "budgets"] }))

    expect(
      await screen.findByRole("link", { name: "Organization" }),
    ).toHaveAttribute("href", "/organization/members")
  })

  it("does not record a gated-off route it answered with the panel", async () => {
    mockMatchMedia(false)
    // Reached by URL, refused by the shell. Recording it would make the panel
    // the page the rail resumes to.
    await renderShell(bootstrap({ surfaces: ["organizations"] }), {
      url: "/settings",
    })
    await screen.findByText("Settings is not available here")

    expect(
      window.localStorage.getItem("otari.dashboard.lastOrganizationLocation"),
    ).toBeNull()
  })

  it("expands a nav group to reveal its nested destinations", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell()

    // Shut to start. The panel stays mounted so it has a height to animate from,
    // but it is `hidden` and `aria-hidden` while shut, so its rows are reachable
    // by neither tab nor a screen reader until the group opens.
    expect(screen.queryByRole("link", { name: "Web search" })).toBeNull()
    expect(screen.getByText("Web search").closest("[hidden]")).not.toBeNull()

    await user.click(screen.getByRole("button", { name: "Tools" }))
    expect(screen.getByRole("link", { name: "Web search" })).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Code execution" }),
    ).toBeInTheDocument()
  })

  it("opens the group that holds the current route, without a click", async () => {
    mockMatchMedia(false)
    // Arriving by bookmark or shared URL should show where you are rather than
    // a shut group that hides it.
    await renderShell(bootstrap(), { url: "/tools/web-search" })

    expect(
      await screen.findByRole("button", { name: "Tools", expanded: true }),
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Web search" })).toBeInTheDocument()
  })

  it("signs out from the account menu rather than the page header", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell()

    // The header carried this until the sidebar grew an account block.
    expect(screen.queryByRole("button", { name: "Log out" })).toBeNull()
    await user.click(screen.getByRole("button", { name: "Account" }))
    expect(
      await screen.findByRole("button", { name: "Log out" }),
    ).toBeInTheDocument()
  })

  it("keeps the bundled guide reachable from the account menu, for the phone", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell()

    // The top bar's cluster is `hidden md:flex`, and this menu is what the
    // mobile drawer contains, so without this row the guide has no control
    // pointing at it below the breakpoint. The row carries `md:hidden` for the
    // mirror-image reason: above it, the top bar already answers.
    await user.click(screen.getByRole("button", { name: "Account" }))
    const guideRow = await screen.findByRole("link", { name: "Documentation" })
    expect(guideRow).toHaveAttribute("href", "/docs")
    expect(guideRow).toHaveClass("md:hidden")
  })

  it("names the scope and the page, leaving out an organization there is one of", async () => {
    mockMatchMedia(false)
    await renderShell()

    // Standalone has exactly one organization and no way to mint a second, so
    // naming it on every page is a segment that never disambiguates anything.
    const crumb = await screen.findByLabelText("Breadcrumb")
    expect(crumb).toHaveTextContent("Overview")
    expect(crumb).not.toHaveTextContent(/organization/i)
  })

  it("names a nested destination after itself, not after its group", async () => {
    mockMatchMedia(false)
    // `navItemForPath` answers with the entry that gates a path, which for a
    // child is its parent; a breadcrumb wants the leaf.
    await renderShell(bootstrap(), { url: "/tools/web-search" })

    expect(await screen.findByLabelText("Breadcrumb")).toHaveTextContent(
      "Web search",
    )
  })

  it("leads the trail with the organization when a deployment can hold several", async () => {
    mockMatchMedia(false)
    // The only way to exercise this: no gateway in this repository reports
    // `hosted` (bootstrap.py answers standalone or hybrid), and the endpoints
    // that would mint a second organization are not mounted in standalone. So
    // the multi-organization trail has no live deployment to be seen on, and
    // this is what keeps it from rotting until the hosted shell arrives.
    await renderShell(bootstrap({ deployment_type: "hosted" }))

    const crumb = await screen.findByLabelText("Breadcrumb")
    expect(crumb).toHaveTextContent("Default Organization")
    expect(crumb).toHaveTextContent("Overview")
  })

  it("drops a nested destination whose own surface the deployment lacks", async () => {
    mockMatchMedia(false)
    // Guardrails is grouped under Routing but served by the tools surface. The
    // link used to render anyway and land on the "not available here" panel.
    await renderShell(bootstrap({ surfaces: ["routing"] }))

    expect(screen.queryByRole("link", { name: "Guardrails" })).toBeNull()
    // And with Guardrails gone, Routing holds one child, so it stops being a
    // group: a disclosure that opens onto a single row asks for a click to tell
    // you nothing. The row wears the parent's name and goes straight to it.
    expect(screen.queryByRole("button", { name: "Routing" })).toBeNull()
    const routing = await screen.findByRole("link", { name: "Routing" })
    expect(routing).toHaveAttribute("href", "/routing")
    expect(screen.queryByRole("link", { name: "Policies" })).toBeNull()
  })

  it("answers a nested route with the panel when the child's own surface is gone", async () => {
    mockMatchMedia(false)
    // The other half of the case above: dropping the link is not enough, because
    // a bookmark still reaches the route. `NAV_CHILD_PARENTS` carries the child's
    // surface onto the entry `navItemForPath` answers with, which is what lets
    // the shell refuse a path whose *parent* surface it does host.
    await renderShell(bootstrap({ surfaces: ["routing"] }), {
      url: "/tools/guardrails",
    })

    expect(
      await screen.findByText("Guardrails is not available here"),
    ).toBeInTheDocument()
    expect(screen.queryByText("PAGE CONTENT")).toBeNull()
  })

  it("reaches a group's nested destinations from the collapsed rail", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    // The gap this closes: collapsed, the group used to link straight to its own
    // page, so Web search and Code execution had no affordance at all and a
    // bookmark was the only way back to them.
    window.localStorage.setItem("otari.dashboard.sidebarCollapsed", "1")
    await renderShell()

    expect(screen.queryByRole("link", { name: "Tools" })).toBeNull()
    await user.click(screen.getByRole("button", { name: "Tools" }))

    expect(
      await screen.findByRole("link", { name: "Web search" }),
    ).toHaveAttribute("href", "/tools/web-search")
    expect(
      screen.getByRole("link", { name: "Code execution" }),
    ).toBeInTheDocument()
  })

  it("marks one link as the current page on a nested route", async () => {
    mockMatchMedia(false)
    // TanStack's Link matches a prefix by default, so /organization/members
    // left `aria-current` on "Organization" too. The className was already
    // driven from the registry; this is the half a screen reader reads.
    await renderShell(bootstrap(), { url: "/organization/members" })

    await screen.findByRole("link", { name: "Members & roles" })
    const current = screen
      .getAllByRole("link")
      .filter((link) => link.getAttribute("aria-current") === "page")
      .map((link) => link.textContent)
    expect(current).toEqual(["Members & roles"])
  })
})
