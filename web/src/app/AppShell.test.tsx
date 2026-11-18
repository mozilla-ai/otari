import { act, cleanup, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { useState } from "react"
import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest"
import { AppShell } from "@/app/AppShell"
import { navItemForPath } from "@/app/nav/registry"
import { navRowClass } from "@/app/nav/rowStyles"
import { Provider } from "@/app/provider"
import type {
  CallerOrganizationMembership,
  DeploymentBootstrap,
  GatewaySettings,
  OrganizationSpendCeiling,
} from "@/client"
import { useOpenCreateKey } from "@/features/keys/openCreateKey"
import { API_ROOT } from "@/shared/api/client"
import { useBreadcrumbLeaf } from "@/shared/hooks/BreadcrumbLeaf"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import type { Entitlements } from "@/shared/hooks/useEntitlements"
import {
  BASE_CAPABILITIES,
  EntitlementProvider,
} from "@/shared/hooks/useEntitlements"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import {
  bootstrap,
  callerOrganizationMembership,
  organizationContext,
  organizationSpendCeiling,
  pendingOrganizationInvitation,
  SURFACES,
} from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"
import { recordEvent, resetTelemetrySpy } from "@/tests/telemetry"

// The telemetry seam, replaced the way a superset build's alias replaces it: the
// base module records nothing, so a navigation is only observable through a
// stand-in.
vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
  const { telemetrySpy } = await import("@/tests/telemetry")
  return { useTelemetry: vi.fn(() => telemetrySpy) }
})

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
    settings?: GatewaySettings
    /** The caller's membership, for the controls that gate on their role. */
    context?: Parameters<typeof organizationContext>[0]
    /** The organizations the caller belongs to, for the organization switcher. */
    memberships?: CallerOrganizationMembership[]
    /** Whether the deployment administration surface admits this caller. */
    operator?: boolean
    /** The caller belongs to no organization, so the context read 404s. */
    noOrganization?: boolean
    /** How many invitations are waiting on the caller. */
    pendingInvitations?: number
    /** What the route at `url` renders. */
    page?: ReactElement
    /** The organization's spend ceilings, for the rail's team budget. */
    spendCeilings?: OrganizationSpendCeiling[]
  } = {},
) {
  const entitlements: Entitlements = {
    capabilities: BASE_CAPABILITIES,
    isLoading: false,
    ...options.entitlements,
  }
  const url = options.url ?? "/"
  // The shell reads the organization context to decide whether to offer the way
  // into that rail, and the switcher reads it for the names it shows. The
  // switcher additionally reads the caller's own memberships, which is a
  // different shape (`{ data, count }`), and the pricing alarm reads settings,
  // so this answers per path rather than handing every request one body.
  const memberships = options.memberships ?? [callerOrganizationMembership()]
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input)
    // The caller axis is a field on the membership context now (see the
    // fall-through below), so this stays only for the page behind the
    // operator-only route, which asks the surface's own gate for its refusal
    // panel. Answered from the same option, because a harness that could tell
    // the rail one thing and the page another would be describing a deployment
    // that does not exist.
    if (path.startsWith(`${API_ROOT}/admin/access`)) {
      return Response.json({ granted: options.operator ?? true })
    }
    if (path.startsWith(`${API_ROOT}/admin/users`)) {
      return Response.json({ data: [], count: 0 })
    }
    if (path.startsWith(`${API_ROOT}/organizations/me/memberships`)) {
      return Response.json({ data: memberships, count: memberships.length })
    }
    if (path.includes(`${API_ROOT}/settings`)) {
      return Response.json(options.settings ?? SETTINGS_WITH_PRICING)
    }
    if (path.startsWith(`${API_ROOT}/organizations/me/pending-memberships`)) {
      const waiting = Array.from(
        { length: options.pendingInvitations ?? 0 },
        (_, index) =>
          pendingOrganizationInvitation({
            organization_member_id: `44444444-4444-4444-4444-44444444444${index}`,
          }),
      )
      return Response.json({ data: waiting, count: waiting.length })
    }
    if (options.noOrganization && path === `${API_ROOT}/organizations/me`) {
      return Response.json(
        { detail: "Organization not found" },
        { status: 404 },
      )
    }
    // The membership context, which carries the caller axis. An operator by
    // default, because eight deployment-wide rows declare that axis and every case
    // below that is about the *surface* or *entitlement* axis wants a whole
    // sidebar to reason about; `options.context` still wins where a case names
    // the field itself.
    if (path === `${API_ROOT}/organizations/me`) {
      return Response.json(
        organizationContext({
          deployment_operator: options.operator ?? true,
          ...options.context,
        }),
      )
    }
    if (
      options.spendCeilings &&
      path.startsWith(`${API_ROOT}/organizations/me/spend-ceilings`)
    ) {
      return Response.json({
        data: options.spendCeilings,
        count: options.spendCeilings.length,
      })
    }
    // The aliases are a bare list, read by the Create API key dialog.
    if (path.includes("/aliases")) return Response.json([])
    // Everything else the chrome reads is a list as far as the rail is
    // concerned, and an empty one draws nothing.
    return Response.json({ data: [], count: 0 })
  })
  return renderWithRouter(options.page ?? <div>PAGE CONTENT</div>, {
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
    //
    // Home, so a page that sends its visitor there is visible. And
    // `/organization`, because the shell asks the router who may open that page
    // before it draws the Organization row; an address the router does not
    // serve reads as ungated.
    routes: [
      { path: "/", element: <div>HOME PAGE</div> },
      { path: "/organization", element: <div>ORG SETTINGS PAGE</div> },
      { path: "/invitations", element: <div>INVITATIONS PAGE</div> },
      {
        path: "/organization/provider-keys",
        element: <div>ORG PROVIDER KEYS PAGE</div>,
      },
    ].filter((route) => route.path !== url),
  })
}

const SETTINGS_WITH_PRICING: GatewaySettings = {
  version: "1.0.0",
  model_discovery: true,
  default_pricing: true,
  require_pricing: false,
  master_key_source: "configured",
  secret_key_configured: true,
  config: [],
}

// require_pricing on with default_pricing off is what raises the alarm.
const SETTINGS_NEEDING_PRICING: GatewaySettings = {
  ...SETTINGS_WITH_PRICING,
  default_pricing: false,
  require_pricing: true,
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

  it("draws the desk's rows in the mobile drawer, and offers Playground from the account menu", async () => {
    // Below `md` the top bar that links Playground is hidden, and the account
    // menu, which the drawer contains, carries it instead of a rail row.
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell()

    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    const sidebar = screen.getByRole("navigation", { name: "Sidebar" })
    await within(sidebar).findByRole("link", { name: "API keys" })
    expect(
      within(sidebar)
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual([
      "Home",
      "Sessions",
      "People",
      "Rules",
      "Models",
      "Tools",
      "API keys",
    ])

    await user.click(screen.getByRole("button", { name: /^Account:/ }))
    expect(await screen.findByRole("link", { name: "Playground" })).toHaveClass(
      "md:hidden",
    )
  })

  it("leaves Playground out of the desktop rail", async () => {
    // The top bar links it there.
    mockMatchMedia(false)
    await renderShell()

    const sidebar = screen.getByRole("navigation", { name: "Sidebar" })
    await within(sidebar).findByRole("link", { name: "Models" })
    expect(
      within(sidebar).queryByRole("link", { name: "Playground" }),
    ).not.toBeInTheDocument()
  })

  it("keeps the mobile drawer controls usable while pricing needs attention", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell(undefined, { settings: SETTINGS_NEEDING_PRICING })

    const warning = await screen.findByText(
      /Requests are rejected until pricing/,
    )
    // Out of flow, so the alarm overlays the shell rather than displacing it.
    expect(warning.closest("main")).toBeNull()

    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    await user.click(screen.getByRole("button", { name: "Close navigation" }))

    expect(
      screen.getByRole("button", { name: "Open navigation" }),
    ).toBeInTheDocument()
  })

  it("dismisses the mobile drawer after navigating to a destination", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell(bootstrap(), { url: "/organization" })

    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    await user.click(await screen.findByRole("button", { name: "Providers" }))
    await user.click(await screen.findByRole("link", { name: "Inference" }))

    expect(
      await screen.findByText("ORG PROVIDER KEYS PAGE"),
    ).toBeInTheDocument()
    expect(document.title).toBe("Inference · Otari")
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

    // On the index first, because that is the half the comment above is about:
    // "/" lights here and must stop lighting once a page under it is open.
    expect(document.title).toBe("Home · Otari")
    expect(screen.getByRole("link", { name: "Home" })).toHaveAttribute(
      "aria-current",
      "page",
    )

    await user.click(await screen.findByRole("link", { name: "Organization" }))
    await user.click(await screen.findByRole("button", { name: "Providers" }))
    const inference = await screen.findByRole("link", { name: "Inference" })
    expect(inference).not.toHaveAttribute("aria-current")

    await user.click(inference)

    expect(
      await screen.findByText("ORG PROVIDER KEYS PAGE"),
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Inference" })).toHaveAttribute(
      "aria-current",
      "page",
    )
    expect(
      screen.getByRole("link", { name: "Org settings" }),
    ).not.toHaveAttribute("aria-current")
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

  it("resets the submenu and preserves focus across the mobile breakpoint", async () => {
    const { listeners } = mockMatchMedia(true)
    const user = userEvent.setup()
    const { container } = await renderShell()

    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    await user.click(
      await screen.findByRole("button", { name: "Organization" }),
    )
    expect(
      await screen.findByRole("link", { name: "Members & roles" }),
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
    // The focused submenu control unmounted, but focus stays on the navigation
    // landmark rather than falling back to the document body.
    expect(container.querySelector("aside")).toHaveFocus()

    // Returning to mobile and reopening starts at the workspace level. The
    // submenu belongs to the drawer that framed it and cannot survive the
    // breakpoint transition.
    act(() => {
      listeners.forEach((cb) => {
        cb({ matches: true } as MediaQueryListEvent)
      })
    })
    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    expect(
      await screen.findByRole("link", { name: "API keys" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: "Members & roles" })).toBeNull()
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

  it("ends the page with room for a control fixed to the corner", async () => {
    const { container } = await renderShell()

    // A superset sets `--shell-corner-clearance` while it pins something to the
    // bottom-right corner, so the end of a page can scroll out from under it.
    const content = container.querySelector("#main-content > div")!
    expect(content.className).toContain(
      "pb-[calc(1.25rem+var(--shell-corner-clearance))]",
    )
    expect(content.className).toContain(
      "md:pb-[calc(1.5rem+var(--shell-corner-clearance))]",
    )
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

  it("links to the documentation site from the top bar", async () => {
    mockMatchMedia(false)
    await renderShell()

    // In the top bar, where the design puts it, rather than inside the account
    // menu: the docs are read alongside a page rather than instead of one.
    const guideLink = await screen.findByRole("link", { name: "Documentation" })
    expect(guideLink).toHaveAttribute("href", "https://docs.otari.ai/")
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

describe("AppShell for a caller with no organization", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    window.localStorage.clear()
  })

  it("draws the index alone, with no way into an organization rail", async () => {
    mockMatchMedia(false)
    await renderShell(bootstrap(), { noOrganization: true })

    const sidebar = await screen.findByRole("navigation", { name: "Sidebar" })
    expect(
      await within(sidebar).findByRole("link", { name: "Home" }),
    ).toBeInTheDocument()
    // A 404 is no organization, not an unavailable context, which would fail
    // open onto the operator rows and the organization rail.
    expect(within(sidebar).getAllByRole("link")).toHaveLength(1)
    expect(screen.queryByRole("link", { name: "Organization" })).toBeNull()
    // There is no workspace to mint a key in, and the server would refuse it.
    expect(screen.queryByRole("button", { name: "Create API key" })).toBeNull()
  })
})

describe("the invitations page in the chrome", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    window.localStorage.clear()
  })

  it("titles the tab and the breadcrumb", async () => {
    mockMatchMedia(false)
    await renderShell(bootstrap(), { url: "/invitations" })

    expect(await screen.findByLabelText("Breadcrumb")).toHaveTextContent(
      "Invitations",
    )
    expect(document.title).toBe("Invitations · Otari")
  })

  it("closes the mobile drawer when the switcher opens the inbox", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell(bootstrap(), { pendingInvitations: 1 })

    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    await user.click(
      await screen.findByRole("button", { name: /^Switch workspace/ }),
    )
    await user.click(
      await screen.findByRole("button", { name: "1 invitation waiting" }),
    )

    expect(await screen.findByText("INVITATIONS PAGE")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Open navigation" }),
    ).toHaveAttribute("aria-expanded", "false")
  })
})

describe("the rail's closing rules", () => {
  // The footer block and the closing band each draw a rule. Between them sits
  // one row, the way onto the organization rail, and on that rail it is gated
  // out: the two rules then land 4px apart and read as a single doubled
  // hairline, which is what a reader reported. The band's rule is the
  // unconditional one, mirroring the scope band at the head of the rail, so the
  // footer's is the one that gives way.
  const footerOf = (container: HTMLElement) =>
    container.querySelector('[class*="-mx-3 flex flex-col gap-1 px-3 pt-1"]')

  it("keeps the footer rule where a row sits below it", async () => {
    mockMatchMedia(false)
    const { container } = await renderShell(bootstrap(), { url: "/" })

    expect(
      await screen.findByRole("link", { name: "Organization" }),
    ).toBeInTheDocument()
    expect(footerOf(container)).toHaveClass("border-t")
  })

  it("drops it on the organization rail, where that row is gone", async () => {
    mockMatchMedia(false)
    const { container } = await renderShell(bootstrap(), {
      url: "/organization/members",
    })

    expect(
      await screen.findByRole("link", { name: /^Back to/ }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: "Organization" })).toBeNull()
    expect(footerOf(container)).not.toHaveClass("border-t")
  })
})

describe("the shell's outer box", () => {
  // Found by walking up from the content region rather than by its classes, so
  // the search cannot assert what it is looking for.
  const outerBoxOf = (container: HTMLElement) => {
    const main = container.querySelector("main#main-content")
    if (!main) throw new Error("the shell rendered no content region")
    let box = main as HTMLElement
    while (box.parentElement && box.parentElement !== container) {
      box = box.parentElement
    }
    return box
  }

  // `overflow: hidden` here would make this box a scroll container that focus
  // can scroll, taking the top bar and the rail off screen with no scrollbar
  // and no wheel to bring them back. jsdom computes no layout, so what is
  // pinned is the class that causes it; e2e/parity.models.spec.ts covers the
  // behavior in a browser.
  it("clips its overflow instead of becoming a scroll container", async () => {
    mockMatchMedia(false)
    const { container } = await renderShell()

    const box = outerBoxOf(container)
    expect(box).toHaveClass("overflow-clip")
    expect(box).not.toHaveClass("overflow-hidden")
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
    const sidebar = screen.getByRole("navigation", { name: "Sidebar" })
    // The rail beyond Home waits on the membership read.
    await within(sidebar).findByRole("link", { name: "API keys" })
    // Every label, not a sample: a surface misspelled on a NAV entry hides that
    // destination in every deployment, and only the full list catches it. Kept
    // as an exact comparison rather than a loop of presence checks, because a
    // subset check keeps passing while the list quietly stops being every label.
    expect(
      within(sidebar)
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual([
      "Home",
      "Sessions",
      "People",
      "Rules",
      "Models",
      "Tools",
      "API keys",
    ])
    // One level: no row on this rail is a group.
    expect(within(sidebar).queryAllByRole("button")).toEqual([])
  })

  it("renders every organization destination on that rail", async () => {
    mockMatchMedia(false)
    // Same reasoning as above, for the other context: the organization rail is
    // its own registry, and nothing else compares it against a full list.
    await renderShell(bootstrap(), { url: "/organization/members" })
    // Awaited so the rail has settled before the snapshot below.
    await within(
      screen.getByRole("navigation", { name: "Sidebar" }),
    ).findByRole("button", { name: "Providers" })

    expect(
      within(screen.getByRole("navigation", { name: "Sidebar" }))
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual([
      "Workspaces",
      "Members & roles",
      "Email domains",
      "Single sign-on",
      "Spend & budgets",
      "Guardrails",
      "Org settings",
    ])
    // Providers nests the organization's two kinds of provider key.
    expect(
      within(screen.getByRole("navigation", { name: "Sidebar" }))
        .getAllByRole("button")
        .map((button) => button.textContent),
    ).toEqual(["Providers"])
    // Settings and Accounts are on the deployment rail now, not this one.
    expect(
      within(screen.getByRole("navigation", { name: "Sidebar" })).queryByRole(
        "link",
        { name: "Accounts" },
      ),
    ).toBeNull()
    expect(screen.getByText("General")).toBeInTheDocument()
  })

  it("sends the organization's Providers rows to its two key pages", async () => {
    mockMatchMedia(false)
    // The registry test reads the declaration and the pages' own tests render
    // them without a rail, so nothing else asserts the rail draws them.
    await renderShell(bootstrap(), { url: "/organization/members" })
    const sidebar = screen.getByRole("navigation", { name: "Sidebar" })
    const user = userEvent.setup()
    await user.click(
      await within(sidebar).findByRole("button", { name: "Providers" }),
    )
    const inference = await within(sidebar).findByRole("link", {
      name: "Inference",
    })
    expect(inference).toHaveAttribute("href", "/organization/provider-keys")
    expect(
      within(sidebar).getByRole("link", { name: "Web search" }),
    ).toHaveAttribute("href", "/organization/web-search-keys")
    // Guardrails then Org settings after them, read off the rendered order
    // rather than off the registry.
    expect(
      within(sidebar)
        .getAllByRole("link")
        .map((link) => link.textContent)
        .slice(-4),
    ).toEqual(["Inference", "Web search", "Guardrails", "Org settings"])

    await user.click(inference)

    expect(
      await screen.findByText("ORG PROVIDER KEYS PAGE"),
    ).toBeInTheDocument()
  })

  it("hides a destination whose surface the deployment does not host", async () => {
    mockMatchMedia(false)
    // A surface the bootstrap omits takes its link with it.
    await renderShell(bootstrap({ surfaces: ["models", "settings"] }))

    // Awaited first, so the absences below are gated rows rather than rows
    // still waiting on the membership read.
    expect(
      await screen.findByRole("link", { name: "Models" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: "API keys" })).toBeNull()
    expect(screen.queryByRole("link", { name: "Tools" })).toBeNull()
    // Ungated and still present: the index is the deployment's front page.
    expect(screen.getByRole("link", { name: "Home" })).toBeInTheDocument()
  })

  it("drops a section header once its whole group is gated away", async () => {
    mockMatchMedia(false)
    // The organization rail is the one with headings. "Cost & billing" holds
    // Spend & budgets alone, so a deployment without that surface has nothing
    // to put under it, and an empty heading over nothing is worse than none.
    await renderShell(
      bootstrap({
        surfaces: SURFACES.filter((surface) => surface !== "budgets"),
      }),
      { url: "/organization/members" },
    )
    const sidebar = screen.getByRole("navigation", { name: "Sidebar" })
    await within(sidebar).findByRole("link", { name: "Members & roles" })

    expect(within(sidebar).queryByText("Cost & billing")).toBeNull()
    expect(within(sidebar).getByText("People & access")).toBeInTheDocument()
    expect(within(sidebar).getByText("General")).toBeInTheDocument()
  })
})

describe("AppShell entitlement gating", () => {
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

    const sidebar = screen.getByRole("navigation", { name: "Sidebar" })
    await within(sidebar).findByRole("link", { name: "API keys" })
    expect(
      within(sidebar)
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual([
      "Home",
      "Sessions",
      "People",
      "Rules",
      "Models",
      "Tools",
      "API keys",
    ])
  })

  it("answers a gated-off destination with a panel, not the page", async () => {
    mockMatchMedia(false)
    // A bookmark or a shared URL still lands on the route after its link is
    // gone. Rendering the page anyway would fire requests the server refuses.
    // Driven from the surface axis, the only one a shipping entry declares;
    // useNavVisibility.test.tsx covers the other two on synthetic entries,
    // which is where they have users today.
    await renderShell(bootstrap({ surfaces: ["models"] }), {
      url: "/keys",
    })

    expect(
      await screen.findByText("API keys is not available here"),
    ).toBeInTheDocument()
    expect(screen.queryByText("PAGE CONTENT")).toBeNull()
    // The sidebar is behind a menu on a phone, so the panel carries its own way
    // out and its copy names the menu.
    expect(
      screen.getByText(/Choose another page from the menu/),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Go to Home" }),
    ).toBeInTheDocument()
  })

  it("sends a caller who does not operate the deployment Home from an operator-only route", async () => {
    mockMatchMedia(false)
    // A page someone may not use is not a refusal panel for them, and not the
    // deployment's "not available here" panel either, which would be false:
    // the deployment serves it, to someone else. Their address goes Home, as
    // an unknown one does.
    await renderShell(bootstrap(), {
      url: "/admin/accounts",
      operator: false,
    })

    expect(await screen.findByText("HOME PAGE")).toBeInTheDocument()
    expect(screen.queryByText("PAGE CONTENT")).toBeNull()
    expect(screen.queryByText("Accounts is not available here")).toBeNull()
  })

  it("answers the operator-only route with the panel when the surface is absent", async () => {
    mockMatchMedia(false)
    // The other half of that split, and the reason the page may call
    // `useDeploymentAdminAccess()` with no surface gate of its own: the route
    // declares `surface: "admin"`, so a deployment that does not host it never
    // reaches the page at all, and the caller axis is never asked.
    await renderShell(bootstrap({ surfaces: ["models"] }), {
      url: "/admin/accounts",
      operator: true,
    })

    expect(
      await screen.findByText("Accounts is not available here"),
    ).toBeInTheDocument()
    expect(screen.queryByText("PAGE CONTENT")).toBeNull()
    expect(
      vi
        .mocked(globalThis.fetch)
        .mock.calls.some((call) =>
          String(call[0]).includes(`${API_ROOT}/admin/access`),
        ),
    ).toBe(false)
  })

  it("adds the operator-only row to the rail once the caller is known to be one", async () => {
    mockMatchMedia(false)
    // The case rides on one row, so it is only as good as that row's gate, and
    // a row losing its gate is a registry edit that says nothing about this
    // file. So the premise is asserted rather than assumed.
    expect(
      navItemForPath("/admin/hosted-providers")?.operatorOnly,
    ).toBeDefined()
    await renderShell(bootstrap(), { operator: true, url: "/settings" })

    expect(
      await within(
        screen.getByRole("navigation", { name: "Sidebar" }),
      ).findByRole("link", { name: "Hosted providers" }),
    ).toBeInTheDocument()
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
    // The selected treatment, asserted as the classes because they are what a
    // reader of the rail actually sees; `aria-current` is covered separately.
    // The fill is `bg-surface-subtle`, the louder of the two rungs; hover takes
    // the quieter `surface-alt`, so a transient state cannot out-shout a
    // permanent one. The left edge is the part hover can never borrow, which is why it is
    // asserted here rather than left to `rowStyles.test.ts`: this is the test
    // that proves exactly one row wears it.
    expect(members.className).toContain("bg-surface-subtle")
    expect(members.className).toContain("border-foreground")
    expect(parent.className).not.toContain("bg-surface-subtle")
    expect(parent.className).not.toContain("border-foreground")
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
    await renderShell(bootstrap(), { url: "/settings" })

    expect(await screen.findByText("PAGE CONTENT")).toBeInTheDocument()
  })

  it("leaves a path the registry does not declare alone", async () => {
    mockMatchMedia(false)
    // The account page is not a registered destination, so the registry has no
    // opinion on it and must not gate it away with the rest.
    await renderShell(bootstrap({ surfaces: [] }), {
      entitlements: { capabilities: [] },
      url: "/account",
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
    expect(screen.queryByRole("link", { name: "Sessions" })).toBeNull()
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
    // Awaited, not assumed: the row is gated on `canManage(context.data)`, so it
    // arrives with the membership context rather than with the menu.
    expect(
      await within(menu).findByRole("button", { name: "Create workspace" }),
    ).toBeInTheDocument()
  })

  it("withholds Create workspace from a role the server would refuse", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    // `POST /api/v1/workspaces` is owners and admins only, and the Workspaces page
    // gates its own create control on the same predicate. Offering it here would
    // hand a member the whole form and report the refusal as a 403 after they
    // had typed a name.
    await renderShell(bootstrap(), { context: { role: "member" } })

    await user.click(
      await screen.findByRole("button", { name: /^Switch workspace/ }),
    )
    const menu = await screen.findByRole("dialog")
    // The organization row first, and awaited: it is what says the membership
    // context landed, so the absence below is a refused row rather than a row
    // whose answer had not arrived yet. Both come from that one read.
    expect(
      await within(menu).findByText("Default Organization"),
    ).toBeInTheDocument()
    expect(
      within(menu).queryByRole("button", { name: "Create workspace" }),
    ).toBeNull()
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
    // Providers on the organization rail is the group a rail still has.
    await renderShell(bootstrap(), { url: "/organization/members" })
    const sidebar = screen.getByRole("navigation", { name: "Sidebar" })
    await within(sidebar).findByRole("button", { name: "Providers" })

    // Shut to start. The panel stays mounted so it has a height to animate from,
    // but it is `hidden` and `aria-hidden` while shut, so its rows are reachable
    // by neither tab nor a screen reader until the group opens.
    expect(
      within(sidebar).queryByRole("link", { name: "Web search" }),
    ).toBeNull()
    expect(
      within(sidebar).getByText("Web search").closest("[hidden]"),
    ).not.toBeNull()

    await user.click(within(sidebar).getByRole("button", { name: "Providers" }))
    expect(
      within(sidebar).getByRole("link", { name: "Web search" }),
    ).toBeInTheDocument()
    expect(
      within(sidebar).getByRole("link", { name: "Inference" }),
    ).toBeInTheDocument()
  })

  it("opens the group that holds the current route, without a click", async () => {
    mockMatchMedia(false)
    // Arriving by bookmark or shared URL should show where you are rather than
    // a shut group that hides it.
    await renderShell(bootstrap(), { url: "/organization/web-search-keys" })

    expect(
      await screen.findByRole("button", { name: "Providers", expanded: true }),
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Web search" })).toBeInTheDocument()
  })

  it("signs out from the account menu rather than the page header", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell()

    // The header carried this until the sidebar grew an account block.
    expect(screen.queryByRole("button", { name: "Log out" })).toBeNull()
    await user.click(screen.getByRole("button", { name: /^Account:/ }))
    expect(
      await screen.findByRole("button", { name: "Log out" }),
    ).toBeInTheDocument()
  })

  it("keeps the documentation reachable from the account menu, for the phone", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell()

    // The top bar's cluster is `hidden md:flex`, and this menu is what the
    // mobile drawer contains, so without this row the docs have no control
    // pointing at it below the breakpoint. The row carries `md:hidden` for the
    // mirror-image reason: above it, the top bar already answers.
    await user.click(screen.getByRole("button", { name: /^Account:/ }))
    const guideRow = await screen.findByRole("link", { name: "Documentation" })
    expect(guideRow).toHaveAttribute("href", "https://docs.otari.ai/")
    expect(guideRow).toHaveClass("md:hidden")
  })

  it("names the scope and the page, leaving out an organization there is one of", async () => {
    mockMatchMedia(false)
    await renderShell()

    // First boot provisions one organization and most deployments keep exactly
    // that, so naming it on every page is a segment that never disambiguates
    // anything; the scope switcher names the active one where it matters.
    const crumb = await screen.findByLabelText("Breadcrumb")
    expect(crumb).toHaveTextContent("Home")
    expect(crumb).not.toHaveTextContent(/organization/i)
  })

  it("names a nested destination after itself, not after its group", async () => {
    mockMatchMedia(false)
    // `navItemForPath` answers with the entry that gates a path, which for a
    // child is its parent; a breadcrumb wants the leaf.
    await renderShell(bootstrap(), { url: "/organization/web-search-keys" })

    const crumb = await screen.findByLabelText("Breadcrumb")
    expect(crumb).toHaveTextContent("Web search")
    expect(crumb).not.toHaveTextContent("Providers")
  })

  it("offers no way into the deployment rail to a caller who operates nothing there", async () => {
    mockMatchMedia(false)
    // The control and the rail agree because the shell resolves one through the
    // same predicate as the other: a non-operator sees no Deployment row, and
    // would find no rows if they reached it. Absent rather than disabled, since
    // `/admin/accounts` is gated `unlisted` precisely so the shell does not
    // admit it exists.
    await renderShell(bootstrap(), { operator: false })

    await screen.findByRole("button", { name: /^Account:/ })
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: /^Account:/ }))

    expect(
      await screen.findByRole("link", { name: "Account settings" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: "Deployment" })).toBeNull()
  })

  it("offers the Organization switch on the workspace rail and on neither other", async () => {
    mockMatchMedia(false)
    // Stated as a rule about the workspace rail rather than as "not the
    // organization one". Written as a negation, a third rail inherits the row by
    // being merely not-the-organization, and the deployment rail would carry a
    // way into a context it is a sibling of rather than a way out of itself.
    // Queried on the whole shell, not inside the Sidebar landmark: the switch
    // row sits in the rail's footer, below the nav rather than in it.
    const rail = () => screen.getByRole("navigation", { name: "Sidebar" })
    await renderShell(bootstrap(), { url: "/" })
    expect(
      await screen.findByRole("link", { name: "Organization" }),
    ).toBeInTheDocument()

    cleanup()
    await renderShell(bootstrap(), { url: "/settings", operator: true })
    await within(rail()).findByRole("link", { name: "Accounts" })
    expect(screen.queryByRole("link", { name: "Organization" })).toBeNull()

    cleanup()
    await renderShell(bootstrap(), { url: "/organization/members" })
    await within(rail()).findByRole("link", { name: "Members & roles" })
    expect(screen.queryByRole("link", { name: "Organization" })).toBeNull()
  })

  it("names the deployment as the scope of its own pages", async () => {
    mockMatchMedia(false)
    // The deployment has no name to show, unlike the other two scopes, which
    // are data. The word is the scope, and it is the same word the control that
    // reaches these pages carries.
    // With a workspace selected, so the scope crumb is one this case could
    // fail on: with none, "does not name a scope" is trivially true and the
    // assertion proves nothing.
    const inWorkspace = {
      workspace_memberships: [
        {
          name: "Default workspace",
          role: "owner",
          workspace_id: "44444444-4444-4444-4444-444444444444",
        },
      ],
    }
    await renderShell(bootstrap(), {
      url: "/settings",
      operator: true,
      context: inWorkspace,
    })

    const crumb = await screen.findByLabelText("Breadcrumb")
    expect(crumb).toHaveTextContent("Deployment")
    expect(crumb).toHaveTextContent("Settings")
    expect(crumb).not.toHaveTextContent(/workspace/i)
    expect(crumb).not.toHaveTextContent(/organization/i)
    // Its own rail, and the row is marked: these are ordinary rail rows now, so
    // the highlight comes from the path every other row is on rather than from
    // anything built here.
    const rail = screen.getByRole("navigation", { name: "Sidebar" })
    const row = await within(rail).findByRole("link", { name: "Settings" })
    expect(row).toHaveAttribute("aria-current", "page")
    expect(
      within(rail).getByRole("link", { name: "Accounts" }),
    ).toBeInTheDocument()
    // And neither of the other two rails is showing.
    expect(within(rail).queryByRole("link", { name: "Home" })).toBeNull()
    expect(
      within(rail).queryByRole("link", { name: "Members & roles" }),
    ).toBeNull()
  })

  it("still names the workspace on an ordinary page, which is what the case above turns off", async () => {
    mockMatchMedia(false)
    // The control for the assertion above. Same fixture, a page a rail does own,
    // and the scope is in the trail: without this, "the trail names no scope"
    // could pass because the harness never produces one.
    await renderShell(bootstrap(), {
      url: "/usage",
      context: {
        workspace_memberships: [
          {
            name: "Default workspace",
            role: "owner",
            workspace_id: "44444444-4444-4444-4444-444444444444",
          },
        ],
      },
    })

    expect(await screen.findByLabelText("Breadcrumb")).toHaveTextContent(
      "Default workspace",
    )
  })

  it("answers a page with no row with the panel when its own surface is gone", async () => {
    mockMatchMedia(false)
    // The workspace's guardrails open from Rules and are served by the tools
    // surface. A bookmark still reaches the route on a deployment without it,
    // and the entry `navItemForPath` answers with carries that surface.
    await renderShell(bootstrap({ surfaces: ["routing"] }), {
      url: "/tools/guardrails",
    })

    expect(
      await screen.findByText("Guardrails is not available here"),
    ).toBeInTheDocument()
    expect(screen.queryByText("PAGE CONTENT")).toBeNull()
  })

  it("publishes the rail's footprint to the stylesheet", async () => {
    mockMatchMedia(false)
    // Guards the wiring, not the geometry. `globals.css` turns `data-rail` into
    // `--rail-width`, which is the only thing keeping a `position: fixed`
    // overlay centered on the content rather than on the window; drop the
    // attribute and the variable falls back to `0px`, the overlay quietly
    // re-centers, and nothing fails. Whether the arithmetic is right is a
    // question about layout, which jsdom cannot answer: that is covered by the
    // Playwright case in `e2e/dashboard.spec.ts`.
    const { container } = await renderShell()

    expect(container.querySelector("[data-rail]")).toHaveAttribute(
      "data-rail",
      "expanded",
    )
  })

  it("says collapsed and drawer apart from expanded", async () => {
    mockMatchMedia(false)
    window.localStorage.setItem("otari.dashboard.sidebarCollapsed", "1")
    const { container, unmount } = await renderShell()
    expect(container.querySelector("[data-rail]")).toHaveAttribute(
      "data-rail",
      "collapsed",
    )
    unmount()

    // Below `md` the rail is off-canvas and costs the content nothing, which is
    // neither of the other two. It keeps saying so with a collapsed preference
    // still stored, which is the case that was wrong when this attribute had
    // only two values.
    mockMatchMedia(true)
    const drawer = await renderShell()
    expect(drawer.container.querySelector("[data-rail]")).toHaveAttribute(
      "data-rail",
      "drawer",
    )
  })

  it("reaches a group's nested destinations from the collapsed rail", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    // Collapsed, a group is a flyout rather than a link to its own page, so
    // each of its rows keeps an affordance.
    window.localStorage.setItem("otari.dashboard.sidebarCollapsed", "1")
    await renderShell(bootstrap(), { url: "/organization/members" })
    const sidebar = screen.getByRole("navigation", { name: "Sidebar" })
    await within(sidebar).findByRole("button", { name: "Providers" })

    expect(
      within(sidebar).queryByRole("link", { name: "Providers" }),
    ).toBeNull()
    await user.click(within(sidebar).getByRole("button", { name: "Providers" }))

    expect(
      await screen.findByRole("link", { name: "Web search" }),
    ).toHaveAttribute("href", "/organization/web-search-keys")
    expect(screen.getByRole("link", { name: "Inference" })).toHaveAttribute(
      "href",
      "/organization/provider-keys",
    )
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

describe("the telemetry the sidebar records", () => {
  beforeEach(() => {
    resetTelemetrySpy()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    window.localStorage.clear()
  })

  it("records a move to another destination", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell(bootstrap(), { url: "/organization" })

    await user.click(await screen.findByRole("link", { name: "Guardrails" }))

    expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.TAB_CHANGED, {
      tab_name: "guardrails",
      context: "organization_settings",
    })
  })

  it("names the rail the destination belongs to, not the one on screen", async () => {
    // The organization rail is a context switch rather than a section, and the
    // row that crosses into it is on the workspace rail. Reading the context
    // from the destination is what puts that move on the rail it landed on.
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell(bootstrap(), { url: "/organization/members" })

    await user.click(await screen.findByRole("button", { name: "Providers" }))
    await user.click(await screen.findByRole("link", { name: "Inference" }))

    expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.TAB_CHANGED, {
      tab_name: "provider-keys",
      context: "organization_settings",
    })
  })

  it("does not record a click on the row you are already on", async () => {
    // Counting those would inflate whichever page people sit on longest.
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell(bootstrap(), { url: "/organization/guardrails" })

    await user.click(await screen.findByRole("link", { name: "Guardrails" }))

    expect(recordEvent).not.toHaveBeenCalled()
  })

  it("names the index rather than reporting it as an empty string", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell(bootstrap(), { url: "/models" })

    await user.click(screen.getByRole("link", { name: "Home" }))

    expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.TAB_CHANGED, {
      tab_name: "index",
      context: "workspace_sidebar",
    })
  })

  it("records a deployment page under its own context, not the workspace one", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell(bootstrap(), { url: "/settings", operator: true })

    const rail = screen.getByRole("navigation", { name: "Sidebar" })
    await user.click(
      await within(rail).findByRole("link", { name: "Accounts" }),
    )

    expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.TAB_CHANGED, {
      tab_name: "accounts",
      context: "deployment_settings",
    })
  })

  it("records entering the organization rail from the footer row", async () => {
    // The row that crosses into the other rail does not go through
    // `NavRowLink`, so tracking the rows alone left every entry to and exit
    // from that rail unrecorded while every row inside it was recorded.
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell()

    // `findByRole`, as the sibling test above does: this row appears only once
    // the organization context resolves, since it gates on the caller's role.
    await user.click(await screen.findByRole("link", { name: "Organization" }))

    expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.TAB_CHANGED, {
      tab_name: "members",
      context: "organization_settings",
    })
  })

  it("records leaving it by the way back", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell(bootstrap(), { url: "/organization/members" })

    await user.click(await screen.findByRole("link", { name: /^Back to/ }))

    expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.TAB_CHANGED, {
      tab_name: "index",
      context: "workspace_sidebar",
    })
  })
})

type Role = "owner" | "admin" | "member" | "viewer"

/** A caller of `role` in the Platform workspace who operates nothing. */
function asRole(role: Role) {
  return {
    operator: false,
    context: {
      role,
      workspace_memberships: [
        { workspace_id: "ws-platform", name: "Platform", role },
      ],
    },
  }
}

/** Every link in the rail, by href, a group's hidden rows included. */
function railLinks(): string[] {
  return within(screen.getByRole("navigation", { name: "Sidebar" }))
    .getAllByRole("link", { hidden: true })
    .map((link) => link.getAttribute("href") ?? "")
}

const WORKSPACE_RAIL = [
  "/",
  "/sessions",
  "/people",
  "/rules",
  "/models",
  "/tools",
  "/keys",
]

describe("the workspace rail, per role", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    window.localStorage.clear()
  })

  for (const role of ["owner", "admin"] as const) {
    it(`gives an ${role} the rail and the Organization row`, async () => {
      mockMatchMedia(false)
      await renderShell(bootstrap(), asRole(role))
      const organization = await screen.findByRole("link", {
        name: "Organization",
      })
      expect(organization).toHaveAttribute("href", "/organization/members")
      expect(railLinks()).toEqual(WORKSPACE_RAIL)
    })
  }

  for (const role of ["member", "viewer"] as const) {
    it(`gives a ${role} the same rail without the Organization row`, async () => {
      mockMatchMedia(false)
      await renderShell(bootstrap(), asRole(role))
      await screen.findByRole("link", { name: "API keys" })
      expect(railLinks()).toEqual(WORKSPACE_RAIL)
      expect(screen.queryByRole("link", { name: "Organization" })).toBeNull()
      expect(screen.queryByRole("button", { name: "Organization" })).toBeNull()
    })
  }

  it("offers a member no Organization row on a phone either", async () => {
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell(bootstrap(), asRole("member"))
    await user.click(screen.getByRole("button", { name: "Open navigation" }))
    await screen.findByRole("link", { name: "API keys" })
    expect(screen.queryByRole("button", { name: "Organization" })).toBeNull()
  })

  it("opens the organization rail from the footer row on desktop", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell(bootstrap(), asRole("owner"))
    await user.click(await screen.findByRole("link", { name: "Organization" }))
    await screen.findByRole("link", { name: "Members & roles" })
    expect(railLinks()).toEqual([
      "/workspaces",
      "/organization/members",
      "/organization/domains",
      "/organization/sso",
      "/budgets",
      "/organization/provider-keys",
      "/organization/web-search-keys",
      "/organization/guardrails",
      "/organization",
    ])
  })

  // The workspace's guardrails open to owners and admins only, so that page is
  // checked as an owner.
  for (const [path, page, role] of [
    ["/routing", "Routing", "member"],
    ["/routing/coding", "Routing", "member"],
    ["/tools/guardrails", "Guardrails", "owner"],
  ] as const) {
    it(`selects Rules on ${path} and names it in the breadcrumb`, async () => {
      mockMatchMedia(false)
      await renderShell(bootstrap(), { url: path, ...asRole(role) })
      const sidebar = screen.getByRole("navigation", { name: "Sidebar" })
      const rules = await within(sidebar).findByRole("link", { name: "Rules" })
      expect(rules).toHaveAttribute("aria-current", "page")
      expect(rules.className).toBe(
        `${navRowClass({ isActive: true, isCollapsed: false })} w-full`,
      )
      const tools = within(sidebar).getByRole("link", { name: "Tools" })
      expect(tools).not.toHaveAttribute("aria-current")
      expect(tools.className).toBe(
        `${navRowClass({ isActive: false, isCollapsed: false })} w-full`,
      )
      const trail = within(
        screen.getByRole("navigation", { name: "Breadcrumb" }),
      ).getAllByRole("listitem")
      expect(
        trail
          .map((item) => item.textContent)
          .filter((text) => text !== "/")
          .slice(-2),
      ).toEqual(["Rules", page])
    })
  }

  for (const [path, row] of [
    ["/members", "People"],
    ["/requests", "Sessions"],
    ["/tools/web-search", "Tools"],
  ] as const) {
    it(`announces ${row} as the current row on ${path}, and no other`, async () => {
      mockMatchMedia(false)
      await renderShell(bootstrap(), { url: path, ...asRole("member") })
      const sidebar = screen.getByRole("navigation", { name: "Sidebar" })
      expect(
        await within(sidebar).findByRole("link", { name: row }),
      ).toHaveAttribute("aria-current", "page")
      expect(
        within(sidebar)
          .getAllByRole("link")
          .filter((link) => link.getAttribute("aria-current") === "page"),
      ).toHaveLength(1)
    })
  }

  it("ends the breadcrumb on the record a page names, after a link back to where it sits", async () => {
    function ServerPage() {
      useBreadcrumbLeaf("notion")
      return <div>PAGE CONTENT</div>
    }
    mockMatchMedia(false)
    await renderShell(bootstrap(), {
      url: "/tools/mcp-servers/srv-1",
      page: <ServerPage />,
      ...asRole("member"),
    })
    const trail = screen.getByRole("navigation", { name: "Breadcrumb" })

    expect(await within(trail).findByText("notion")).toHaveAttribute(
      "aria-current",
      "page",
    )
    expect(within(trail).getByRole("link", { name: "Tools" })).toHaveAttribute(
      "href",
      "/tools",
    )
  })
})

/** The openings so far; the shell's page views share the same recorder. */
function openings() {
  return recordEvent.mock.calls.filter(
    ([event]) => event === TELEMETRY_EVENTS.CREATE_API_KEY_OPENED,
  )
}

describe("the rail's Create API key", () => {
  // The dialog is its own lazy chunk. Its first import compiles a large module
  // graph, which on a loaded runner outlasts `findByRole`'s one second, so it
  // is loaded before any test opens it.
  beforeAll(async () => {
    await import("@/features/keys/CreateKeyDialog")
  })

  beforeEach(() => {
    resetTelemetrySpy()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    window.localStorage.clear()
  })

  it("is named on hover when the rail is collapsed", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell(bootstrap(), asRole("owner"))
    await user.click(
      await screen.findByRole("button", { name: "Collapse sidebar" }),
    )
    const create = await screen.findByRole("button", { name: "Create API key" })
    expect(create).not.toHaveTextContent("Create API key")
    await user.hover(create)
    expect(await screen.findByRole("tooltip")).toHaveTextContent(
      "Create API key",
    )
  })

  it("records that the dialog was opened from the sidebar, once per opening", async () => {
    mockMatchMedia(false)
    const user = userEvent.setup()
    await renderShell(bootstrap(), asRole("member"))
    await user.click(
      await screen.findByRole("button", { name: "Create API key" }),
    )
    const dialog = await screen.findByRole("dialog", { name: "Create API key" })
    expect(openings()).toEqual([
      [TELEMETRY_EVENTS.CREATE_API_KEY_OPENED, { entry_point: "sidebar" }],
    ])
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await user.click(
      await screen.findByRole("button", { name: "Create API key" }),
    )
    await waitFor(() => expect(openings()).toHaveLength(2))
  })

  // Twice: after its first open the dialog stays mounted, and takes focus
  // before the drawer has closed around the button that opened it.
  it("hands focus back to the drawer's toggle on a phone", async () => {
    // A browser refuses focus inside an `inert` subtree, the closed drawer
    // included; jsdom does not.
    const focus = HTMLElement.prototype.focus
    vi.spyOn(HTMLElement.prototype, "focus").mockImplementation(function (
      this: HTMLElement,
      options?: FocusOptions,
    ) {
      if (!this.closest("[inert]")) focus.call(this, options)
    })
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell(bootstrap(), asRole("owner"))
    for (const _ of [1, 2]) {
      await user.click(screen.getByRole("button", { name: "Open navigation" }))
      await user.click(
        await screen.findByRole("button", { name: "Create API key" }),
      )
      const dialog = await screen.findByRole("dialog", {
        name: "Create API key",
      })
      await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: "Open navigation" }),
        ).toHaveFocus(),
      )
    }
  })

  it("names the drawer's toggle only for an opening from the drawer", async () => {
    // An opener that is gone by the time the dialog closes, as the key-created
    // step leaves it: focus then goes to whatever target the opening named.
    function PageOpener() {
      const openCreateKey = useOpenCreateKey()
      const [isShown, setShown] = useState(true)
      return isShown ? (
        <button
          type="button"
          onClick={() => {
            openCreateKey("api_keys_page")
            setShown(false)
          }}
        >
          New key here
        </button>
      ) : null
    }
    mockMatchMedia(true)
    const user = userEvent.setup()
    await renderShell(bootstrap(), {
      ...asRole("owner"),
      page: <PageOpener />,
    })
    await user.click(screen.getByRole("button", { name: "New key here" }))
    const dialog = await screen.findByRole("dialog", { name: "Create API key" })
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Create API key" }),
      ).toBeNull(),
    )
    // A frame for the dialog's own restore to have run.
    await new Promise((resolve) => requestAnimationFrame(resolve))
    expect(
      screen.getByRole("button", { name: "Open navigation" }),
    ).not.toHaveFocus()
  })

  it("is not offered to a viewer, who is refused a key", async () => {
    mockMatchMedia(false)
    await renderShell(bootstrap(), asRole("viewer"))
    // A rail row first: it arrives with the caller's role, so the absence
    // below is a refusal rather than an answer still coming.
    await screen.findByRole("link", { name: "API keys" })
    expect(screen.queryByRole("button", { name: "Create API key" })).toBeNull()
  })

  it("leads the workspace rail only", async () => {
    mockMatchMedia(false)
    await renderShell(bootstrap(), {
      url: "/organization/members",
      ...asRole("owner"),
    })
    await screen.findByRole("link", { name: "Members & roles" })
    expect(screen.queryByRole("button", { name: "Create API key" })).toBeNull()
  })
})

describe("the team budget at the rail's foot", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    window.localStorage.clear()
  })

  // A period still running, so the spend it reports is this period's.
  const LIMIT = [
    organizationSpendCeiling({
      max_budget: 250,
      current_spend: 50,
      period_end: "2999-01-01T00:00:00Z",
    }),
  ]

  it("shows a manager the organization's limit, linking Home", async () => {
    mockMatchMedia(false)
    await renderShell(bootstrap(), {
      url: "/keys",
      spendCeilings: LIMIT,
      ...asRole("owner"),
    })
    const budget = await screen.findByRole("link", { name: /Team budget/ })
    expect(budget).toHaveAttribute("href", "/")
    expect(budget).toHaveTextContent("20%")
  })

  it("leaves it out of the collapsed rail", async () => {
    mockMatchMedia(false)
    window.localStorage.setItem("otari.dashboard.sidebarCollapsed", "1")
    await renderShell(bootstrap(), {
      url: "/keys",
      spendCeilings: LIMIT,
      ...asRole("owner"),
    })
    await screen.findByRole("link", { name: "API keys" })
    expect(screen.queryByText("Team budget")).toBeNull()
    // And it is the collapse that withholds it.
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Expand sidebar" }))
    expect(
      await screen.findByRole("link", { name: /Team budget/ }),
    ).toBeInTheDocument()
  })

  it("shows a member nothing, whatever the limit", async () => {
    mockMatchMedia(false)
    await renderShell(bootstrap(), {
      url: "/keys",
      spendCeilings: LIMIT,
      ...asRole("member"),
    })
    await screen.findByRole("link", { name: "API keys" })
    expect(screen.queryByText("Team budget")).toBeNull()
  })
})
