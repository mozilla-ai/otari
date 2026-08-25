import { Button, Disclosure, Popover } from "@heroui/react"
import { Link, Outlet, useLocation } from "@tanstack/react-router"
import { clsx } from "clsx"
import type { MouseEvent as ReactMouseEvent } from "react"
import { useCallback, useEffect, useRef, useState } from "react"
import type { IconType } from "react-icons"
import {
  FiArrowLeft,
  FiChevronDown,
  FiChevronRight,
  FiMenu,
  FiSettings,
  FiSidebar,
  FiX,
} from "react-icons/fi"
import { ConnectionStatus } from "@/app/ConnectionStatus"
import { AccountMenu } from "@/app/nav/AccountMenu"
import { Breadcrumbs } from "@/app/nav/Breadcrumbs"
import { lastLocation, rememberLocation } from "@/app/nav/navigationHistory"
import {
  isPathVisible,
  NAV_SECTIONS,
  navContextForPath,
  navItemForPath,
  navLabelForPath,
  ORG_NAV_SECTIONS,
  visibleNavSections,
} from "@/app/nav/registry"
import {
  NAV_ICON_CLASS,
  NAV_SECTION_HEADING_CLASS,
  navIndicatorClass,
  navRowClass,
} from "@/app/nav/rowStyles"
import { TopBarActions } from "@/app/nav/TopBarActions"
import type { NavItem, NavPath } from "@/app/nav/types"
import {
  useNavVisibility,
  useRouteVisibility,
  useSurfaceVisibility,
} from "@/app/nav/useNavVisibility"
import { WorkspaceSwitcher } from "@/app/nav/WorkspaceSwitcher"
import { EntitlementResolver } from "@/app/overlayEntitlementResolver"
import { PostSignInGate } from "@/app/overlayPostSignInGate"
import { PendingPage } from "@/app/PendingPage"
import { TelemetryIdentity } from "@/app/TelemetryIdentity"
import { UpdatePrompt } from "@/app/UpdatePrompt"
import { PricingWarning } from "@/features/models/PricingWarning"
import { canManage } from "@/features/organization/roles"
import { useOrganizationContext } from "@/shared/api/hooks"
import { EmptyState } from "@/shared/components/ui"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useEntitlements } from "@/shared/hooks/useEntitlements"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

// One width, not a range. The rail used to be draggable between 200 and 480px
// and to remember where it was left; it is now the design's 264px, or 72px
// collapsed, which is also what `otari-ai/frontend`'s shell is fixed at
// (`w-[16.5rem]` / `w-[4.5rem]`). Collapse is the only size control, and the
// only one either rail needs: the rows inside are a fixed layout, so the width
// between those two values buys a longer truncation point and nothing else,
// while a drag handle on the page's main seam is a thing to catch by accident.
const SIDEBAR_WIDTH = "w-[16.5rem]"
const COLLAPSED_SIDEBAR_WIDTH = "w-[4.5rem]"
const SIDEBAR_COLLAPSED_KEY = "otari.dashboard.sidebarCollapsed"

// Below this width the sidebar's fixed footprint squashes page content, so it
// switches to an off-canvas drawer toggled from the header. Matches Tailwind's
// `md` breakpoint (the classes that hide the trigger and drawer chrome use `md:`).
const MOBILE_QUERY = "(max-width: 767px)"

function readIsMobile(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function")
    return false
  return window.matchMedia(MOBILE_QUERY).matches
}

function readStoredCollapsed(): boolean {
  if (typeof window === "undefined") return false
  try {
    return window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1"
  } catch {
    return false
  }
}

/**
 * What a destination is called in `TAB_CHANGED`.
 *
 * The last segment of the path, which is the derivation
 * `otari-ai/frontend/src/app/SidebarItems.tsx` already uses, so a page that
 * exists in both rails is reported under one name. The index has no segment to
 * take and is named rather than reported as an empty string.
 */
function tabNameForPath(to: NavPath): string {
  return to.split("/").filter(Boolean).pop() ?? "index"
}

/**
 * Which rail a `TAB_CHANGED` belongs to, in the platform's own vocabulary.
 *
 * `otari-ai/frontend/src/app/nav/registry.ts` sends `"workspace_sidebar"` and
 * `"organization_settings"` for this property, so those are the values sent
 * here. A value used as a breakdown is as much a shared vocabulary as the event
 * name over it: `context: "workspace"` beside a historical
 * `context: "workspace_sidebar"` splits one funnel exactly the way a renamed
 * event would.
 */
function navTrackContext(to: NavPath): string {
  return navContextForPath(to) === "organization"
    ? "organization_settings"
    : "workspace_sidebar"
}

/**
 * Records a move between sidebar destinations.
 *
 * A hook rather than a call inside `NavRowLink`, because the rows inside a rail
 * are not the only way to move within the sidebar: the two footer rows that
 * cross between the rails navigate to a registry destination without going
 * through `NavRowLink` at all, and tracking only the rows meant every entry to
 * and exit from the organization rail went unrecorded while every row inside it
 * was recorded.
 */
function useRecordNavigation(): (to: NavPath, isActive: boolean) => void {
  const { recordEvent } = useTelemetry()

  return (to, isActive) => {
    // Only a move: clicking the row you are already on is not a navigation, and
    // counting it would inflate whichever page people sit on longest.
    if (isActive) {
      return
    }
    // The context is read from the destination rather than from the rail that
    // was showing, so a row that crosses between the two is recorded where it
    // landed.
    recordEvent(TELEMETRY_EVENTS.TAB_CHANGED, {
      tab_name: tabNameForPath(to),
      context: navTrackContext(to),
    })
  }
}

/**
 * One row of the rail, pointing at one destination.
 *
 * Shared by the leaves, a group's children, and a group that has collapsed to a
 * single child, so the three cannot drift: they are the same row with a
 * different indent and a different label.
 *
 * Collapsed, the label survives as the accessible name and as the tooltip,
 * because the visible text is what a sighted reader loses and the only thing an
 * assistive one had.
 */
function NavRowLink({
  to,
  label,
  icon: Icon,
  isActive,
  collapsed,
  nested,
  onNavigate,
}: {
  to: NavPath
  label: string
  icon?: IconType
  isActive: boolean
  collapsed?: boolean
  nested?: boolean
  onNavigate: () => void
}) {
  const recordNavigation = useRecordNavigation()

  return (
    <Link
      to={to}
      // Exact, because the default is a prefix match: on /organization/members
      // that leaves `aria-current` on "Organization" as well as on the child.
      activeOptions={{ exact: true }}
      onClick={() => {
        recordNavigation(to, isActive)
        onNavigate()
      }}
      className={navRowClass({ isActive, collapsed, nested })}
      aria-label={collapsed ? label : undefined}
      title={collapsed ? label : undefined}
    >
      {/* A nested row draws no glyph: the indent is what marks it as one, and
          repeating the parent's lane would undo that. The flyout a collapsed
          group opens is the exception, and it is not nested: those rows hang in
          a menu with no indent to read. */}
      {Icon && !nested ? (
        <Icon className={NAV_ICON_CLASS} aria-hidden="true" />
      ) : null}
      {collapsed ? null : (
        <span className="min-w-0 flex-1 truncate">{label}</span>
      )}
    </Link>
  )
}

/**
 * A sidebar entry with destinations nested under it, drawn the way the design
 * draws Routing and Tools: a row that expands rather than navigates, and
 * indented children below it.
 *
 * Open when the current route is one of its children, so arriving by URL shows
 * where you are rather than a collapsed group. Held in state after that, so
 * closing it stays closed while you read the page it opened.
 *
 * Three shapes, and which one it takes is decided by how many children survive
 * gating and whether the rail is collapsed:
 *
 * **One child** and it is not a group at all, but that child wearing the
 * parent's name and glyph. A disclosure that opens onto a single row asks for a
 * click to tell you nothing, and this is reachable: a deployment without the
 * tools surface leaves Routing holding only Policies.
 *
 * **Collapsed** and it is an icon that opens a flyout of the children. This is
 * the case the rail used to lose: the parent linked straight to its own page, so
 * Guardrails, Web search and Code execution had no collapsed affordance at all
 * and a bookmark was the only way back to them.
 */
function NavGroup({
  item,
  currentPath,
  onNavigate,
  isVisible,
  collapsed,
}: {
  item: NavItem
  currentPath: string
  onNavigate: () => void
  isVisible: (item: NavItem) => boolean
  collapsed: boolean
}) {
  // A child declaring its own surface is gated on it. Without this the field
  // was decoration: Guardrails is grouped under Routing but served by the tools
  // surface, so a deployment without that surface kept the link and landed on
  // the "not available here" panel.
  const children = (item.children ?? []).filter((child) =>
    child.surface ? isVisible({ ...item, surface: child.surface }) : true,
  )
  const holdsCurrent = children.some((child) => child.to === currentPath)
  const [open, setOpen] = useState(holdsCurrent)
  const [flyoutOpen, setFlyoutOpen] = useState(false)
  // Follows the route when navigation lands inside the group from elsewhere
  // (a link on a page, a bookmark), without fighting a manual close.
  const [lastHeld, setLastHeld] = useState(holdsCurrent)
  if (holdsCurrent !== lastHeld) {
    setLastHeld(holdsCurrent)
    if (holdsCurrent) setOpen(true)
  }

  const only = children.length === 1 ? children[0] : undefined
  if (only) {
    return (
      <NavRowLink
        to={only.to}
        label={item.label}
        icon={item.icon}
        isActive={currentPath === only.to}
        collapsed={collapsed}
        onNavigate={onNavigate}
      />
    )
  }

  if (collapsed) {
    return (
      <Popover isOpen={flyoutOpen} onOpenChange={setFlyoutOpen}>
        {/* HeroUI's Button, not a plain one: the popover wires its trigger
            through react-aria, and a bare <button> leaves it unopenable. */}
        <Button
          variant="ghost"
          aria-label={item.label}
          className={`${navRowClass({ isActive: holdsCurrent, collapsed: true })} w-auto!`}
        >
          <item.icon className={NAV_ICON_CLASS} aria-hidden="true" />
        </Button>
        <Popover.Content placement="right top">
          <Popover.Dialog
            aria-label={item.label}
            className="flex w-56 flex-col gap-0.5"
          >
            {/* Named, because the icon that opened this is the only other thing
                saying which group these belong to, and it is off to the side. */}
            <p className="flex min-h-8 items-center px-3 text-overline">
              {item.label}
            </p>
            {children.map((child) => (
              <NavRowLink
                key={child.to}
                to={child.to}
                label={child.label}
                icon={child.icon}
                isActive={currentPath === child.to}
                onNavigate={() => {
                  setFlyoutOpen(false)
                  onNavigate()
                }}
              />
            ))}
          </Popover.Dialog>
        </Popover.Content>
      </Popover>
    )
  }

  return (
    // HeroUI's own disclosure, which is what `otari-ai/frontend`'s
    // `NavigationBranch` opens its branches with, so the two rails expand the
    // same way. React Aria measures the panel and writes
    // `--disclosure-panel-height` onto it, which is the variable HeroUI's
    // `.disclosure__content` transitions (200ms height on `--ease-out-quad`,
    // 200ms opacity on `--ease-out`), so the rows slide open *and* shut rather
    // than appearing and vanishing.
    //
    // The gap belongs to the expanded state only: the panel stays mounted at
    // zero height while shut, so an unconditional gap would leave a stray 2px
    // hanging under every closed group.
    <Disclosure
      isExpanded={open}
      onExpandedChange={setOpen}
      className={clsx("flex flex-col", open && "gap-0.5")}
    >
      <Disclosure.Heading>
        <Disclosure.Trigger className={navRowClass({ isActive: holdsCurrent })}>
          <item.icon className={NAV_ICON_CLASS} aria-hidden="true" />
          <span className="min-w-0 flex-1 truncate text-left">
            {item.label}
          </span>
          <FiChevronDown
            aria-hidden="true"
            className={navIndicatorClass({ open })}
          />
        </Disclosure.Trigger>
      </Disclosure.Heading>
      {/* The rows sit straight in the panel rather than in a `Disclosure.Body`,
          which wraps its children in a div carrying 0.5rem of padding that no
          className can reach: it would inset these rows from the lane their
          siblings share and clip their fill short at both edges. */}
      {/* `focus-within:overflow-visible` because HeroUI's `.disclosure__content`
          is `overflow: clip` (that is what keeps the rows out of sight while the
          panel animates), and a nested row is exactly the panel's width, so the
          2px its focus ring paints outside its border box was being cut off on
          every edge. The clip is only needed while the panel is moving, and it
          cannot be moving while something inside it holds focus: closing the
          group puts focus on the trigger, which is outside the panel. */}
      <Disclosure.Content className="flex flex-col focus-within:overflow-visible">
        {children.map((child) => (
          <NavRowLink
            key={child.to}
            to={child.to}
            label={child.label}
            icon={child.icon}
            isActive={currentPath === child.to}
            nested
            onNavigate={onNavigate}
          />
        ))}
      </Disclosure.Content>
    </Disclosure>
  )
}

/**
 * The shell, with the entitlement axis resolved above it and a post-sign-in step
 * able to sit in front of it.
 *
 * Two components rather than one because a provider is invisible to the
 * component that renders it: `AppShellChrome` reads the axis through
 * `useNavVisibility`, so a resolver mounted inside its body would answer every
 * consumer below it and none of the ones that decide the rail. Mounted here it
 * sits above both halves the axis gates, the navigation and the `<Outlet>` the
 * routes render into, which is what the seam is for. The base default renders
 * its children unchanged, so this build's shell is the shell it was.
 *
 * Here rather than in `__root.tsx` or beside `DeploymentProvider` in `App.tsx`,
 * because everything that reads the axis is inside the shell and because the
 * resolver a superset build swaps in issues a query: it belongs behind the auth
 * gate `App.tsx` puts the router behind, not in front of it. So the surfaces
 * that render instead of the shell (the sign-in screen, the public auth pages,
 * the hybrid landing) are deliberately outside it: none of them gates on a
 * capability, and a visitor without a session has nothing to resolve one from.
 *
 * `PostSignInGate` is the seam for anything that has to be shown once, in front
 * of the whole app, after a session exists (otari#789): a hosted signup's
 * profile questions, in the build that has a signup. It is mounted here rather
 * than anywhere else because its position *is* its contract, all three
 * boundaries of it, and the module docstring is where that lives. The base
 * default renders its children unchanged, so this build shows no step at all.
 */
export function AppShell() {
  return (
    <EntitlementResolver>
      <PostSignInGate>
        <AppShellChrome />
      </PostSignInGate>
    </EntitlementResolver>
  )
}

function AppShellChrome() {
  // Navigation is data: the shell renders whatever the registry declares and
  // decides visibility from the deployment and the entitlements,
  // rather than each page asking what it is running against.
  const isVisible = useNavVisibility()
  const recordNavigation = useRecordNavigation()
  const { pathname } = useLocation()
  // A gated-off destination is still reachable by bookmark or shared URL, so the
  // shell answers those with a panel instead of a page whose every request the
  // server would refuse. An unregistered path (the guide, the 404 splat) has no
  // entry and is never gated.
  const currentItem = navItemForPath(pathname)
  // Through the registry's predicate rather than `isVisible(currentItem)` here,
  // so this and the rail memory cannot answer "is this destination served"
  // differently: whichever way the nested case resolves, both read it from one
  // place.
  //
  // The route predicate, not the rail one: they part on the caller axis alone,
  // and both reasons are in `useRouteVisibility`. Short version: that axis is a
  // query, so gating the route on it would show the panel below to a real
  // operator until it answered, and a caller who is not one is owed the page's
  // own words rather than a claim that the deployment does not serve it.
  const isRouteVisible = useRouteVisibility()
  const routeIsGatedOff = !isPathVisible(pathname, isRouteVisible)
  // Kept beside that answer rather than folded into it, because the two are
  // different claims: "gated off" decides whether the page renders, and this
  // decides whether the shell may yet say *why*. The panel below asserts that
  // this deployment does not serve the page, which is not something the shell
  // knows while the entitlement axis is still resolving. Always settled in this
  // build, where `useEntitlements` answers from a constant, so this is a branch
  // a superset build's asynchronous resolver takes and the base never does.
  // `EntitlementGate` has had a `loading` state for precisely this reason since it
  // was written; the rail needs none, because a row that appears late is not a
  // row that told anyone it was missing.
  const { isLoading: entitlementsResolving } = useEntitlements()
  // Narrowed to the entitlement axis, because only that one can still be
  // resolving. A route gated off because this deployment does not host the
  // surface was answered by the bootstrap before the page rendered, so waiting
  // on the entitlement query would hold back a panel that is already correct and
  // that the query cannot change. Asking the surface half separately is what
  // distinguishes the two, since `isPathVisible` composes them and reports only
  // that something hid the route.
  const hostsRouteSurface = useSurfaceVisibility()
  const answerIsStillComing =
    routeIsGatedOff &&
    entitlementsResolving &&
    isPathVisible(pathname, hostsRouteSurface)
  // Which of the two sidebars this path belongs under. The organization context
  // is a separate rail reached from the footer, not a section inside the
  // workspace one, so the two never render together.
  const navContext = navContextForPath(pathname)
  const inOrganization = navContext === "organization"
  const organization = useOrganizationContext()
  const { selected: selectedWorkspace } = useSelectedWorkspace()
  // Always true in a standalone deployment, where the one session is the local
  // operator and it owns the organization the gateway provisioned for itself.
  // Written anyway because it becomes load-bearing the moment per-user sign-in
  // lands (otari-ai#1716), and because an overlay build can already be reached
  // by someone who is not an admin.
  // Fails open when the context errors rather than resolving false: Users,
  // budgets and settings are reachable only through this entry, the routes still
  // work by URL, and the server authorizes every request behind it regardless.
  // Hiding the way there because one query failed strands three destinations.
  const managesOrganization =
    canManage(organization.data) || organization.isError

  // Both rails remember where you were, so the controls that cross between them
  // return you rather than resetting you. Recorded here rather than on those
  // controls' clicks, so leaving a rail by a link on a page or by a bookmark
  // updates the memory too.
  // Both halves take the visibility predicate: a gated-off destination is
  // registered and reachable by URL, so without it the memory would both record
  // the visit that landed on "not available here" and resume onto it.
  useEffect(() => {
    rememberLocation(pathname, isVisible)
  }, [pathname, isVisible])
  const organizationLanding = lastLocation("organization", isVisible)
  const workspaceLanding = lastLocation("workspace", isVisible)
  // Named once: the same string is the row's visible text, its accessible name
  // when collapsed, and its tooltip, and three copies of it is three chances for
  // the name a screen reader hears to drift from the one on screen.
  const backLabel = `Back to ${selectedWorkspace?.name ?? "workspace"}`

  const asideRef = useRef<HTMLElement>(null)
  const mainRef = useRef<HTMLElement>(null)
  const toggleRef = useRef<HTMLButtonElement>(null)
  const orgNavTriggerRef = useRef<HTMLButtonElement>(null)
  const orgNavBackRef = useRef<HTMLButtonElement>(null)
  const restoreSidebarFocusRef = useRef(false)
  const [collapsed, setCollapsed] = useState<boolean>(readStoredCollapsed)
  const [isMobile, setIsMobile] = useState<boolean>(readIsMobile)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  // The organization rail, opened as a level *inside* the drawer rather than by
  // going to a page. Mobile only, and it exists because the two rails are a
  // context switch rather than a section: on the desk the footer row can change
  // the whole rail and leave you looking at it, while on a phone the same row
  // navigated and the drawer closed over the result, so the rail it opened was
  // never a thing you got to read. Here the row opens that rail in place, and
  // choosing a destination in it is what dismisses the drawer.
  const [mobileOrgNavOpen, setMobileOrgNavOpen] = useState(false)

  // Every control that leaves the drawer goes through this rather than lowering
  // the one flag it knows about, so the submenu cannot outlive the drawer that
  // framed it: reopening the menu from a workspace page would otherwise land
  // back on the organization rows the last tap left showing.
  const closeMobileNav = useCallback(() => {
    setMobileNavOpen(false)
    setMobileOrgNavOpen(false)
  }, [])

  // Which of the two rails is drawn. The route decides it, except on mobile,
  // where the drawer can be one level down inside the organization rail while
  // the page behind it is still a workspace page.
  const showOrganizationRail = inOrganization || (isMobile && mobileOrgNavOpen)
  // Filtered before it is indexed, so the divider and top margin below key off
  // the first *rendered* section rather than the first registered one.
  const visibleSections = visibleNavSections(
    showOrganizationRail ? ORG_NAV_SECTIONS : NAV_SECTIONS,
    isVisible,
  )

  // Track the mobile breakpoint so the sidebar can render as an off-canvas
  // drawer below it and as the fixed-width rail above it. Closing the drawer when
  // the viewport grows past the breakpoint keeps a stale open state from leaving
  // a fixed overlay stranded over the desktop layout.
  useEffect(() => {
    if (
      typeof window === "undefined" ||
      typeof window.matchMedia !== "function"
    )
      return
    const query = window.matchMedia(MOBILE_QUERY)
    const onChange = (event: MediaQueryListEvent) => {
      if (!event.matches) {
        restoreSidebarFocusRef.current =
          asideRef.current?.contains(document.activeElement) ?? false
        closeMobileNav()
      }
      setIsMobile(event.matches)
    }
    // Safari < 14 (and some older engines) only expose the deprecated
    // addListener/removeListener; fall back to it so the shell doesn't throw.
    if (typeof query.addEventListener === "function") {
      query.addEventListener("change", onChange)
      return () => query.removeEventListener("change", onChange)
    }
    query.addListener(onChange)
    return () => query.removeListener(onChange)
  }, [closeMobileNav])

  // Escape closes the drawer, matching the dismissible-overlay convention. The
  // organization submenu first when it is open, so one press unwinds one level
  // rather than taking the whole menu with it when the row you were after is
  // the level above.
  useEffect(() => {
    if (!mobileNavOpen) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return
      if (mobileOrgNavOpen) setMobileOrgNavOpen(false)
      else setMobileNavOpen(false)
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [mobileNavOpen, mobileOrgNavOpen])

  // Focus management for the mobile drawer, which is a modal overlay: move focus
  // into it when it opens and restore focus to the toggle when it closes, so
  // keyboard and screen-reader users are neither stranded inside a hidden panel
  // nor dropped back to the top of the document. A breakpoint change to desktop
  // keeps focus on the sidebar when it came from the drawer, rather than moving
  // it to the now-hidden toggle or losing it when the submenu unmounts.
  useEffect(() => {
    if (!isMobile) {
      if (restoreSidebarFocusRef.current) {
        restoreSidebarFocusRef.current = false
        asideRef.current?.focus()
      }
      return
    }
    if (mobileNavOpen) {
      asideRef.current?.focus()
    } else if (asideRef.current?.contains(document.activeElement)) {
      toggleRef.current?.focus()
    }
  }, [isMobile, mobileNavOpen])

  // The submenu is a level inside the drawer rather than a second overlay, so it
  // moves focus the way the drawer does: onto the control that leaves the level
  // when it opens, and back onto the row that opened it when it closes. Both
  // controls unmount when the level changes, so without this a tap would leave
  // focus on an element that is gone and drop a keyboard or AT cursor to the top
  // of the document.
  useEffect(() => {
    if (!isMobile || !mobileNavOpen) return
    if (mobileOrgNavOpen) {
      orgNavBackRef.current?.focus()
      return
    }
    // Only when the control that closed the level has left focus behind it. On
    // the render that opens the drawer, focus is on the panel itself, and
    // pulling it down to the footer would skip the whole rail.
    if (document.activeElement === document.body) {
      orgNavTriggerRef.current?.focus()
    }
  }, [isMobile, mobileNavOpen, mobileOrgNavOpen])

  useEffect(() => {
    try {
      window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? "1" : "0")
    } catch {
      // Ignore storage errors; the collapse state still applies for this session.
    }
  }, [collapsed])

  // Move focus (and scroll) to the page's main region, past the header and the
  // whole nav. A plain anchor to `#main-content` can't do this: the router runs
  // on hash history, so that href would register as a route change. Focusing
  // the ref directly keeps the route intact; `main` carries tabIndex={-1} so it
  // can accept programmatic focus without joining the tab order.
  const skipToMain = useCallback(
    (event: ReactMouseEvent<HTMLButtonElement>) => {
      event.preventDefault()
      // Focusing the region also scrolls it into view, so no separate scroll call.
      mainRef.current?.focus()
    },
    [],
  )

  // Collapse is a desktop-only affordance; on mobile the drawer always shows
  // the full-width, labeled nav.
  const effectiveCollapsed = isMobile ? false : collapsed
  // While the mobile drawer is open, the page behind it is inert: that is what
  // keeps an AT virtual cursor and Tab out of controls nobody can see. The top
  // bar is deliberately not included, because the control that closes the drawer
  // is in it, and the trail beside that control is the one thing worth reading
  // while the drawer is open.
  const backgroundInert = isMobile && mobileNavOpen ? true : undefined

  return (
    <div className="relative flex h-full flex-col overflow-hidden">
      {/* The first tab stop: a keyboard user can jump straight to the page body
          instead of tabbing through the whole nav on every route. Visually hidden
          until focused, then pinned top-left over the header (z above it). Goes
          inert with the drawer (like the header/main it targets) so it is not the
          one live background control an AT cursor reaches ahead of the modal, only
          to no-op against an inert main. */}
      <button
        type="button"
        inert={backgroundInert}
        onClick={skipToMain}
        className="sr-only focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:z-50 focus:rounded-lg focus:border focus:border-accent focus:bg-surface focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-link focus:shadow-md focus:outline-none"
      >
        Skip to main content
      </button>
      {/* Renders nothing. Mounted here rather than at the root because it names
          the actor a session belongs to, and this shell is what a session gets
          you; the pages in front of one have no actor to name. */}
      <TelemetryIdentity />
      <UpdatePrompt />
      <ConnectionStatus />
      <PricingWarning />
      {/* `relative` so the mobile drawer can be offset from *this row* rather
          than from the viewport. The row's top edge is the header's top edge,
          and the banners above it (the update prompt, the connection status) are
          in flow, so a viewport-relative offset would leave
          the drawer covering the header by however tall they are, taking the
          only control that closes it with them. */}
      <div className="relative flex min-h-0 flex-1">
        <aside
          ref={asideRef}
          id="app-sidebar"
          // Named on mobile, where it is a panel that slides over the page
          // rather than the page's own rail. Not a dialog and not modal: the
          // design fills the viewport below the top bar, so nothing is left
          // behind it to trap focus away from, and the control that dismisses it
          // lives in that bar. While closed it is off-canvas, so inert takes its
          // links out of the tab order and the accessibility tree until opened.
          aria-label={isMobile ? "Navigation" : undefined}
          // Programmatically focusable on both layouts so a breakpoint change
          // can preserve focus on the navigation landmark without adding it to
          // the natural tab order.
          tabIndex={-1}
          inert={isMobile && !mobileNavOpen ? true : undefined}
          className={clsx(
            "flex flex-col gap-4 border-r border-border bg-background-alt p-3 focus:outline-none",
            isMobile
              ? clsx(
                  // Full width, starting below the top bar: `top-14` pairs with
                  // the header's `min-h-14`, which is exact because everything in
                  // that bar truncates rather than wrapping. Absolute within the
                  // row the header leads, not fixed to the viewport, so a banner
                  // above that row moves the drawer down with the header instead
                  // of leaving it over the top of it: the header holds the only
                  // control that closes this, since the design fills the
                  // viewport below the bar and so has no backdrop to dim, no
                  // shadow to lift it off a page you cannot see, and nothing
                  // behind it to dismiss it by.
                  // 250ms on `--ease-out-fluid` is what HeroUI slides its
                  // own Drawer in on, and this is the same gesture.
                  "absolute inset-x-0 top-14 bottom-0 z-40 w-full transition-transform duration-250 ease-out-fluid motion-reduce:transition-none",
                  mobileNavOpen ? "translate-x-0" : "-translate-x-full",
                )
              : clsx(
                  // 150ms, which is what `otari-ai/frontend`'s shell collapses
                  // its own rail in.
                  "relative shrink-0 transition-[width] duration-150 motion-reduce:transition-none",
                  collapsed ? COLLAPSED_SIDEBAR_WIDTH : SIDEBAR_WIDTH,
                ),
          )}
        >
          {/* The scope the rail below belongs to. In the workspace context that
              is the switcher; in the organization context it is the way back
              out, which is how the prototype leaves that rail. */}
          {showOrganizationRail ? (
            <div className="flex min-h-14 items-center">
              {inOrganization ? (
                <Link
                  to={workspaceLanding?.to ?? "/"}
                  onClick={() => {
                    // Leaving the organization rail is a sidebar move like any
                    // other; it just does not go through `NavRowLink`.
                    const to = workspaceLanding?.to ?? "/"
                    recordNavigation(to, pathname === to)
                    closeMobileNav()
                  }}
                  className={navRowClass({ collapsed: effectiveCollapsed })}
                  aria-label={effectiveCollapsed ? backLabel : undefined}
                  title={effectiveCollapsed ? backLabel : undefined}
                >
                  <FiArrowLeft aria-hidden="true" className={NAV_ICON_CLASS} />
                  {effectiveCollapsed ? null : (
                    <span className="min-w-0 flex-1 truncate">{backLabel}</span>
                  )}
                </Link>
              ) : (
                // The submenu's way back, which closes a level rather than
                // navigating: the page under the drawer is still the workspace
                // page you opened the menu from, so there is nothing to go back
                // *to* yet. Same row and same name as the link above, because it
                // means the same thing to whoever reads it and differs only in
                // whether the route has moved yet. `cursor-pointer` because a
                // bare button resolves to the default arrow, which is the one
                // thing `navRowClass` leaves to its call sites.
                <button
                  type="button"
                  ref={orgNavBackRef}
                  onClick={() => setMobileOrgNavOpen(false)}
                  className={`${navRowClass()} cursor-pointer`}
                >
                  <FiArrowLeft aria-hidden="true" className={NAV_ICON_CLASS} />
                  <span className="min-w-0 flex-1 truncate text-left">
                    {backLabel}
                  </span>
                </button>
              )}
            </div>
          ) : (
            <WorkspaceSwitcher collapsed={effectiveCollapsed} />
          )}
          <nav
            // Named because the header's breadcrumb is a navigation landmark
            // too, and two unnamed ones give a screen-reader user no way to tell
            // the rail from the trail.
            aria-label="Sidebar"
            // Expanded, one 2px rhythm runs through rows *and* between groups:
            // the 32px heading block is what separates one group from the next.
            // Collapsed there are no headings, so the gap has to do that work.
            className={clsx(
              "flex min-h-0 flex-1 flex-col overflow-y-auto overflow-x-hidden",
              effectiveCollapsed ? "gap-3" : "gap-0.5",
            )}
          >
            {visibleSections.map(({ section, items }) => {
              return (
                <section
                  key={section.id}
                  aria-label={section.label}
                  className="flex flex-col gap-0.5"
                >
                  {/* A heading labels each group when expanded. Collapsed there is
                      no width for one, and an unlabeled group never had one, so in
                      both cases the rhythm above does the separating instead of a
                      rule: a divider between every pair of groups reads as five
                      lists rather than one rail. */}
                  {!effectiveCollapsed && section.label ? (
                    <p className={NAV_SECTION_HEADING_CLASS}>{section.label}</p>
                  ) : null}
                  <div className="flex flex-col gap-0.5">
                    {items.map((item) =>
                      item.children ? (
                        <NavGroup
                          key={item.to}
                          item={item}
                          currentPath={pathname}
                          onNavigate={closeMobileNav}
                          isVisible={isVisible}
                          collapsed={effectiveCollapsed}
                        />
                      ) : (
                        // Highlighted from the registry's own answer rather than
                        // from `activeProps`, whose default match is a prefix
                        // one: on `/organization/members` that lights up
                        // "General" as well, since `/organization` is its parent
                        // route. `navItemForPath` prefers the exact entry, and a
                        // future child route (`/routing/new`) still resolves to
                        // its parent, which is the highlight that route wants.
                        <NavRowLink
                          key={item.to}
                          to={item.to}
                          label={item.label}
                          icon={item.icon}
                          isActive={currentItem?.to === item.to}
                          collapsed={effectiveCollapsed}
                          // Tapping a destination dismisses the mobile drawer so
                          // the page it landed on is visible, not behind it.
                          onNavigate={closeMobileNav}
                        />
                      ),
                    )}
                  </div>
                </section>
              )
            })}
          </nav>
          {/* The account block, set off by a rule as in the navigation prototype:
              the way onto the organization rail, the bundled guide, and the
              account control whose menu carries appearance and sign-out. */}
          <div className="flex flex-col gap-1 border-t border-border pt-1 pb-[env(safe-area-inset-bottom)]">
            {/* The way into the organization rail. Only in the workspace
                context, since the organization one has its own way back, and
                only for someone who manages the organization: it is the single
                destination the design hides outright rather than degrading to
                read-only (artboard A2 is the member variant, and this row is
                what it drops).

                Drawn as an ordinary nav row, which is how the design draws it.
                It used to be a bordered box with a trailing chevron, on the
                argument that a context switch should not look like a page and
                that an operator whose sidebar used to list Users, Budgets and
                Settings needed to find where they went. Both were true and
                neither survives the design: the box makes the footer read as a
                button bar under the rail rather than as the end of it.

                Two controls for one entry, because the rail behaves differently
                under the two layouts. On the desk it navigates, and the rail it
                swaps to is left on screen to read. On a phone the drawer closes
                over whatever it navigated to, so the same tap would show the
                organization's first page and never the rail that lists the
                rest; there the row opens that rail inside the drawer instead,
                and a destination in it is what dismisses the drawer. That is
                also what earns the trailing chevron back: it promises a submenu
                only where one now opens. */}
            {!showOrganizationRail && managesOrganization ? (
              isMobile ? (
                // `cursor-pointer` because a bare button resolves to the default
                // arrow, which is the one thing `navRowClass` leaves to its call
                // sites (its other rows are links or HeroUI buttons).
                <button
                  type="button"
                  ref={orgNavTriggerRef}
                  onClick={() => setMobileOrgNavOpen(true)}
                  className={`${navRowClass()} cursor-pointer`}
                >
                  <FiSettings aria-hidden="true" className={NAV_ICON_CLASS} />
                  <span className="min-w-0 flex-1 truncate text-left">
                    Organization
                  </span>
                  <FiChevronRight
                    aria-hidden="true"
                    className={NAV_ICON_CLASS}
                  />
                </button>
              ) : (
                <Link
                  to={organizationLanding?.to ?? "/organization/members"}
                  onClick={() => {
                    const to =
                      organizationLanding?.to ?? "/organization/members"
                    recordNavigation(to, pathname === to)
                  }}
                  className={navRowClass({ collapsed: effectiveCollapsed })}
                  aria-label={effectiveCollapsed ? "Organization" : undefined}
                  title={
                    effectiveCollapsed
                      ? "Organization: members, spend and budgets, users, settings"
                      : undefined
                  }
                >
                  <FiSettings aria-hidden="true" className={NAV_ICON_CLASS} />
                  {effectiveCollapsed ? null : (
                    <span className="min-w-0 flex-1 truncate">
                      Organization
                    </span>
                  )}
                </Link>
              )
            ) : null}
            {/* One control, not a stack of links: the guide, appearance, and
                sign-out all live in its menu, which is how the prototype ends
                the rail. Sign-out used to sit in the page header. */}
            {/* The design rules the account row off from the row above it, so
                the control that ends the rail is not read as one more
                destination in the group that changes context. */}
            {!showOrganizationRail && managesOrganization ? (
              <div aria-hidden="true" className="h-px shrink-0 bg-border" />
            ) : null}
            <AccountMenu collapsed={effectiveCollapsed} />
          </div>
        </aside>
        {/* The right-hand pane: the header sits beside the rail rather than above
            it, which is what lets the sidebar run the full height of the window. */}
        <div className="flex min-w-0 flex-1 flex-col">
          <header
            // The design's top bar sits on the page ground rather than on a
            // card fill, so the rail is the only chrome that reads as a surface.
            className="flex min-h-14 shrink-0 items-center gap-4 border-b border-border bg-background pr-5 pl-4"
          >
            <div className="flex min-w-0 flex-1 items-center gap-4">
              <button
                type="button"
                ref={toggleRef}
                onClick={() =>
                  mobileNavOpen ? closeMobileNav() : setMobileNavOpen(true)
                }
                aria-label={
                  mobileNavOpen ? "Close navigation" : "Open navigation"
                }
                aria-expanded={mobileNavOpen}
                aria-controls="app-sidebar"
                className="-ml-1 flex h-11 w-11 shrink-0 items-center justify-center rounded-lg text-muted transition-colors hover:bg-surface-alt hover:text-foreground md:hidden"
              >
                {mobileNavOpen ? (
                  <FiX aria-hidden="true" className="size-5" />
                ) : (
                  <FiMenu aria-hidden="true" className="size-5" />
                )}
              </button>
              {/* Collapse lives here, at the head of the content pane, rather
                  than floating on the rail's edge: the rail now runs the full
                  height of the window and has no edge above the fold to hang it
                  on. Desktop-only, as before; on mobile the drawer is dismissed
                  from the control to its left, which is why that control stays
                  visible under an open drawer. */}
              <button
                type="button"
                onClick={() => setCollapsed((value) => !value)}
                aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                aria-pressed={collapsed}
                title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                className="-ml-1 hidden h-8 w-8 shrink-0 items-center justify-center rounded-md text-muted transition-colors hover:bg-surface-alt hover:text-foreground md:flex"
              >
                <FiSidebar aria-hidden="true" className="size-4" />
              </button>
              <Breadcrumbs pathname={pathname} />
            </div>
            <TopBarActions />
          </header>
          <main
            ref={mainRef}
            id="main-content"
            // tabIndex={-1} lets the skip link move focus here programmatically
            // without adding the region itself to the natural tab order.
            tabIndex={-1}
            inert={backgroundInert}
            className="flex-1 overflow-y-auto focus:outline-none"
          >
            <div className="mx-auto flex max-w-[1800px] flex-col gap-6 px-4 py-5 md:px-6 md:py-6">
              {answerIsStillComing ? (
                <PendingPage />
              ) : routeIsGatedOff ? (
                <EmptyState
                  // The leaf's name, not the group's: someone who followed a
                  // link to Guardrails should not be told "Routing" is missing.
                  // `navLabelForPath` answers for every registered path, and only
                  // a registered path can be gated off, so the fallback is there
                  // for the type rather than for a case that happens.
                  title={`${navLabelForPath(pathname) ?? currentItem?.label ?? "That page"} is not available here`}
                  description="This deployment does not serve that page. Pick a destination from the sidebar."
                />
              ) : (
                <Outlet />
              )}
            </div>
          </main>
        </div>
      </div>
    </div>
  )
}
