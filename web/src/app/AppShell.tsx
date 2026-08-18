import { Link, Outlet, useLocation } from "@tanstack/react-router"
import { clsx } from "clsx"
import type {
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent,
} from "react"
import { useCallback, useEffect, useRef, useState } from "react"
import { ConnectionStatus } from "@/app/ConnectionStatus"
import { AccountMenu } from "@/app/nav/AccountMenu"
import { Breadcrumbs } from "@/app/nav/Breadcrumbs"
import {
  NAV_SECTIONS,
  navContextForPath,
  navItemForPath,
  navLabelForPath,
  ORG_NAV_SECTIONS,
  visibleNavSections,
} from "@/app/nav/registry"
import type { NavItem } from "@/app/nav/types"
import { useNavVisibility } from "@/app/nav/useNavVisibility"
import { WorkspaceSwitcher } from "@/app/nav/WorkspaceSwitcher"
import { UpdatePrompt } from "@/app/UpdatePrompt"
import { PricingWarning } from "@/features/models/PricingWarning"
import { canManage } from "@/features/organization/roles"
import { useOrganizationContext } from "@/shared/api/hooks"
import { EmptyState } from "@/shared/components/ui"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

const MIN_SIDEBAR = 200
const MAX_SIDEBAR = 480
const DEFAULT_SIDEBAR = 240
const COLLAPSED_SIDEBAR = 60
const SIDEBAR_WIDTH_KEY = "otari.dashboard.sidebarWidth"
const SIDEBAR_COLLAPSED_KEY = "otari.dashboard.sidebarCollapsed"
const SIDEBAR_STEP = 16

// Below this width the sidebar's fixed footprint squashes page content, so it
// switches to an off-canvas drawer toggled from the header. Matches Tailwind's
// `md` breakpoint (the classes that hide the trigger and drawer chrome use `md:`).
const MOBILE_QUERY = "(max-width: 767px)"

const clampSidebar = (width: number) =>
  Math.min(MAX_SIDEBAR, Math.max(MIN_SIDEBAR, width))

function readIsMobile(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function")
    return false
  return window.matchMedia(MOBILE_QUERY).matches
}

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'

// Visible, focusable descendants of a container, in DOM order. offsetParent is
// null for display:none nodes (e.g. the desktop-only collapse chevron on mobile),
// so filtering on it keeps the focus trap's first/last from landing on a hidden
// control that can't actually take focus.
function getFocusable(container: HTMLElement | null): HTMLElement[] {
  if (!container) return []
  return Array.from(
    container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
  ).filter((el) => el.offsetParent !== null || el === document.activeElement)
}

function readStoredSidebarWidth(): number {
  if (typeof window === "undefined") return DEFAULT_SIDEBAR
  try {
    const raw = window.localStorage.getItem(SIDEBAR_WIDTH_KEY)
    const parsed = raw ? Number.parseInt(raw, 10) : Number.NaN
    return Number.isNaN(parsed) ? DEFAULT_SIDEBAR : clampSidebar(parsed)
  } catch {
    // Storage can throw when disabled (e.g. blocked cookies / private mode);
    // fall back to the default rather than white-screening the shell.
    return DEFAULT_SIDEBAR
  }
}

function readStoredCollapsed(): boolean {
  if (typeof window === "undefined") return false
  try {
    return window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1"
  } catch {
    return false
  }
}

// Shared by the nav links and the user-guide link below them, so the two agree
// on shape and only differ in what marks the current page.
const navLinkClass = (collapsed: boolean) =>
  clsx(
    "flex items-center rounded-lg py-2 text-sm font-medium transition-colors",
    collapsed ? "justify-center px-0" : "gap-3 px-3",
  )
const NAV_ACTIVE = "bg-primary-subtle text-primary-subtle-foreground"
const NAV_INACTIVE = "text-muted hover:bg-surface-alt hover:text-foreground"

/**
 * A sidebar entry with destinations nested under it, drawn the way the
 * navigation prototype draws Routing and Tools: a row that expands rather than
 * navigates, and indented children below it.
 *
 * Open when the current route is one of its children, so arriving by URL shows
 * where you are rather than a collapsed group. Held in state after that, so
 * closing it stays closed while you read the page it opened.
 *
 * Not rendered when the rail is collapsed: there is no width for the labels, and
 * the parent's icon links straight to its own page instead.
 */
function NavGroup({
  item,
  currentPath,
  onNavigate,
  isVisible,
}: {
  item: NavItem
  currentPath: string
  onNavigate: () => void
  isVisible: (item: NavItem) => boolean
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
  // Follows the route when navigation lands inside the group from elsewhere
  // (a link on a page, a bookmark), without fighting a manual close.
  const [lastHeld, setLastHeld] = useState(holdsCurrent)
  if (holdsCurrent !== lastHeld) {
    setLastHeld(holdsCurrent)
    if (holdsCurrent) setOpen(true)
  }

  return (
    <div className="flex flex-col gap-1">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        className={clsx(
          navLinkClass(false),
          "justify-between",
          holdsCurrent ? NAV_ACTIVE : NAV_INACTIVE,
        )}
      >
        <span className="flex items-center gap-3">
          {item.icon}
          {item.label}
        </span>
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={clsx(
            "h-4 w-4 shrink-0 transition-transform motion-reduce:transition-none",
            open && "rotate-90",
          )}
        >
          <path d="M9 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open
        ? children.map((child) => (
            <Link
              key={child.to}
              to={child.to}
              activeOptions={{ exact: true }}
              onClick={onNavigate}
              className={clsx(
                navLinkClass(false),
                // Indented to the parent's label rather than its icon, which is
                // what marks it as nested without repeating a glyph.
                "pl-11",
                currentPath === child.to ? NAV_ACTIVE : NAV_INACTIVE,
              )}
            >
              {child.label}
            </Link>
          ))
        : null}
    </div>
  )
}

export function AppShell() {
  // Navigation is data: the shell renders whatever the registry declares and
  // decides visibility from the deployment, the entitlements, and the flags,
  // rather than each page asking what it is running against.
  const isVisible = useNavVisibility()
  const { pathname } = useLocation()
  // A gated-off destination is still reachable by bookmark or shared URL, so the
  // shell answers those with a panel instead of a page whose every request the
  // server would refuse. An unregistered path (the guide, the 404 splat) has no
  // entry and is never gated.
  const currentItem = navItemForPath(pathname)
  const routeIsGatedOff = currentItem !== undefined && !isVisible(currentItem)
  // Which of the two sidebars this path belongs under. The organization context
  // is a separate rail reached from the footer, not a section inside the
  // workspace one, so the two never render together.
  const navContext = navContextForPath(pathname)
  const inOrganization = navContext === "organization"
  // Filtered before it is indexed, so the divider and top margin below key off
  // the first *rendered* section rather than the first registered one.
  const visibleSections = visibleNavSections(
    inOrganization ? ORG_NAV_SECTIONS : NAV_SECTIONS,
    isVisible,
  )
  const organization = useOrganizationContext()
  const { selected: selectedWorkspace } = useSelectedWorkspace()
  // Always true in a standalone deployment, where the one session is the local
  // operator and it owns the organization the gateway provisioned for itself.
  // Written anyway because it becomes load-bearing the moment per-user sign-in
  // lands (otari-ai#1716), and because an overlay build can already be reached
  // by someone who is not an admin.
  const managesOrganization = canManage(organization.data)

  const asideRef = useRef<HTMLElement>(null)
  const mainRef = useRef<HTMLElement>(null)
  const toggleRef = useRef<HTMLButtonElement>(null)
  const [sidebarWidth, setSidebarWidth] = useState<number>(
    readStoredSidebarWidth,
  )
  const [collapsed, setCollapsed] = useState<boolean>(readStoredCollapsed)
  const [resizing, setResizing] = useState(false)
  const [isMobile, setIsMobile] = useState<boolean>(readIsMobile)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)

  // Track the mobile breakpoint so the sidebar can render as an off-canvas
  // drawer below it and as the resizable rail above it. Closing the drawer when
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
      setIsMobile(event.matches)
      if (!event.matches) setMobileNavOpen(false)
    }
    // Safari < 14 (and some older engines) only expose the deprecated
    // addListener/removeListener; fall back to it so the shell doesn't throw.
    if (typeof query.addEventListener === "function") {
      query.addEventListener("change", onChange)
      return () => query.removeEventListener("change", onChange)
    }
    query.addListener(onChange)
    return () => query.removeListener(onChange)
  }, [])

  // Escape closes the drawer, matching the dismissible-overlay convention.
  useEffect(() => {
    if (!mobileNavOpen) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileNavOpen(false)
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [mobileNavOpen])

  // Focus management for the mobile drawer, which is a modal overlay: move focus
  // into it when it opens and restore focus to the toggle when it closes, so
  // keyboard and screen-reader users are neither stranded inside a hidden panel
  // nor dropped back to the top of the document. The isMobile guard means a
  // breakpoint change to desktop (which also closes the drawer) never yanks focus
  // to the now-hidden toggle.
  useEffect(() => {
    if (!isMobile) return
    if (mobileNavOpen) {
      asideRef.current?.focus()
    } else if (asideRef.current?.contains(document.activeElement)) {
      toggleRef.current?.focus()
    }
  }, [isMobile, mobileNavOpen])

  // Keep Tab within the open drawer so focus cannot wander to the page behind the
  // backdrop. Paired with the aside being inert while closed, this bounds keyboard
  // focus to whichever surface is actually interactive.
  const trapFocus = useCallback((event: ReactKeyboardEvent<HTMLElement>) => {
    if (event.key !== "Tab") return
    const focusables = getFocusable(asideRef.current)
    if (focusables.length === 0) return
    const first = focusables[0]
    const last = focusables[focusables.length - 1]
    const active = document.activeElement
    if (event.shiftKey && (active === first || active === asideRef.current)) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && active === last) {
      event.preventDefault()
      first.focus()
    }
  }, [])

  useEffect(() => {
    const id = window.setTimeout(() => {
      try {
        window.localStorage.setItem(
          SIDEBAR_WIDTH_KEY,
          String(Math.round(sidebarWidth)),
        )
      } catch {
        // Ignore storage errors; the width still applies for this session.
      }
    }, 200)
    return () => window.clearTimeout(id)
  }, [sidebarWidth])

  useEffect(() => {
    try {
      window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? "1" : "0")
    } catch {
      // Ignore storage errors; the collapse state still applies for this session.
    }
  }, [collapsed])

  const startResize = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      event.preventDefault()
      event.currentTarget.setPointerCapture(event.pointerId)
      setResizing(true)
    },
    [],
  )

  const moveResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return
    const left = asideRef.current?.getBoundingClientRect().left ?? 0
    setSidebarWidth(clampSidebar(event.clientX - left))
  }, [])

  const endResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    setResizing(false)
  }, [])

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

  const nudgeResize = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (event.key === "ArrowLeft") {
        event.preventDefault()
        setSidebarWidth((width) => clampSidebar(width - SIDEBAR_STEP))
      } else if (event.key === "ArrowRight") {
        event.preventDefault()
        setSidebarWidth((width) => clampSidebar(width + SIDEBAR_STEP))
      }
    },
    [],
  )

  const width = collapsed ? COLLAPSED_SIDEBAR : sidebarWidth
  // The collapse rail and resize handle are desktop-only affordances; on mobile
  // the drawer always shows the full-width, labeled nav.
  const effectiveCollapsed = isMobile ? false : collapsed
  // While the mobile drawer is open, make everything behind it (header + page)
  // inert so a modal really is modal: aria-modal alone isn't universally honored,
  // so this is what keeps an AT virtual cursor and Tab out of the obscured
  // controls, not just the aside's own focus trap.
  const backgroundInert = isMobile && mobileNavOpen ? true : undefined

  return (
    <div
      className={clsx(
        "relative flex h-full flex-col overflow-hidden",
        resizing && "cursor-col-resize select-none",
      )}
    >
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
      <UpdatePrompt />
      <ConnectionStatus />
      <PricingWarning />
      <div className="flex min-h-0 flex-1">
        {/* On mobile the drawer floats over the page; a backdrop dims the content
            behind it and dismisses it on tap. A non-interactive div (not a
            button): dismissal by pointer is a convenience, keyboard users close
            with Escape, and an aria-hidden interactive element is a contradiction. */}
        {isMobile && mobileNavOpen ? (
          <div
            aria-hidden="true"
            onClick={() => setMobileNavOpen(false)}
            className="fixed inset-0 z-30 bg-backdrop/40 md:hidden"
          />
        ) : null}
        {/* biome-ignore lint/a11y/useAriaPropsSupportedByRole: the role is conditional (dialog on mobile), which the rule cannot evaluate */}
        <aside
          ref={asideRef}
          id="app-sidebar"
          // On mobile the drawer is a modal dialog; give it a name and mark it
          // modal while open. While closed it is off-canvas, so inert takes its
          // links out of the tab order and the accessibility tree until opened.
          role={isMobile ? "dialog" : undefined}
          aria-modal={isMobile && mobileNavOpen ? true : undefined}
          aria-label={isMobile ? "Navigation" : undefined}
          tabIndex={isMobile ? -1 : undefined}
          inert={isMobile && !mobileNavOpen ? true : undefined}
          onKeyDown={isMobile && mobileNavOpen ? trapFocus : undefined}
          style={isMobile ? undefined : { width }}
          className={clsx(
            "flex flex-col border-r border-border bg-background-alt focus:outline-none",
            isMobile
              ? clsx(
                  "fixed inset-y-0 left-0 z-40 w-[17rem] shadow-xl transition-transform duration-200",
                  mobileNavOpen ? "translate-x-0" : "-translate-x-full",
                )
              : clsx(
                  "relative shrink-0",
                  !resizing && "transition-[width] duration-150",
                ),
          )}
        >
          {/* The scope the rail below belongs to. In the workspace context that
              is the switcher; in the organization context it is the way back
              out, which is how the prototype leaves that rail. */}
          {inOrganization ? (
            <Link
              to="/"
              onClick={() => setMobileNavOpen(false)}
              className={clsx(
                navLinkClass(effectiveCollapsed),
                NAV_INACTIVE,
                effectiveCollapsed ? "mx-2 mt-3" : "mx-3 mt-3",
              )}
              aria-label={
                effectiveCollapsed
                  ? `Back to ${selectedWorkspace?.name ?? "workspace"}`
                  : undefined
              }
              title={
                effectiveCollapsed
                  ? `Back to ${selectedWorkspace?.name ?? "workspace"}`
                  : undefined
              }
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="h-5 w-5 shrink-0"
              >
                <path
                  d="M15 6l-6 6 6 6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              {effectiveCollapsed
                ? null
                : `Back to ${selectedWorkspace?.name ?? "workspace"}`}
            </Link>
          ) : (
            <div className="pt-3">
              <WorkspaceSwitcher collapsed={effectiveCollapsed} />
            </div>
          )}
          <nav
            // Named because the header's breadcrumb is a navigation landmark
            // too, and two unnamed ones give a screen-reader user no way to tell
            // the rail from the trail.
            aria-label="Sidebar"
            className={clsx(
              "flex flex-col py-4",
              effectiveCollapsed ? "px-2" : "px-3",
            )}
          >
            {visibleSections.map(({ section, items }, sectionIndex) => {
              return (
                <div
                  key={section.id}
                  className={sectionIndex > 0 ? "mt-4" : undefined}
                >
                  {/* A header labels each group when expanded; a thin divider stands
                      in for it when the sidebar is collapsed, or when a group has no
                      header of its own (e.g. Settings) to set it off from the group
                      above. */}
                  {!effectiveCollapsed && section.label ? (
                    <div className="px-3 pb-1 text-[11px] font-semibold tracking-wider text-muted uppercase">
                      {section.label}
                    </div>
                  ) : null}
                  {sectionIndex > 0 &&
                  (effectiveCollapsed || !section.label) ? (
                    <div className="mx-1 mb-2 border-t border-border" />
                  ) : null}
                  <div className="flex flex-col gap-1">
                    {items.map((item) =>
                      item.children && !effectiveCollapsed ? (
                        <NavGroup
                          key={item.to}
                          item={item}
                          currentPath={pathname}
                          onNavigate={() => setMobileNavOpen(false)}
                          isVisible={isVisible}
                        />
                      ) : (
                        <Link
                          key={item.to}
                          to={item.to}
                          // Exact, because the default is a prefix match: on
                          // /organization/members that leaves `aria-current` on
                          // "Organization" as well as on the child. The class
                          // below was already driven from the registry; this is
                          // the half a screen reader reads.
                          activeOptions={{ exact: true }}
                          // Highlighted from the registry's own answer rather than
                          // from `activeProps`, whose default match is a prefix
                          // one: on `/organization/members` that lights up
                          // "General" as well, since `/organization` is its parent
                          // route. `navItemForPath` prefers the exact entry, and a
                          // future child route (`/routing/new`) still resolves to
                          // its parent, which is the highlight that route wants.
                          className={clsx(
                            navLinkClass(effectiveCollapsed),
                            currentItem?.to === item.to
                              ? NAV_ACTIVE
                              : NAV_INACTIVE,
                          )}
                          // Tapping a destination dismisses the mobile drawer so the
                          // page it navigated to is visible, not hidden behind it.
                          onClick={() => setMobileNavOpen(false)}
                          aria-label={
                            effectiveCollapsed ? item.label : undefined
                          }
                          title={effectiveCollapsed ? item.label : undefined}
                        >
                          {item.icon}
                          {effectiveCollapsed ? null : item.label}
                        </Link>
                      ),
                    )}
                  </div>
                </div>
              )
            })}
          </nav>
          {/* The account block, set off by a rule as in the navigation prototype:
              the way onto the organization rail, the bundled guide, and the
              account control whose menu carries appearance and sign-out. */}
          <div className="mt-auto flex flex-col gap-1 border-t border-border pt-2 pb-3">
            {/* The way into the organization rail. Only in the workspace
                context, since the organization one has its own way back, and
                only for someone who manages the organization: it is the single
                destination the prototype hides outright rather than degrading
                to read-only.

                Drawn as a bordered row with a trailing chevron rather than as
                another muted link, because it is the only control here that
                changes context rather than opening a page, and because Users,
                Budgets and Settings all moved behind it: an operator upgrading
                from a sidebar that listed them needs to find this. */}
            {!inOrganization && managesOrganization ? (
              <Link
                to="/organization/members"
                onClick={() => setMobileNavOpen(false)}
                className={clsx(
                  navLinkClass(effectiveCollapsed),
                  "border border-border bg-surface text-foreground transition-colors hover:border-accent hover:bg-surface-alt",
                  effectiveCollapsed ? "mx-2" : "mx-3",
                )}
                aria-label={effectiveCollapsed ? "Organization" : undefined}
                title={
                  effectiveCollapsed
                    ? "Organization: members, spend and budgets, users, settings"
                    : undefined
                }
              >
                <svg
                  aria-hidden="true"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="h-5 w-5 shrink-0"
                >
                  <circle cx="12" cy="12" r="3" />
                  <path
                    d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"
                    strokeLinejoin="round"
                  />
                </svg>
                {effectiveCollapsed ? null : (
                  <>
                    Organization
                    <svg
                      aria-hidden="true"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      className="ml-auto h-4 w-4 shrink-0 text-muted"
                    >
                      <path
                        d="M9 6l6 6-6 6"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </>
                )}
              </Link>
            ) : null}
            {/* One control, not a stack of links: the guide, appearance, and
                sign-out all live in its menu, which is how the prototype ends
                the rail. Sign-out used to sit in the page header. */}
            <AccountMenu collapsed={effectiveCollapsed} />
          </div>
          {collapsed || isMobile ? null : (
            // biome-ignore lint/a11y/useSemanticElements: <hr> is a thematic break; this is a keyboard-operable resize handle
            <div
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize sidebar"
              aria-valuenow={Math.round(sidebarWidth)}
              aria-valuemin={MIN_SIDEBAR}
              aria-valuemax={MAX_SIDEBAR}
              tabIndex={0}
              onPointerDown={startResize}
              onPointerMove={moveResize}
              onPointerUp={endResize}
              onKeyDown={nudgeResize}
              className={clsx(
                "absolute top-0 right-0 z-10 h-full w-1.5 cursor-col-resize touch-none transition-colors",
                "hover:bg-accent focus-visible:bg-accent focus:outline-none",
                resizing ? "bg-accent" : "bg-transparent",
              )}
            />
          )}
        </aside>
        {/* The right-hand pane: the header sits beside the rail rather than above
            it, which is what lets the sidebar run the full height of the window. */}
        <div className="flex min-w-0 flex-1 flex-col">
          <header
            inert={backgroundInert}
            className="flex shrink-0 items-center justify-between border-b border-border bg-surface px-5 py-3"
          >
            <div className="flex items-center gap-2.5">
              <button
                type="button"
                ref={toggleRef}
                onClick={() => setMobileNavOpen((value) => !value)}
                aria-label={
                  mobileNavOpen ? "Close navigation" : "Open navigation"
                }
                aria-expanded={mobileNavOpen}
                aria-controls="app-sidebar"
                className="-ml-1 flex h-8 w-8 items-center justify-center rounded-lg text-muted transition-colors hover:bg-surface-alt hover:text-foreground md:hidden"
              >
                <svg
                  aria-hidden="true"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="h-5 w-5"
                >
                  <path
                    d="M4 6h16M4 12h16M4 18h16"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
              {/* Collapse lives here, at the head of the content pane, rather
                  than floating on the rail's edge: the rail now runs the full
                  height of the window and has no edge above the fold to hang it
                  on. Desktop-only, as before; on mobile the drawer is dismissed
                  from the control to its left or from the backdrop. */}
              <button
                type="button"
                onClick={() => setCollapsed((value) => !value)}
                aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                aria-pressed={collapsed}
                title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                className="-ml-1 hidden h-7 w-7 items-center justify-center rounded-lg text-muted transition-colors hover:bg-surface-alt hover:text-foreground md:flex"
              >
                <svg
                  aria-hidden="true"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="h-4 w-4"
                >
                  <rect x="3" y="4" width="18" height="16" rx="2" />
                  <path d="M9 4v16" strokeLinecap="round" />
                </svg>
              </button>
              <Breadcrumbs pathname={pathname} />
            </div>
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
              {routeIsGatedOff ? (
                <EmptyState
                  // The leaf's name, not the group's: someone who followed a
                  // link to Guardrails should not be told "Routing" is missing.
                  title={`${navLabelForPath(pathname) ?? currentItem.label} is not available here`}
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
