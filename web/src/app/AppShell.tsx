import { Button } from "@heroui/react"
import { Link, Outlet, useLocation } from "@tanstack/react-router"
import { clsx } from "clsx"
import type {
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent,
} from "react"
import { useCallback, useEffect, useRef, useState } from "react"
import { ConnectionStatus } from "@/app/ConnectionStatus"
import { NAV_SECTIONS, navItemForPath } from "@/app/nav/registry"
import { useNavVisibility } from "@/app/nav/useNavVisibility"
import { UpdatePrompt } from "@/app/UpdatePrompt"
import { useAuth } from "@/features/auth/AuthContext"
import { PricingWarning } from "@/features/models/PricingWarning"
import { EmptyState } from "@/shared/components/ui"

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
const NAV_ACTIVE = "bg-[var(--otari-brand-tint)] text-[var(--otari-brand-dark)]"
const NAV_INACTIVE =
  "text-[var(--otari-muted)] hover:bg-[var(--otari-bg)] hover:text-[var(--otari-ink)]"

export function AppShell() {
  const { logout } = useAuth()
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
        className="sr-only focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:z-50 focus:rounded-lg focus:border focus:border-[var(--otari-brand)] focus:bg-[var(--otari-surface)] focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-[var(--otari-brand-dark)] focus:shadow-md focus:outline-none"
      >
        Skip to main content
      </button>
      <header
        inert={backgroundInert}
        className="flex shrink-0 items-center justify-between border-b border-[var(--otari-line)] bg-[var(--otari-surface)] px-5 py-3"
      >
        <div className="flex items-center gap-2.5">
          <button
            type="button"
            ref={toggleRef}
            onClick={() => setMobileNavOpen((value) => !value)}
            aria-label={mobileNavOpen ? "Close navigation" : "Open navigation"}
            aria-expanded={mobileNavOpen}
            aria-controls="app-sidebar"
            className="-ml-1 flex h-8 w-8 items-center justify-center rounded-lg text-[var(--otari-muted)] transition-colors hover:bg-[var(--otari-bg)] hover:text-[var(--otari-ink)] md:hidden"
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
          <img src="/favicon.svg" alt="" className="h-7 w-7 shrink-0" />
          <span className="text-base font-semibold text-[var(--otari-ink)]">
            Otari
          </span>
        </div>
        <Button
          size="sm"
          variant="outline"
          onPress={logout}
          aria-label="Sign out"
        >
          Sign out
        </Button>
      </header>
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
            className="fixed inset-0 z-30 bg-black/40 md:hidden"
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
            "flex flex-col border-r border-[var(--otari-line)] bg-[var(--otari-surface)] focus:outline-none",
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
          {/* A round chevron on the sidebar's edge toggles collapse — floats over
              the border for a polished, VS Code / Notion-style affordance.
              Desktop-only: on mobile the drawer is dismissed from the header or
              backdrop instead. */}
          <button
            type="button"
            onClick={() => setCollapsed((value) => !value)}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-pressed={collapsed}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="absolute -right-3 top-4 z-30 hidden h-6 w-6 items-center justify-center rounded-full border border-[var(--otari-line)] bg-[var(--otari-surface)] text-[var(--otari-muted)] shadow-sm transition-colors hover:border-[var(--otari-brand)] hover:text-[var(--otari-brand-dark)] md:flex"
          >
            <svg
              aria-hidden="true"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              className={clsx(
                "h-3.5 w-3.5 transition-transform",
                collapsed && "rotate-180",
              )}
            >
              <path
                d="M15 6l-6 6 6 6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          <nav
            className={clsx(
              "flex flex-col py-4",
              effectiveCollapsed ? "px-2" : "px-3",
            )}
          >
            {NAV_SECTIONS.map((section, sectionIndex) => {
              const items = section.items.filter(isVisible)
              if (items.length === 0) {
                return null
              }
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
                    <div className="px-3 pb-1 text-[11px] font-semibold tracking-wider text-[var(--otari-muted)] uppercase">
                      {section.label}
                    </div>
                  ) : null}
                  {sectionIndex > 0 &&
                  (effectiveCollapsed || !section.label) ? (
                    <div className="mx-1 mb-2 border-t border-[var(--otari-line)]" />
                  ) : null}
                  <div className="flex flex-col gap-1">
                    {items.map((item) => (
                      <Link
                        key={item.to}
                        to={item.to}
                        activeProps={{ className: NAV_ACTIVE }}
                        inactiveProps={{ className: NAV_INACTIVE }}
                        // Tapping a destination dismisses the mobile drawer so the
                        // page it navigated to is visible, not hidden behind it.
                        onClick={() => setMobileNavOpen(false)}
                        aria-label={effectiveCollapsed ? item.label : undefined}
                        title={effectiveCollapsed ? item.label : undefined}
                        className={navLinkClass(effectiveCollapsed)}
                      >
                        {item.icon}
                        {effectiveCollapsed ? null : item.label}
                      </Link>
                    ))}
                  </div>
                </div>
              )
            })}
          </nav>
          {/* Footer links, pinned to the bottom of the rail. The user guide is the
              dashboard's own docs, bundled with the running gateway (see DocsPage);
              otari.ai is a subtler pointer to the hosted product below it. */}
          <div className="mt-auto flex flex-col gap-1 pb-3">
            <Link
              to="/docs"
              activeProps={{ className: NAV_ACTIVE }}
              inactiveProps={{ className: NAV_INACTIVE }}
              // Tapping dismisses the mobile drawer, like the primary nav above.
              onClick={() => setMobileNavOpen(false)}
              aria-label={effectiveCollapsed ? "User guide" : undefined}
              title={effectiveCollapsed ? "User guide" : undefined}
              className={clsx(
                navLinkClass(effectiveCollapsed),
                // Indented by a margin rather than the nav's padding: this block
                // sits outside <nav>, pinned to the bottom of the rail.
                effectiveCollapsed ? "mx-2" : "mx-3",
              )}
            >
              {/* An open book: the operator guide for this dashboard. Decorative;
                  the link is labeled by its text (or aria-label when collapsed). */}
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="h-5 w-5 shrink-0"
              >
                <path
                  d="M12 6.5C10.5 5 8 4.5 4 4.5V18c4 0 6.5.5 8 2 1.5-1.5 4-2 8-2V4.5c-4 0-6.5.5-8 2z"
                  strokeLinejoin="round"
                />
                <path d="M12 6.5V20" strokeLinecap="round" />
              </svg>
              {effectiveCollapsed ? null : "User guide"}
            </Link>
            <a
              href="https://otari.ai"
              target="_blank"
              rel="noreferrer"
              title="otari.ai: the hosted Otari gateway"
              className={clsx(
                "flex items-center rounded-lg py-2 text-xs font-medium text-[var(--otari-muted)] transition-colors hover:bg-[var(--otari-bg)] hover:text-[var(--otari-brand-dark)]",
                effectiveCollapsed
                  ? "mx-2 justify-center px-0"
                  : "mx-3 gap-2 px-3",
              )}
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="h-4 w-4 shrink-0"
              >
                <path
                  d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"
                  strokeLinejoin="round"
                />
              </svg>
              {effectiveCollapsed ? null : (
                <span className="flex-1">
                  otari.ai <span aria-hidden>↗</span>
                </span>
              )}
            </a>
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
                "hover:bg-[var(--otari-brand)] focus-visible:bg-[var(--otari-brand)] focus:outline-none",
                resizing ? "bg-[var(--otari-brand)]" : "bg-transparent",
              )}
            />
          )}
        </aside>
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
                title={`${currentItem.label} is not available here`}
                description="This deployment does not serve that page. Pick a destination from the sidebar."
              />
            ) : (
              <Outlet />
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
