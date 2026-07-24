import { Button } from "@heroui/react";
import { clsx } from "clsx";
import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent, ReactNode } from "react";
import { NavLink, Outlet } from "react-router-dom";

import { useAuth } from "@/auth/AuthContext";
import { ConnectionStatus } from "@/components/ConnectionStatus";
import { PricingWarning } from "@/components/PricingWarning";
import { UpdatePrompt } from "@/components/UpdatePrompt";

const MIN_SIDEBAR = 200;
const MAX_SIDEBAR = 480;
const DEFAULT_SIDEBAR = 240;
const COLLAPSED_SIDEBAR = 60;
const SIDEBAR_WIDTH_KEY = "otari.dashboard.sidebarWidth";
const SIDEBAR_COLLAPSED_KEY = "otari.dashboard.sidebarCollapsed";
const SIDEBAR_STEP = 16;

// Below this width the sidebar's fixed footprint squashes page content, so it
// switches to an off-canvas drawer toggled from the header. Matches Tailwind's
// `md` breakpoint (the classes that hide the trigger and drawer chrome use `md:`).
const MOBILE_QUERY = "(max-width: 767px)";

const clampSidebar = (width: number) => Math.min(MAX_SIDEBAR, Math.max(MIN_SIDEBAR, width));

function readIsMobile(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
  return window.matchMedia(MOBILE_QUERY).matches;
}

const FOCUSABLE_SELECTOR = 'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])';

// Visible, focusable descendants of a container, in DOM order. offsetParent is
// null for display:none nodes (e.g. the desktop-only collapse chevron on mobile),
// so filtering on it keeps the focus trap's first/last from landing on a hidden
// control that can't actually take focus.
function getFocusable(container: HTMLElement | null): HTMLElement[] {
  if (!container) return [];
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (el) => el.offsetParent !== null || el === document.activeElement,
  );
}

function readStoredSidebarWidth(): number {
  if (typeof window === "undefined") return DEFAULT_SIDEBAR;
  try {
    const raw = window.localStorage.getItem(SIDEBAR_WIDTH_KEY);
    const parsed = raw ? Number.parseInt(raw, 10) : Number.NaN;
    return Number.isNaN(parsed) ? DEFAULT_SIDEBAR : clampSidebar(parsed);
  } catch {
    // Storage can throw when disabled (e.g. blocked cookies / private mode);
    // fall back to the default rather than white-screening the shell.
    return DEFAULT_SIDEBAR;
  }
}

function readStoredCollapsed(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

interface NavItem {
  to: string;
  label: string;
  section: string;
  icon: ReactNode;
  // NavLink matches by prefix; the index route ("/") needs `end` or it stays
  // highlighted on every page.
  end?: boolean;
}

// Sidebar groups, in display order. "Observability" is what the gateway did
// (the request log today; usage analytics and an overview dashboard later) and
// leads the sidebar; "Catalog" is what the gateway serves (providers, their
// models, and aliases over them); "Access" is who may call it (keys, users,
// budgets); "system" holds standalone config with no header. Grouping keeps the
// list legible as the dashboard grows.
const NAV_SECTIONS: { key: string; label?: string }[] = [
  { key: "home" },
  { key: "observability", label: "Observability" },
  { key: "catalog", label: "Catalog" },
  { key: "access", label: "Access" },
  { key: "system" },
];

const NAV: NavItem[] = [
  {
    to: "/",
    section: "home",
    label: "Overview",
    // The index/home, so it leads the sidebar above the grouped sections.
    end: true,
    icon: (
      // Four panes: an at-a-glance dashboard of the gateway.
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 shrink-0">
        <rect x="3.5" y="3.5" width="7" height="7" rx="1.5" strokeLinejoin="round" />
        <rect x="13.5" y="3.5" width="7" height="7" rx="1.5" strokeLinejoin="round" />
        <rect x="3.5" y="13.5" width="7" height="7" rx="1.5" strokeLinejoin="round" />
        <rect x="13.5" y="13.5" width="7" height="7" rx="1.5" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    to: "/activity",
    section: "observability",
    label: "Activity",
    icon: (
      // A pulse/activity line: the per-request log of what the gateway served.
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 shrink-0">
        <path d="M3 12h4l2.5-6 4 12 2.5-6H21" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    to: "/usage",
    section: "observability",
    label: "Usage",
    icon: (
      // A bar chart: aggregate spend and volume over time, beside the activity log.
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 shrink-0">
        <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    to: "/providers",
    section: "catalog",
    label: "Providers",
    icon: (
      // A server stack: upstream provider services, distinct from the API-keys key.
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 shrink-0">
        <rect x="3.5" y="4.5" width="17" height="6" rx="1.5" strokeLinejoin="round" />
        <rect x="3.5" y="13.5" width="17" height="6" rx="1.5" strokeLinejoin="round" />
        <path d="M7 7.5h.01M7 16.5h.01" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    to: "/users",
    section: "access",
    label: "Users",
    icon: (
      // Two figures: the principals that keys and budgets attach to.
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 shrink-0">
        <circle cx="9" cy="8" r="3.2" strokeLinejoin="round" />
        <path d="M3.5 19a5.5 5.5 0 0 1 11 0" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M16 5.2a3.2 3.2 0 0 1 0 5.6M17.5 19a5.5 5.5 0 0 0-3-4.9" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    to: "/keys",
    section: "access",
    label: "API keys",
    icon: (
      // The key glyph now belongs to API keys (Providers moved to a server stack).
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 shrink-0">
        <circle cx="7.5" cy="15.5" r="3.5" />
        <path d="M10 13l7-7M14 5l3 3M16.5 7.5l2-2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    to: "/budgets",
    section: "access",
    label: "Budgets",
    icon: (
      // A wallet: the spending limits callers are held to, alongside the keys.
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 shrink-0">
        <path d="M3 7.5A1.5 1.5 0 0 1 4.5 6H18a1.5 1.5 0 0 1 1.5 1.5V9" strokeLinejoin="round" />
        <rect x="3" y="7.5" width="18" height="12" rx="1.5" strokeLinejoin="round" />
        <path d="M16 13.5h.01" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M21 12v3h-3.5a1.5 1.5 0 0 1 0-3H21z" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    to: "/models",
    section: "catalog",
    label: "Models",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 shrink-0">
        <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z" strokeLinejoin="round" />
        <path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    to: "/aliases",
    section: "catalog",
    label: "Aliases",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 shrink-0">
        <path d="M20.6 13.4L13.4 20.6a2 2 0 0 1-2.8 0l-7-7A2 2 0 0 1 3 12.2V5a2 2 0 0 1 2-2h7.2a2 2 0 0 1 1.4.6l7 7a2 2 0 0 1 0 2.8z" strokeLinejoin="round" />
        <circle cx="7.5" cy="7.5" r="1.5" />
      </svg>
    ),
  },
  {
    to: "/tools",
    section: "system",
    label: "Tools & Guardrails",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 shrink-0">
        <path
          d="M14.7 6.3a4 4 0 0 1 5 5l-8.4 8.4a2 2 0 0 1-2.8 0l-2.2-2.2a2 2 0 0 1 0-2.8z"
          strokeLinejoin="round"
        />
        <path d="M12 9 5 16" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    to: "/settings",
    section: "system",
    label: "Settings",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 shrink-0">
        <circle cx="12" cy="12" r="3" />
        <path
          d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"
          strokeLinejoin="round"
        />
      </svg>
    ),
  },
];

export function AppShell() {
  const { logout } = useAuth();

  const asideRef = useRef<HTMLElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const [sidebarWidth, setSidebarWidth] = useState<number>(readStoredSidebarWidth);
  const [collapsed, setCollapsed] = useState<boolean>(readStoredCollapsed);
  const [resizing, setResizing] = useState(false);
  const [isMobile, setIsMobile] = useState<boolean>(readIsMobile);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  // Track the mobile breakpoint so the sidebar can render as an off-canvas
  // drawer below it and as the resizable rail above it. Closing the drawer when
  // the viewport grows past the breakpoint keeps a stale open state from leaving
  // a fixed overlay stranded over the desktop layout.
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const query = window.matchMedia(MOBILE_QUERY);
    const onChange = (event: MediaQueryListEvent) => {
      setIsMobile(event.matches);
      if (!event.matches) setMobileNavOpen(false);
    };
    // Safari < 14 (and some older engines) only expose the deprecated
    // addListener/removeListener; fall back to it so the shell doesn't throw.
    if (typeof query.addEventListener === "function") {
      query.addEventListener("change", onChange);
      return () => query.removeEventListener("change", onChange);
    }
    query.addListener(onChange);
    return () => query.removeListener(onChange);
  }, []);

  // Escape closes the drawer, matching the dismissible-overlay convention.
  useEffect(() => {
    if (!mobileNavOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileNavOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [mobileNavOpen]);

  // Focus management for the mobile drawer, which is a modal overlay: move focus
  // into it when it opens and restore focus to the toggle when it closes, so
  // keyboard and screen-reader users are neither stranded inside a hidden panel
  // nor dropped back to the top of the document. The isMobile guard means a
  // breakpoint change to desktop (which also closes the drawer) never yanks focus
  // to the now-hidden toggle.
  useEffect(() => {
    if (!isMobile) return;
    if (mobileNavOpen) {
      asideRef.current?.focus();
    } else if (asideRef.current?.contains(document.activeElement)) {
      toggleRef.current?.focus();
    }
  }, [isMobile, mobileNavOpen]);

  // Keep Tab within the open drawer so focus cannot wander to the page behind the
  // backdrop. Paired with the aside being inert while closed, this bounds keyboard
  // focus to whichever surface is actually interactive.
  const trapFocus = useCallback((event: ReactKeyboardEvent<HTMLElement>) => {
    if (event.key !== "Tab") return;
    const focusables = getFocusable(asideRef.current);
    if (focusables.length === 0) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement;
    if (event.shiftKey && (active === first || active === asideRef.current)) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus();
    }
  }, []);

  useEffect(() => {
    const id = window.setTimeout(() => {
      try {
        window.localStorage.setItem(SIDEBAR_WIDTH_KEY, String(Math.round(sidebarWidth)));
      } catch {
        // Ignore storage errors; the width still applies for this session.
      }
    }, 200);
    return () => window.clearTimeout(id);
  }, [sidebarWidth]);

  useEffect(() => {
    try {
      window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      // Ignore storage errors; the collapse state still applies for this session.
    }
  }, [collapsed]);

  const startResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setResizing(true);
  }, []);

  const moveResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
    const left = asideRef.current?.getBoundingClientRect().left ?? 0;
    setSidebarWidth(clampSidebar(event.clientX - left));
  }, []);

  const endResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setResizing(false);
  }, []);

  const nudgeResize = useCallback((event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      setSidebarWidth((width) => clampSidebar(width - SIDEBAR_STEP));
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      setSidebarWidth((width) => clampSidebar(width + SIDEBAR_STEP));
    }
  }, []);

  const width = collapsed ? COLLAPSED_SIDEBAR : sidebarWidth;
  // The collapse rail and resize handle are desktop-only affordances; on mobile
  // the drawer always shows the full-width, labelled nav.
  const effectiveCollapsed = isMobile ? false : collapsed;
  // While the mobile drawer is open, make everything behind it (header + page)
  // inert so a modal really is modal: aria-modal alone isn't universally honored,
  // so this is what keeps an AT virtual cursor and Tab out of the obscured
  // controls, not just the aside's own focus trap.
  const backgroundInert = isMobile && mobileNavOpen ? true : undefined;

  return (
    <div className={clsx("relative flex h-full flex-col overflow-hidden", resizing && "cursor-col-resize select-none")}>
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
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5">
              <path d="M4 6h16M4 12h16M4 18h16" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          <img src="/favicon.svg" alt="" className="h-7 w-7 shrink-0" />
          <span className="text-base font-semibold text-[var(--otari-ink)]">Otari</span>
        </div>
        <Button size="sm" variant="outline" onPress={logout} aria-label="Sign out">
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
              : clsx("relative shrink-0", !resizing && "transition-[width] duration-150"),
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
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              className={clsx("h-3.5 w-3.5 transition-transform", collapsed && "rotate-180")}
            >
              <path d="M15 6l-6 6 6 6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          <nav className={clsx("flex flex-col py-4", effectiveCollapsed ? "px-2" : "px-3")}>
            {NAV_SECTIONS.map((section, sectionIndex) => {
              const items = NAV.filter((item) => item.section === section.key);
              if (items.length === 0) {
                return null;
              }
              return (
                <div key={section.key} className={sectionIndex > 0 ? "mt-4" : undefined}>
                  {/* A header labels each group when expanded; a thin divider stands
                      in for it when the sidebar is collapsed, or when a group has no
                      header of its own (e.g. Settings) to set it off from the group
                      above. */}
                  {!effectiveCollapsed && section.label ? (
                    <div className="px-3 pb-1 text-[11px] font-semibold tracking-wider text-[var(--otari-muted)] uppercase">
                      {section.label}
                    </div>
                  ) : null}
                  {sectionIndex > 0 && (effectiveCollapsed || !section.label) ? (
                    <div className="mx-1 mb-2 border-t border-[var(--otari-line)]" />
                  ) : null}
                  <div className="flex flex-col gap-1">
                    {items.map((item) => (
                      <NavLink
                        key={item.to}
                        to={item.to}
                        end={item.end}
                        // Tapping a destination dismisses the mobile drawer so the
                        // page it navigated to is visible, not hidden behind it.
                        onClick={() => setMobileNavOpen(false)}
                        aria-label={effectiveCollapsed ? item.label : undefined}
                        title={effectiveCollapsed ? item.label : undefined}
                        className={({ isActive }) =>
                          clsx(
                            "flex items-center rounded-lg py-2 text-sm font-medium transition-colors",
                            effectiveCollapsed ? "justify-center px-0" : "gap-3 px-3",
                            isActive
                              ? "bg-[var(--otari-brand-tint)] text-[var(--otari-brand-dark)]"
                              : "text-[var(--otari-muted)] hover:bg-[var(--otari-bg)] hover:text-[var(--otari-ink)]",
                          )
                        }
                      >
                        {item.icon}
                        {effectiveCollapsed ? null : item.label}
                      </NavLink>
                    ))}
                  </div>
                </div>
              );
            })}
          </nav>
          {/* Subtle pointer to the hosted product; muted until hovered. */}
          <a
            href="https://otari.ai"
            target="_blank"
            rel="noreferrer"
            title="otari.ai — the hosted Otari gateway"
            className={clsx(
              "mt-auto mb-3 flex items-center rounded-lg py-2 text-xs font-medium text-[var(--otari-muted)] transition-colors hover:bg-[var(--otari-bg)] hover:text-[var(--otari-brand-dark)]",
              effectiveCollapsed ? "mx-2 justify-center px-0" : "mx-3 gap-2 px-3",
            )}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-4 w-4 shrink-0">
              <path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" strokeLinejoin="round" />
            </svg>
            {effectiveCollapsed ? null : (
              <span className="flex-1">
                otari.ai <span aria-hidden>↗</span>
              </span>
            )}
          </a>
          {collapsed || isMobile ? null : (
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
        <main inert={backgroundInert} className="flex-1 overflow-y-auto">
          <div className="mx-auto flex max-w-[1800px] flex-col gap-6 px-4 py-5 md:px-6 md:py-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
