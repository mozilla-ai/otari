import { Button } from "@heroui/react";
import { clsx } from "clsx";
import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent, ReactNode } from "react";
import { NavLink, Outlet } from "react-router-dom";

import { useAuth } from "@/auth/AuthContext";
import { PricingWarning } from "@/components/PricingWarning";
import { UpdatePrompt } from "@/components/UpdatePrompt";

const MIN_SIDEBAR = 200;
const MAX_SIDEBAR = 480;
const DEFAULT_SIDEBAR = 240;
const COLLAPSED_SIDEBAR = 60;
const SIDEBAR_WIDTH_KEY = "otari.dashboard.sidebarWidth";
const SIDEBAR_COLLAPSED_KEY = "otari.dashboard.sidebarCollapsed";
const SIDEBAR_STEP = 16;

const clampSidebar = (width: number) => Math.min(MAX_SIDEBAR, Math.max(MIN_SIDEBAR, width));

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
}

// Sidebar groups, in display order. "Catalog" is what the gateway serves
// (providers, their models, and aliases over them); "Access" is who may call it
// (keys today; users/budgets later); "system" holds standalone config with no
// header. Grouping keeps the list legible as the dashboard grows.
const NAV_SECTIONS: { key: string; label?: string }[] = [
  { key: "catalog", label: "Catalog" },
  { key: "access", label: "Access" },
  { key: "system" },
];

const NAV: NavItem[] = [
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
  const [sidebarWidth, setSidebarWidth] = useState<number>(readStoredSidebarWidth);
  const [collapsed, setCollapsed] = useState<boolean>(readStoredCollapsed);
  const [resizing, setResizing] = useState(false);

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

  return (
    <div className={clsx("relative flex h-full flex-col overflow-hidden", resizing && "cursor-col-resize select-none")}>
      <header className="flex shrink-0 items-center justify-between border-b border-[var(--otari-line)] bg-[var(--otari-surface)] px-5 py-3">
        <div className="flex items-center gap-2.5">
          <img src="/favicon.svg" alt="" className="h-7 w-7 shrink-0" />
          <span className="text-base font-semibold text-[var(--otari-ink)]">Otari</span>
        </div>
        <Button size="sm" variant="outline" onPress={logout} aria-label="Sign out">
          Sign out
        </Button>
      </header>
      <UpdatePrompt />
      <PricingWarning />
      <div className="flex min-h-0 flex-1">
        <aside
          ref={asideRef}
          style={{ width }}
          className={clsx(
            "relative flex shrink-0 flex-col border-r border-[var(--otari-line)] bg-[var(--otari-surface)]",
            !resizing && "transition-[width] duration-150",
          )}
        >
          {/* A round chevron on the sidebar's edge toggles collapse — floats over
              the border for a polished, VS Code / Notion-style affordance. */}
          <button
            type="button"
            onClick={() => setCollapsed((value) => !value)}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-pressed={collapsed}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="absolute -right-3 top-4 z-30 flex h-6 w-6 items-center justify-center rounded-full border border-[var(--otari-line)] bg-[var(--otari-surface)] text-[var(--otari-muted)] shadow-sm transition-colors hover:border-[var(--otari-brand)] hover:text-[var(--otari-brand-dark)]"
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
          <nav className={clsx("flex flex-col py-4", collapsed ? "px-2" : "px-3")}>
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
                  {!collapsed && section.label ? (
                    <div className="px-3 pb-1 text-[11px] font-semibold tracking-wider text-[var(--otari-muted)] uppercase">
                      {section.label}
                    </div>
                  ) : null}
                  {sectionIndex > 0 && (collapsed || !section.label) ? (
                    <div className="mx-1 mb-2 border-t border-[var(--otari-line)]" />
                  ) : null}
                  <div className="flex flex-col gap-1">
                    {items.map((item) => (
                      <NavLink
                        key={item.to}
                        to={item.to}
                        aria-label={collapsed ? item.label : undefined}
                        title={collapsed ? item.label : undefined}
                        className={({ isActive }) =>
                          clsx(
                            "flex items-center rounded-lg py-2 text-sm font-medium transition-colors",
                            collapsed ? "justify-center px-0" : "gap-3 px-3",
                            isActive
                              ? "bg-[var(--otari-brand-tint)] text-[var(--otari-brand-dark)]"
                              : "text-[var(--otari-muted)] hover:bg-[var(--otari-bg)] hover:text-[var(--otari-ink)]",
                          )
                        }
                      >
                        {item.icon}
                        {collapsed ? null : item.label}
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
              collapsed ? "mx-2 justify-center px-0" : "mx-3 gap-2 px-3",
            )}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-4 w-4 shrink-0">
              <path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" strokeLinejoin="round" />
            </svg>
            {collapsed ? null : (
              <span className="flex-1">
                otari.ai <span aria-hidden>↗</span>
              </span>
            )}
          </a>
          {collapsed ? null : (
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
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto flex max-w-[1800px] flex-col gap-6 px-6 py-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
