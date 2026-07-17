import { Button } from "@heroui/react";
import { clsx } from "clsx";
import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent, ReactNode } from "react";

import { useAuth } from "@/auth/AuthContext";
import { UpdatePrompt } from "@/components/UpdatePrompt";

export type PageKey = "models" | "settings";

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
  key: PageKey;
  label: string;
  icon: ReactNode;
}

const NAV: NavItem[] = [
  {
    key: "models",
    label: "Models",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 shrink-0">
        <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z" strokeLinejoin="round" />
        <path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    key: "settings",
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

export function AppShell({
  page,
  onNavigate,
  children,
}: {
  page: PageKey;
  onNavigate: (page: PageKey) => void;
  children: ReactNode;
}) {
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
    <div className={clsx("flex min-h-full", resizing && "cursor-col-resize select-none")}>
      <aside
        ref={asideRef}
        style={{ width }}
        className={clsx(
          "relative flex shrink-0 flex-col border-r border-[var(--otari-line)] bg-[var(--otari-surface)]",
          !resizing && "transition-[width] duration-150",
        )}
      >
        <div className={clsx("flex items-center py-5", collapsed ? "justify-center px-0" : "gap-2.5 px-5")}>
          <img src="/favicon.svg" alt="" className="h-7 w-7 shrink-0" />
          {collapsed ? null : <span className="text-base font-semibold text-[var(--otari-ink)]">Otari</span>}
        </div>
        <nav className={clsx("flex flex-col gap-1", collapsed ? "px-2" : "px-3")}>
          {NAV.map((item) => (
            <button
              key={item.key}
              type="button"
              onClick={() => onNavigate(item.key)}
              aria-current={page === item.key ? "page" : undefined}
              aria-label={collapsed ? item.label : undefined}
              title={collapsed ? item.label : undefined}
              className={clsx(
                "flex items-center rounded-lg py-2 text-sm font-medium transition-colors",
                collapsed ? "justify-center px-0" : "gap-3 px-3",
                page === item.key
                  ? "bg-[var(--otari-brand-tint)] text-[var(--otari-brand-dark)]"
                  : "text-[var(--otari-muted)] hover:bg-[var(--otari-bg)] hover:text-[var(--otari-ink)]",
              )}
            >
              {item.icon}
              {collapsed ? null : item.label}
            </button>
          ))}
        </nav>
        <div className={clsx("mt-auto flex flex-col gap-2 py-5", collapsed ? "items-center px-2" : "px-5")}>
          <button
            type="button"
            onClick={() => setCollapsed((value) => !value)}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-pressed={collapsed}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            className={clsx(
              "flex items-center rounded-lg py-2 text-sm font-medium text-[var(--otari-muted)] transition-colors hover:bg-[var(--otari-bg)] hover:text-[var(--otari-ink)]",
              collapsed ? "justify-center px-0" : "gap-3 px-3",
            )}
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className={clsx("h-5 w-5 shrink-0 transition-transform", collapsed && "rotate-180")}
            >
              <path d="M15 6l-6 6 6 6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            {collapsed ? null : "Collapse"}
          </button>
          <Button size="sm" variant="outline" onPress={logout} aria-label="Sign out">
            {collapsed ? (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-4 w-4">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M16 17l5-5-5-5M21 12H9" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            ) : (
              "Sign out"
            )}
          </Button>
        </div>
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
          <UpdatePrompt />
          {children}
        </div>
      </main>
    </div>
  );
}
