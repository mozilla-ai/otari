import { Button } from "@heroui/react";
import { clsx } from "clsx";
import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent, ReactNode } from "react";

import { useAuth } from "@/auth/AuthContext";
import { useHealth } from "@/api/hooks";
import { UpdatePrompt } from "@/components/UpdatePrompt";

export type PageKey = "overview" | "usage" | "models" | "settings";

const MIN_SIDEBAR = 200;
const MAX_SIDEBAR = 480;
const DEFAULT_SIDEBAR = 240;
const SIDEBAR_WIDTH_KEY = "otari.dashboard.sidebarWidth";
const SIDEBAR_STEP = 16;

const clampSidebar = (width: number) => Math.min(MAX_SIDEBAR, Math.max(MIN_SIDEBAR, width));

function readStoredSidebarWidth(): number {
  if (typeof window === "undefined") return DEFAULT_SIDEBAR;
  const raw = window.localStorage.getItem(SIDEBAR_WIDTH_KEY);
  const parsed = raw ? Number.parseInt(raw, 10) : Number.NaN;
  return Number.isNaN(parsed) ? DEFAULT_SIDEBAR : clampSidebar(parsed);
}

interface NavItem {
  key: PageKey;
  label: string;
  icon: ReactNode;
}

const NAV: NavItem[] = [
  {
    key: "overview",
    label: "Overview",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5">
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </svg>
    ),
  },
  {
    key: "usage",
    label: "Usage",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5">
        <path d="M3 3v18h18" strokeLinecap="round" />
        <path d="M7 14l3-4 4 3 4-6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    key: "models",
    label: "Models",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5">
        <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z" strokeLinejoin="round" />
        <path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    key: "settings",
    label: "Settings",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5">
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
  const health = useHealth();

  const asideRef = useRef<HTMLElement>(null);
  const [sidebarWidth, setSidebarWidth] = useState<number>(readStoredSidebarWidth);
  const [resizing, setResizing] = useState(false);

  useEffect(() => {
    const id = window.setTimeout(() => {
      window.localStorage.setItem(SIDEBAR_WIDTH_KEY, String(Math.round(sidebarWidth)));
    }, 200);
    return () => window.clearTimeout(id);
  }, [sidebarWidth]);

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

  return (
    <div className={clsx("flex min-h-full", resizing && "cursor-col-resize select-none")}>
      <aside
        ref={asideRef}
        style={{ width: sidebarWidth }}
        className="relative flex shrink-0 flex-col border-r border-[var(--otari-line)] bg-[var(--otari-surface)]"
      >
        <div className="flex items-center gap-2.5 px-5 py-5">
          <img src="/favicon.svg" alt="" className="h-7 w-7" />
          <span className="text-base font-semibold text-[var(--otari-ink)]">Otari</span>
        </div>
        <nav className="flex flex-col gap-1 px-3">
          {NAV.map((item) => (
            <button
              key={item.key}
              type="button"
              onClick={() => onNavigate(item.key)}
              aria-current={page === item.key ? "page" : undefined}
              className={clsx(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                page === item.key
                  ? "bg-[var(--otari-brand-tint)] text-[var(--otari-brand-dark)]"
                  : "text-[var(--otari-muted)] hover:bg-[var(--otari-bg)] hover:text-[var(--otari-ink)]",
              )}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>
        <div className="mt-auto flex flex-col gap-3 px-5 py-5 text-xs text-[var(--otari-muted)]">
          <span>
            {health.data?.version ? `v${health.data.version}` : "Gateway"}
            {health.data?.mode ? ` · ${health.data.mode}` : ""}
          </span>
          <Button size="sm" variant="outline" onPress={logout}>
            Sign out
          </Button>
        </div>
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
