import { Button } from "@heroui/react";
import { clsx } from "clsx";
import type { ReactNode } from "react";

import { useAuth } from "@/auth/AuthContext";
import { useHealth } from "@/api/hooks";

export type PageKey = "overview" | "usage" | "models" | "keys" | "users";

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
    key: "keys",
    label: "API keys",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5">
        <circle cx="8" cy="15" r="4" />
        <path d="M10.85 12.15 19 4M16 7l3 3M14 9l2 2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    key: "users",
    label: "Users",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5">
        <circle cx="12" cy="8" r="4" />
        <path d="M4 21c0-4 3.5-6 8-6s8 2 8 6" strokeLinecap="round" />
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

  return (
    <div className="flex min-h-full">
      <aside className="flex w-60 shrink-0 flex-col border-r border-[var(--otari-line)] bg-[var(--otari-surface)]">
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
      </aside>
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-6xl flex-col gap-6 p-8">{children}</div>
      </main>
    </div>
  );
}
