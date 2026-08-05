import { lazy, Suspense } from "react";
import type { ReactNode } from "react";
import { HashRouter, Navigate, Route, Routes } from "react-router-dom";

import { useAuth } from "@/auth/AuthContext";
import { AppShell } from "@/components/AppShell";
import { Login } from "@/components/Login";

const ActivityPage = lazy(async () => ({
  default: (await import("@/pages/ActivityPage")).ActivityPage,
}));
const RoutingPage = lazy(async () => ({
  default: (await import("@/pages/RoutingPage")).RoutingPage,
}));
const BudgetsPage = lazy(async () => ({
  default: (await import("@/pages/BudgetsPage")).BudgetsPage,
}));
const DocsPage = lazy(async () => ({
  default: (await import("@/pages/DocsPage")).DocsPage,
}));
const KeysPage = lazy(async () => ({
  default: (await import("@/pages/KeysPage")).KeysPage,
}));
const ModelsPage = lazy(async () => ({
  default: (await import("@/pages/ModelsPage")).ModelsPage,
}));
const OverviewIndex = lazy(async () => ({
  default: (await import("@/pages/OverviewPage")).OverviewIndex,
}));
const ProvidersPage = lazy(async () => ({
  default: (await import("@/pages/ProvidersPage")).ProvidersPage,
}));
const SettingsPage = lazy(async () => ({
  default: (await import("@/pages/SettingsPage")).SettingsPage,
}));
const ToolsGuardrailsPage = lazy(async () => ({
  default: (await import("@/pages/ToolsGuardrailsPage")).ToolsGuardrailsPage,
}));
const UsagePage = lazy(async () => ({
  default: (await import("@/pages/UsagePage")).UsagePage,
}));
const UsersPage = lazy(async () => ({
  default: (await import("@/pages/UsersPage")).UsersPage,
}));

function withPageLoading(page: ReactNode) {
  return (
    <Suspense fallback={<div role="status">Loading page…</div>}>
      {page}
    </Suspense>
  );
}

export default function App() {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Login />;
  }

  return (
    <HashRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={withPageLoading(<OverviewIndex />)} />
          <Route path="providers" element={withPageLoading(<ProvidersPage />)} />
          <Route path="keys" element={withPageLoading(<KeysPage />)} />
          <Route path="users" element={withPageLoading(<UsersPage />)} />
          <Route path="budgets" element={withPageLoading(<BudgetsPage />)} />
          <Route path="activity" element={withPageLoading(<ActivityPage />)} />
          <Route path="usage" element={withPageLoading(<UsagePage />)} />
          <Route path="models" element={withPageLoading(<ModelsPage />)} />
          {/* Aliases were folded into Routing, which lists and manages them as the
              one-target policies they are. The old path redirects so bookmarks and
              any link still pointing at it keep working. */}
          <Route path="aliases" element={<Navigate to="/routing" replace />} />
          <Route path="routing" element={withPageLoading(<RoutingPage />)} />
          <Route path="tools" element={withPageLoading(<ToolsGuardrailsPage />)} />
          <Route path="settings" element={withPageLoading(<SettingsPage />)} />
          <Route path="docs" element={withPageLoading(<DocsPage />)} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}
