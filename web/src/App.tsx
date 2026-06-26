import { useState } from "react";

import { useAuth } from "@/auth/AuthContext";
import { AppShell } from "@/components/AppShell";
import type { PageKey } from "@/components/AppShell";
import { Login } from "@/components/Login";
import { KeysPage } from "@/pages/KeysPage";
import { ModelsPage } from "@/pages/ModelsPage";
import { OverviewPage } from "@/pages/OverviewPage";
import { UsagePage } from "@/pages/UsagePage";
import { UsersPage } from "@/pages/UsersPage";

export default function App() {
  const { isAuthenticated } = useAuth();
  const [page, setPage] = useState<PageKey>("overview");

  if (!isAuthenticated) {
    return <Login />;
  }

  return (
    <AppShell page={page} onNavigate={setPage}>
      {page === "overview" ? <OverviewPage /> : null}
      {page === "usage" ? <UsagePage /> : null}
      {page === "models" ? <ModelsPage /> : null}
      {page === "keys" ? <KeysPage /> : null}
      {page === "users" ? <UsersPage /> : null}
    </AppShell>
  );
}
