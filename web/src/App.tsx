import { useState } from "react";

import { useAuth } from "@/auth/AuthContext";
import { AppShell } from "@/components/AppShell";
import type { PageKey } from "@/components/AppShell";
import { Login } from "@/components/Login";
import { ModelsPage } from "@/pages/ModelsPage";
import { SettingsPage } from "@/pages/SettingsPage";

export default function App() {
  const { isAuthenticated } = useAuth();
  const [page, setPage] = useState<PageKey>("models");

  if (!isAuthenticated) {
    return <Login />;
  }

  return (
    <AppShell page={page} onNavigate={setPage}>
      {page === "models" ? <ModelsPage /> : null}
      {page === "settings" ? <SettingsPage /> : null}
    </AppShell>
  );
}
