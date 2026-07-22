import { HashRouter, Navigate, Route, Routes } from "react-router-dom";

import { useAuth } from "@/auth/AuthContext";
import { AppShell } from "@/components/AppShell";
import { Login } from "@/components/Login";
import { ActivityPage } from "@/pages/ActivityPage";
import { AliasesPage } from "@/pages/AliasesPage";
import { BudgetsPage } from "@/pages/BudgetsPage";
import { KeysPage } from "@/pages/KeysPage";
import { ModelsPage } from "@/pages/ModelsPage";
import { OverviewIndex } from "@/pages/OverviewPage";
import { ProvidersPage } from "@/pages/ProvidersPage";
import { SettingsPage } from "@/pages/SettingsPage";
import { ToolsGuardrailsPage } from "@/pages/ToolsGuardrailsPage";
import { UsagePage } from "@/pages/UsagePage";
import { UsersPage } from "@/pages/UsersPage";

export default function App() {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Login />;
  }

  return (
    <HashRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<OverviewIndex />} />
          <Route path="providers" element={<ProvidersPage />} />
          <Route path="keys" element={<KeysPage />} />
          <Route path="users" element={<UsersPage />} />
          <Route path="budgets" element={<BudgetsPage />} />
          <Route path="activity" element={<ActivityPage />} />
          <Route path="usage" element={<UsagePage />} />
          <Route path="models" element={<ModelsPage />} />
          <Route path="aliases" element={<AliasesPage />} />
          <Route path="tools" element={<ToolsGuardrailsPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}
