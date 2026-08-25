import { createFileRoute } from "@tanstack/react-router"

import { DeploymentAccountsPage } from "@/features/admin/DeploymentAccountsPage"

export const Route = createFileRoute("/admin/accounts")({
  component: DeploymentAccountsPage,
})
