import { createFileRoute, Outlet } from "@tanstack/react-router"

import { DeploymentScope } from "@/shared/hooks/DeploymentScope"

// A layout, not a page: everything under /deployment reads the deployment-wide
// routes, whoever renders it, because this is what marks it so.
export const Route = createFileRoute("/deployment")({
  component: () => (
    <DeploymentScope>
      <Outlet />
    </DeploymentScope>
  ),
})
