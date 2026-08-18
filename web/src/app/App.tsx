import { RouterProvider } from "@tanstack/react-router"
import { HybridLanding } from "@/app/HybridLanding"
import { router } from "@/app/router"
import type { DeploymentBootstrap } from "@/client"
import { useAuth } from "@/features/auth/AuthContext"
import { Login } from "@/features/auth/Login"
import { ErrorBanner } from "@/shared/components/ui"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider, useDeployment } from "@/shared/hooks/useDeployment"

export default function App({
  bootstrap,
}: {
  bootstrap: DeploymentBootstrap | null
}) {
  // Null means /v1/bootstrap did not answer (see main.tsx). The app deliberately
  // has no fallback deployment to assume: rendering a management dashboard at a
  // gateway that does not serve one is the failure this contract exists to
  // prevent, so say what happened instead.
  if (!bootstrap) {
    return (
      <div className="flex min-h-full items-center justify-center p-6">
        <div className="w-full max-w-md">
          <ErrorBanner
            error={
              new Error(
                "Could not reach the gateway, so the dashboard does not know what it is connected to. Check that it is running, then reload.",
              )
            }
          />
        </div>
      </div>
    )
  }

  return (
    <DeploymentProvider value={bootstrap}>
      <DeploymentRoot />
    </DeploymentProvider>
  )
}

/**
 * Which of the three roots this deployment gets, decided from the bootstrap
 * alone. No page below here reads the deployment mode again.
 */
function DeploymentRoot() {
  const { deployment_type, session_type } = useDeployment()
  const { isAuthenticated } = useAuth()

  // A hybrid gateway is data-plane only: otari.ai owns its organizations,
  // credentials, routing, budgets and usage, and a second management UI beside
  // that one is what the deployment contract rules out. Hosted otari.ai serves
  // the same dashboard as standalone, so it falls through.
  if (deployment_type === "hybrid") {
    return <HybridLanding />
  }

  // Any deployment that issues a session needs one before the shell renders.
  // Only the local operator signs in *here*, since a hosted session is minted by
  // otari.ai's own flow, so a hosted deployment reaching this line is a routing
  // bug to fix when that half lands. Gating on "issues no session" rather than on
  // "issues this one" is what makes that bug a wrong screen instead of an
  // unauthenticated shell whose every query 401s in a loop.
  if (session_type !== "none" && !isAuthenticated) {
    return <Login />
  }

  // Auth gates the router rather than living inside it: signing in is the one
  // decision no route gets to make. The route table and the shell it renders
  // into are in src/routes, wired up in src/app/router.tsx.
  //
  // The selected workspace wraps the router because the shell's switcher and the
  // pages below it read the same selection, and it is seeded from the
  // organization context, which needs a session: inside the auth gate, never
  // above it.
  return (
    <SelectedWorkspaceProvider>
      <RouterProvider router={router} />
    </SelectedWorkspaceProvider>
  )
}
