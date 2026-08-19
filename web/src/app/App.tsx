import { RouterProvider } from "@tanstack/react-router"
import { useEffect, useState } from "react"
import { HybridLanding } from "@/app/HybridLanding"
import { router } from "@/app/router"
import type { DeploymentBootstrap } from "@/client"
import { useAuth } from "@/features/auth/AuthContext"
import { Login } from "@/features/auth/Login"
import { AcceptInvitationPage } from "@/features/invitations/AcceptInvitationPage"
import { ErrorBanner } from "@/shared/components/ui"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider, useDeployment } from "@/shared/hooks/useDeployment"

/**
 * The hash path, live: it changes without a reload (following an emailed
 * link while a tab is already open, or the accept page navigating away when
 * it is done), and `DeploymentRoot` has to notice, unlike the bootstrap and
 * auth state everything else here reads once per load.
 */
function useHashPath(): string {
  const [hash, setHash] = useState(() => window.location.hash)
  useEffect(() => {
    const onHashChange = () => setHash(window.location.hash)
    window.addEventListener("hashchange", onHashChange)
    return () => window.removeEventListener("hashchange", onHashChange)
  }, [])
  return hash
}

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
 * Which of this deployment's roots renders, decided from the bootstrap and
 * (for one of them) the URL, rather than the route table: signing in is the
 * one decision no route gets to make, and accepting an invitation is one no
 * *session* gets to require. No page below here reads the deployment mode
 * again.
 */
function DeploymentRoot() {
  const { deployment_type, session_type } = useDeployment()
  const { isAuthenticated } = useAuth()
  const hash = useHashPath()

  // A hybrid gateway is data-plane only: otari.ai owns its organizations,
  // credentials, routing, budgets and usage, and a second management UI beside
  // that one is what the deployment contract rules out. Hosted otari.ai serves
  // the same dashboard as standalone, so it falls through. Checked first: a
  // hybrid gateway holds no tenancy state, so an invitation link reaching one
  // is a link this deployment cannot honor, and the landing page's own
  // explanation is more useful here than a page that would just 404.
  if (deployment_type === "hybrid") {
    return <HybridLanding />
  }

  // The one URL every visitor may reach without a session or the master key:
  // the recipient of an emailed invitation holds neither. Every route under
  // `src/routes/` lives behind the auth gate below, on purpose, so this has to
  // be checked ahead of it rather than added there. `AcceptInvitationPage`
  // itself changes the hash away from this prefix once it is done, which is
  // what makes this reactive to the URL rather than only to the first paint.
  if (hash.startsWith("#/accept-invitation")) {
    return <AcceptInvitationPage />
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
