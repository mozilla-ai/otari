import { RouterProvider } from "@tanstack/react-router"
import {
  lazy,
  Suspense,
  useLayoutEffect,
  useState,
  useSyncExternalStore,
} from "react"
import { PublicPageTitle } from "@/app/PublicPageTitle"
import { router } from "@/app/router"
import { ErrorBoundary } from "@/design-system/feedback/ErrorBoundary"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { useAuth } from "@/features/auth/AuthContext"
import { Login } from "@/features/auth/Login"
import { PublicAuthPage } from "@/features/auth/PublicAuthPage"
import {
  isAcceptInvitationPath,
  isSsoAuthorizePath,
  type PublicAuthPath,
  publicAuthPath,
} from "@/features/auth/publicAuthPaths"
import { AcceptInvitationPage } from "@/features/invitations/AcceptInvitationPage"
import {
  publicCatalogPath,
  publicCatalogRoute,
  takeRememberedModel,
} from "@/features/models/publicCatalog"
import { publicProvidersPath } from "@/features/providers/publicProviders"
import {
  currentRoute,
  routeHref,
  routePath,
} from "@/shared/helpers/appLocation"
import type { WireBootstrap } from "@/shared/helpers/bootstrap"
import { normalizeBootstrap } from "@/shared/helpers/bootstrap"
import { providerDisplayName } from "@/shared/helpers/providers"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider, useDeployment } from "@/shared/hooks/useDeployment"

/**
 * The route, live: it changes without a reload (following an emailed
 * link while a tab is already open, or the accept page navigating away when
 * it is done), and both `App` and `DeploymentRoot` have to notice, unlike the
 * bootstrap and auth state everything else here reads once per load. Read once
 * in `App` and passed down rather than called in both, so one listener decides
 * which branch renders and when the boundary around it resets.
 */
// Split out of the entry chunk: it pulls the whole Models feature (both views,
// the drawer, the table), which every visitor of every deployment would
// otherwise download, `public_catalog: false` included. The route tree already
// splits the signed-in copy; this is the same split from the other side.
const PublicCatalogPage = lazy(() =>
  import("@/features/models/PublicCatalogPage").then((module) => ({
    default: module.PublicCatalogPage,
  })),
)

// Split for the same reason as the catalog above.
const PublicProvidersPage = lazy(() =>
  import("@/features/providers/PublicProvidersPage").then((module) => ({
    default: module.PublicProvidersPage,
  })),
)

function subscribeToLocation(onChange: () => void): () => void {
  return router.history.subscribe(onChange)
}

// Read off the router's history rather than `window.location`: it notifies as
// a navigation is queued, before the browser's address catches up, and it also
// hears `navigateTo` and Back, since it wraps `pushState` and `replaceState`.
function readRoute(): string {
  return currentRoute(router.history.location)
}

function useRoute(): string {
  return useSyncExternalStore(subscribeToLocation, readRoute)
}

export default function App({
  bootstrap,
}: {
  // `WireBootstrap`, not `DeploymentBootstrap`: this is the one component that
  // takes the payload as it came off the wire, and an older gateway sends fewer
  // fields than the generated type promises. `normalizeBootstrap` below is
  // where it becomes the complete shape everything under here reads. Never
  // absent: `DeploymentGate` renders this only once the bootstrap answered.
  bootstrap: WireBootstrap
}) {
  // Read here rather than only in `DeploymentRoot` because the boundary below
  // resets on it: which branch renders is a function of the route, so a throw in
  // one of them must not outlive the navigation away from it.
  const route = useRoute()

  // Outside the provider rather than inside it, because the provider's own
  // correction memo reads `sign_in_methods` and is therefore one of the things
  // that can throw on a bootstrap this dashboard did not expect.
  return (
    <ErrorBoundary resetKey={route}>
      <DeploymentProvider value={normalizeBootstrap(bootstrap)}>
        <DeploymentRoot route={route} />
      </DeploymentProvider>
    </ErrorBoundary>
  )
}

/**
 * Which of this deployment's roots renders, decided from the bootstrap and
 * (for one of them) the URL, rather than the route table: signing in is the
 * one decision no route gets to make, and accepting an invitation is one no
 * *session* gets to require.
 */
const AppAuthorizationPage = lazy(() =>
  import("@/features/auth/AppAuthorizationPage").then((module) => ({
    default: module.AppAuthorizationPage,
  })),
)

function DeploymentRoot({ route }: { route: string }) {
  const { public_catalog, open_signup } = useDeployment()
  const { isAuthenticated } = useAuth()

  // The one URL every visitor may reach without a session or the master key:
  // the recipient of an emailed invitation holds neither. Every route under
  // `src/routes/` lives behind the auth gate below, on purpose, so this has to
  // be checked ahead of it rather than added there. `AcceptInvitationPage`
  // itself changes the route away from this prefix once it is done, which is
  // what makes this reactive to the URL rather than only to the first paint.
  //
  // Keyed on the route itself, not just gated by it: the page reads its token
  // from the route exactly once, in its own initial state, so a second
  // invitation link opened in the same tab (pasted over the first without a
  // full reload) would otherwise re-render the *same* component instance and
  // keep validating and accepting the first token. The key forces React to
  // tear down and remount on any route change under this prefix, which is what
  // makes "once" mean once per link rather than once per tab.
  if (isAcceptInvitationPath(route)) {
    return (
      <PublicPageTitle page="Accept invitation">
        <AcceptInvitationPage key={route} />
      </PublicPageTitle>
    )
  }

  // The rest of the auth surface a visitor may reach without a session
  // (otari#650): claiming a rostered identity, confirming an address from an
  // emailed link, and recovering a forgotten password. Checked here for the
  // same reason as the line above, and keyed on the route for a version of the
  // same reason: `/verify-email` verifies on mount and `/reset-password`
  // holds a typed-in password, so a second emailed link pasted over the first
  // in an open tab has to tear both down rather than re-render on top of the
  // previous link's result.
  const publicAuth = publicAuthPath(route)
  if (publicAuth || isSsoAuthorizePath(route)) {
    // Signup's heading is "Claim your account" where signup is closed, so its
    // tab says the same. A direct link naming an organization's slug carries
    // no entry of its own in that fixed table, so it falls back to the one
    // title every sign-in surface here shares.
    const title =
      publicAuth === "/signup" && !open_signup
        ? "Claim account"
        : publicAuth
          ? PUBLIC_AUTH_TITLES[publicAuth]
          : "Sign in"
    return (
      <PublicPageTitle page={title}>
        <PublicAuthPage path={publicAuth} route={route} key={route} />
      </PublicPageTitle>
    )
  }

  // The catalog, where the deployment has opened it to visitors. Only for a
  // visitor: a signed-in caller reaches the same pages through the router,
  // priced for their organization. Keyed on the route so a second model opened
  // in the tab remounts the view with its own selection.
  const publicCatalog = publicCatalogPath(route)
  if (public_catalog && !isAuthenticated && publicCatalog !== null) {
    return (
      <PublicPageTitle page="Models">
        <Suspense fallback={<PageLoading label="Loading models…" />}>
          <PublicCatalogPage key={route} modelId={publicCatalog.modelId} />
        </Suspense>
      </PublicPageTitle>
    )
  }

  // The providers the catalog is served by, under the same switch and for the
  // same visitor.
  const publicProviders = publicProvidersPath(route)
  if (public_catalog && !isAuthenticated && publicProviders !== null) {
    return (
      <PublicPageTitle
        page={
          publicProviders.name
            ? providerDisplayName(publicProviders.name)
            : "Providers"
        }
      >
        <Suspense fallback={<PageLoading label="Loading providers…" />}>
          <PublicProvidersPage key={route} name={publicProviders.name} />
        </Suspense>
      </PublicPageTitle>
    )
  }

  // The shell renders only behind a session.
  if (!isAuthenticated) {
    return (
      <PublicPageTitle page="Sign in">
        <Login />
      </PublicPageTitle>
    )
  }

  // Auth gates the router rather than living inside it: signing in is the one
  // decision no route gets to make. The route table and the shell it renders
  // into are in src/routes, wired up in src/app/router.tsx.
  if (routePath(route) === "/oauth/authorize") {
    return (
      <Suspense fallback={<PageLoading label="Loading authorization…" />}>
        <AppAuthorizationPage key={route} />
      </Suspense>
    )
  }
  return <SignedInRoot />
}

/**
 * The router, for a session. On its first render it reopens the model a
 * visitor chose on the public catalog before signing in, and only where the
 * session landed on the home page: a deep link the caller followed wins.
 * Done before the router mounts, so the home page never flashes first.
 */
function SignedInRoot() {
  useState(() => {
    const modelId = takeRememberedModel()
    if (modelId && routePath(currentRoute()) === "/") {
      router.history.replace(routeHref(publicCatalogRoute(modelId)))
    }
  })
  // The router outlives this component: signing out unmounts it and signing in
  // mounts it again, and the pages in front of a session can move the address
  // in between (sign-in to recovery and back). On a remount TanStack loads only
  // when it has never loaded, so without this it keeps showing the page it last
  // resolved under the new address. A child's layout effect runs first, so the
  // router is subscribed and has read the address by the time this asks.
  useLayoutEffect(() => {
    const resolved = router.state.resolvedLocation
    if (resolved && resolved.href !== router.latestLocation.href) {
      void router.load()
    }
  }, [])
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

const PUBLIC_AUTH_TITLES: Record<PublicAuthPath, string> = {
  "/signup": "Create account",
  "/check-email": "Check your email",
  "/resend-verification": "Resend verification",
  "/recover-password": "Recover password",
  "/verify-email": "Verify email",
  "/reset-password": "Reset password",
  "/setup": "Set up",
  "/sign-in/google/callback": "Sign in",
  "/sign-in/github/callback": "Sign in",
  "/sign-in/sso/callback": "Sign in",
}
