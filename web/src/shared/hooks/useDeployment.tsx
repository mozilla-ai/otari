/**
 * The deployment context the shell renders from, read once per page load.
 *
 * `main.tsx` fetches `/v1/bootstrap` before it mounts anything and hands the
 * answer to `App`, which puts it here. That ordering is the point: which
 * deployment served this page decides whether a sign-in screen, a management
 * dashboard, or a data-plane landing page is the right first paint, so the app
 * must not have to guess it and then correct itself.
 *
 * It lives in `shared/` rather than in `app/` because features read it and a
 * feature may not import the composition root. It is a context rather than a
 * query because it describes the server that served this page: it cannot change
 * without a reload, and it must survive the cache clear a sign-out performs.
 *
 * One field is the exception, and it is the reason this provider holds state at
 * all. Claiming a deployment retires master-key sign-in
 * (`PUT /v1/auth/password`, otari#649), which is the app changing the server's
 * answer rather than the server changing it underneath: `sign_in_methods` goes
 * from `["master_key"]` to `["password"]` the moment that call succeeds, and
 * the response says so in `master_key_sign_in_retired`. That field is the
 * server's own answer and not "this call succeeded": only the operator's
 * password claims the deployment (otari#702), so a member changing theirs on an
 * unclaimed one gets `false` and nothing here moves. Every consumer of that
 * fact has to move together, or the tab that claimed keeps a sign-in screen
 * offering the credential the gateway now refuses, an account menu naming a
 * session kind that ended, and a password page that asks to claim a deployment
 * already claimed. So the claim
 * reports itself through `useRetireMasterKeySignIn` and this provider serves
 * the corrected bootstrap from then on.
 */

import type { ReactNode } from "react"
import { createContext, useContext, useMemo, useState } from "react"

import type { DeploymentBootstrap } from "@/client"

const DeploymentContext = createContext<DeploymentBootstrap | null>(null)
const RetireMasterKeySignInContext = createContext<(() => void) | null>(null)

// What `_sign_in_methods` answers once the operator identity holds a password
// (`api/routes/bootstrap.py`). Named rather than inlined so the override is
// visibly the server's own value and not a shape invented here.
const PASSWORD_ONLY: DeploymentBootstrap["sign_in_methods"] = ["password"]

export function DeploymentProvider({
  value,
  children,
}: {
  value: DeploymentBootstrap
  children: ReactNode
}) {
  const [masterKeyRetired, setMasterKeyRetired] = useState(false)

  // Identity-stable in the case that always holds before a claim and forever
  // after a reload: `effective` *is* `value` unless this tab did the claiming,
  // so the context does not hand every consumer a new object on each render.
  const effective = masterKeyRetired
    ? { ...value, sign_in_methods: PASSWORD_ONLY }
    : value

  return (
    <DeploymentContext.Provider value={effective}>
      <RetireMasterKeySignInContext.Provider
        value={() => setMasterKeyRetired(true)}
      >
        {children}
      </RetireMasterKeySignInContext.Provider>
    </DeploymentContext.Provider>
  )
}

/**
 * Report that this deployment has been claimed, so the rest of the tab stops
 * offering the master key as a sign-in.
 *
 * Only `PUT /v1/auth/password` may call this, and only on a response that says
 * `master_key_sign_in_retired`. It is one-way, like the act it describes: no
 * endpoint clears a password, so nothing puts the master key back.
 */
export function useRetireMasterKeySignIn(): () => void {
  const retire = useContext(RetireMasterKeySignInContext)
  if (!retire) {
    throw new Error(
      "useRetireMasterKeySignIn must be used within a DeploymentProvider",
    )
  }
  return retire
}

/** The bootstrap this page was served with. Throws outside the provider. */
export function useDeployment(): DeploymentBootstrap {
  const bootstrap = useContext(DeploymentContext)
  if (!bootstrap) {
    throw new Error("useDeployment must be used within a DeploymentProvider")
  }
  return bootstrap
}

/**
 * Whether this deployment hosts a given management surface.
 *
 * Surfaces, not capabilities: otari.ai spends "capability" on the entitlement
 * axis (is this org licensed for it), down to a nav item's `capability` field.
 * This is the deployment axis (does this process host it at all), and the two
 * vocabularies meet in one shell at M5. See ARCHITECTURE.md.
 *
 * The client-side half of a gate, so it can only hide a surface, never grant
 * one: the server still authorizes every request behind it.
 */
export function useSurfaces(): (surface: string) => boolean {
  const { surfaces } = useDeployment()
  return useMemo(() => {
    const hosted = new Set(surfaces)
    return (surface: string) => hosted.has(surface)
  }, [surfaces])
}
