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
 * all: `sign_in_methods`, which the app itself can change. Two acts change it.
 *
 * Claiming a deployment retires master-key sign-in
 * (`PUT /v1/auth/password`, otari#649), which is the app changing the server's
 * answer rather than the server changing it underneath: `sign_in_methods` goes
 * from `["master_key"]` to `["password"]` the moment that call succeeds, and
 * the response says so in `master_key_sign_in_retired`. That field is the
 * server's own answer and not "this call succeeded": only the operator's
 * password claims the deployment (otari#702), so a member changing theirs on an
 * unclaimed one gets `false` and nothing here moves. Every consumer of that
 * fact has to move together, or the tab that claimed keeps a sign-in screen
 * offering the credential the gateway now refuses and a password page that asks
 * to claim a deployment already claimed. So the claim reports itself through
 * `useRetireMasterKeySignIn` and this provider serves the corrected bootstrap
 * from then on.
 *
 * Registering the first passkey, or deleting the last one, is the second
 * (otari#652): the gateway publishes `passkey` exactly while some credential
 * could answer a sign-in, so the account page reports the change through
 * `useOfferPasskeySignIn`. Unlike the claim this one is reversible, because
 * deleting a passkey is, so it carries the value rather than being one-way.
 */

import type { ReactNode } from "react"
import { createContext, useContext, useMemo, useState } from "react"

import type { DeploymentBootstrap } from "@/client"

const DeploymentContext = createContext<DeploymentBootstrap | null>(null)
const RetireMasterKeySignInContext = createContext<(() => void) | null>(null)
const OfferPasskeySignInContext = createContext<
  ((offered: boolean) => void) | null
>(null)

// What `_sign_in_methods` answers once the operator identity holds a password
// (`api/routes/bootstrap.py`). Named rather than inlined so the override is
// visibly the server's own value and not a shape invented here.
const PASSWORD_ONLY: DeploymentBootstrap["sign_in_methods"] = ["password"]
// The gateway sorts `sign_in_methods`, so a corrected list is sorted too rather
// than appended to: a consumer comparing the array, not just probing it with
// `includes`, should not be able to tell a corrected bootstrap from a fetched one.
const PASSKEY: DeploymentBootstrap["sign_in_methods"][number] = "passkey"

export function DeploymentProvider({
  value,
  children,
}: {
  value: DeploymentBootstrap
  children: ReactNode
}) {
  const [masterKeyRetired, setMasterKeyRetired] = useState(false)
  // null until this tab changes it, so the server's own answer stands: a
  // boolean seeded from `value` would make a re-render after a reload look like
  // a correction.
  const [passkeysOffered, setPasskeysOffered] = useState<boolean | null>(null)

  // Identity-stable in the case that always holds before either correction and
  // forever after a reload: `effective` *is* `value` unless this tab did the
  // claiming or changed its passkeys, so the context does not hand every
  // consumer a new object on each render.
  const effective = useMemo(() => {
    if (!masterKeyRetired && passkeysOffered === null) {
      return value
    }
    const typed = masterKeyRetired ? PASSWORD_ONLY : value.sign_in_methods
    const withoutPasskey = typed.filter((method) => method !== PASSKEY)
    const offered =
      passkeysOffered === null
        ? value.sign_in_methods.includes(PASSKEY)
        : passkeysOffered
    return {
      ...value,
      sign_in_methods: offered
        ? [...withoutPasskey, PASSKEY].sort()
        : withoutPasskey,
    }
  }, [value, masterKeyRetired, passkeysOffered])

  return (
    <DeploymentContext.Provider value={effective}>
      <RetireMasterKeySignInContext.Provider
        value={() => setMasterKeyRetired(true)}
      >
        <OfferPasskeySignInContext.Provider value={setPasskeysOffered}>
          {children}
        </OfferPasskeySignInContext.Provider>
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

/**
 * Report whether this deployment now holds a passkey that could sign somebody in.
 *
 * Called by the account page after registering the first one or deleting the
 * last, so the sign-in screen a later sign-out lands on offers the button (or
 * stops offering it) without a reload. Carries the value rather than being
 * one-way, because unlike claiming a deployment this is reversible.
 */
export function useOfferPasskeySignIn(): (offered: boolean) => void {
  const offer = useContext(OfferPasskeySignInContext)
  if (!offer) {
    throw new Error(
      "useOfferPasskeySignIn must be used within a DeploymentProvider",
    )
  }
  return offer
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
