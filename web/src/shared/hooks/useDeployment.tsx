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
 */

import type { ReactNode } from "react"
import { createContext, useContext, useMemo } from "react"

import type { DeploymentBootstrap } from "@/client"

const DeploymentContext = createContext<DeploymentBootstrap | null>(null)

export function DeploymentProvider({
  value,
  children,
}: {
  value: DeploymentBootstrap
  children: ReactNode
}) {
  return (
    <DeploymentContext.Provider value={value}>
      {children}
    </DeploymentContext.Provider>
  )
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
 * Whether this deployment serves a given management capability.
 *
 * The client-side half of a gate, so it can only hide a surface, never grant
 * one: the server still authorizes every request behind it.
 */
export function useCapabilities(): (capability: string) => boolean {
  const { capabilities } = useDeployment()
  return useMemo(() => {
    const available = new Set(capabilities)
    return (capability: string) => available.has(capability)
  }, [capabilities])
}
