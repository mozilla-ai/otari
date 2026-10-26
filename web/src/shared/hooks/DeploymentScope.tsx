import { createContext, type ReactNode, useContext } from "react"

const DeploymentScopeContext = createContext(false)

/**
 * Marks the pages below it as the deployment's own, so the data hooks they call
 * read the deployment-wide routes (`/usage`, `/keys`, `/routing/policies`,
 * `/traces`) instead of the caller's organization's.
 *
 * The page decides which data it shows, not the caller's role: the same Usage
 * page reads one organization in that organization's rail and every tenant in
 * the deployment's, whoever is looking. The role only decides whether the
 * deployment's pages are offered at all, which the nav registry and the server
 * both enforce.
 */
export function DeploymentScope({ children }: { children: ReactNode }) {
  return (
    <DeploymentScopeContext value={true}>{children}</DeploymentScopeContext>
  )
}

/** Whether the page this renders in is one of the deployment's own. */
export function useIsDeploymentScope(): boolean {
  return useContext(DeploymentScopeContext)
}
