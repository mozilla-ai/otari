import type { UpdateCheckStatus } from "@/client"
import { useDeploymentOperator } from "@/shared/api/organizations"
import { useUpdateCheck } from "@/shared/api/updateCheck"
import { useSurfaces } from "@/shared/hooks/useDeployment"

/**
 * The update check, for a deployment operator on a deployment that publishes
 * it, and `undefined` for everyone else or until it has loaded.
 *
 * Read through `isOperator` as well as gating the query on it, for the reason
 * `PricingWarning` does: a disabled query still hands back what sits under its
 * key, and a caller demoted mid-session must not keep the prompt.
 */
export function useOperatorUpdateCheck(): UpdateCheckStatus | undefined {
  const { isOperator } = useDeploymentOperator()
  const isHosted = useSurfaces()("update_check")
  const query = useUpdateCheck(isOperator && isHosted)
  return isOperator && isHosted ? query.data : undefined
}

/** The host a check is sent to, for the sentence that names it. */
export function endpointHost(endpoint: string): string {
  try {
    return new URL(endpoint).host
  } catch {
    return endpoint
  }
}

export const UPDATE_CHECK_DOC = "update-check.md"

/**
 * The newest of some release versions (`0.30.10` is newer than `0.30.9`, which
 * a string sort gets backwards), or `""` for none.
 */
export function newestVersion(versions: string[]): string {
  const parts = (version: string) => version.split(".").map(Number)
  return versions.reduce((newest, version) => {
    const [candidate, held] = [parts(version), parts(newest)]
    const difference =
      Array.from(
        { length: Math.max(candidate.length, held.length) },
        (_part, index) => (candidate[index] ?? 0) - (held[index] ?? 0),
      ).find((delta) => delta !== 0) ?? 0
    return difference > 0 ? version : newest
  }, "")
}
