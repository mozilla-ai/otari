import type { ApiKey } from "@/client"

/**
 * Whether a row of Home's API keys cut is a key the caller can open on API
 * keys and act on. The keys endpoint answers that per caller: the deployment's
 * operator reads every key (`/keys`); anyone else reads their own and, as an
 * owner or admin, the organization's service keys (`/organizations/me/keys`).
 * A row the usage log has and that list does not, someone else's key, would
 * open a page that does not show it and a write the server refuses with 404.
 */
export function canOpenInKeys(
  filterKey: string | null,
  listed: readonly Pick<ApiKey, "id">[],
): boolean {
  return listed.some((key) => key.id === filterKey)
}
