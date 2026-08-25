/**
 * The client half of the two changes the deployment admin API refuses.
 *
 * Both are gates the server already enforces (`DeploymentUserService`), written
 * here so a control that would always be refused is disabled with the reason on
 * it rather than offered and then failed. Neither decides who may do what.
 *
 * Directional, matching the server: only the changes that can lock somebody out
 * are blocked, so granting an account back a flag it lost is still offered on
 * both protected rows.
 */

import type { DeploymentUser } from "@/client"

/**
 * Why deactivating this account, or taking its operator access, is refused.
 *
 * `undefined` when it is allowed. The two reasons are answered by the server on
 * the row itself: nothing else the dashboard fetches names the caller's own
 * identity, and the bootstrap marker is not on any other contract.
 */
export function accountLockoutReason(
  account: DeploymentUser,
): string | undefined {
  if (account.is_self) {
    return "This is your own account; another operator has to make this change"
  }
  if (account.is_bootstrap_operator) {
    return "The bootstrap operator is how master-key sign-in reaches this deployment"
  }
  return undefined
}

/** How the account is named in a row, a control's label, and a confirmation. */
export function accountLabel(account: DeploymentUser): string {
  return account.full_name?.trim() || account.email || account.id
}

/**
 * One account's organizations, as the row lists them.
 *
 * A suspended membership is kept and marked, where the organization roster drops
 * it: an account whose every membership is suspended is what this page exists to
 * find, and a row saying "no organizations" for one would hide the finding.
 */
export function organizationSummary(account: DeploymentUser): string {
  // Optional on the generated type because the field carries a server-side
  // default, which the generator reads as "may be omitted". It never is.
  const organizations = account.organizations ?? []
  if (organizations.length === 0) return "None"
  return organizations
    .map((organization) =>
      organization.status === "active"
        ? `${organization.name} (${organization.role})`
        : `${organization.name} (${organization.role}, ${organization.status})`,
    )
    .join(", ")
}
