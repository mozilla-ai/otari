/**
 * The rules of the Invite people dialog, as pure functions: which roles the
 * caller may grant, what each one means, and what is wrong with the addresses
 * before anything is sent.
 */

import type {
  BulkInviteOrganizationMembersResult,
  MembershipRole,
} from "@/client"
import { looksLikeAddress, parseAddresses } from "@/shared/helpers/addresses"

/** The role they hold in the workspace they join: a viewer views, anyone else is a member there. */
export function teamRoleFor(role: MembershipRole): "member" | "viewer" {
  return role === "viewer" ? "viewer" : "member"
}

/** The help under Workspace, by the role picked. */
export function teamHelp(role: MembershipRole): string {
  return role === "owner" || role === "admin"
    ? "Where their own keys go. Owners and admins see every workspace."
    : `Where their keys and sessions go. They join it as a ${teamRoleFor(role)}.`
}

/**
 * What is wrong with the addresses, or undefined when they can be sent. Every
 * malformed address is named, so it is fixed before the round trip.
 */
export function addressProblem(text: string): string | undefined {
  const addresses = parseAddresses(text)
  if (addresses.length === 0) return "Enter at least one email address."
  const bad = addresses.filter((address) => !looksLikeAddress(address))
  if (bad.length === 0) return undefined
  return bad.length === 1
    ? `${bad[0]} is not an email address. Correct it or take it out.`
    : `${bad.join(", ")} are not email addresses. Correct them or take them out.`
}

/** The primary's label: one person or none is "Invite". */
export function inviteLabel(count: number): string {
  return count > 1 ? `Invite ${count} people` : "Invite"
}

/** Whether the result needs saying, or the dialog can just close. */
export function needsOutcome(
  result: BulkInviteOrganizationMembersResult,
): boolean {
  return (
    result.failed.length > 0 || result.invited.some((one) => !one.mail_sent)
  )
}

/** The result view's title and the sentence under it. */
export function outcomeCopy(result: BulkInviteOrganizationMembersResult): {
  title: string
  description: string
} {
  const total = result.invited.length + result.failed.length
  const mailed = result.invited.filter((one) => one.mail_sent)
  const links = result.invited.length - mailed.length
  const parts = [
    mailed.length === 1
      ? `An invitation email is on its way to ${mailed[0].email}.`
      : mailed.length > 1
        ? `Invitation emails are on their way to ${mailed.length} people.`
        : "",
    links === 1
      ? "Share this link with the person it names."
      : links > 1
        ? "Share each link with the person it names."
        : "",
    result.failed.length === 1
      ? "This address was not invited:"
      : result.failed.length > 1
        ? "These addresses were not invited:"
        : "",
  ]
  return {
    title: `${result.invited.length} of ${total} invited`,
    description: parts.filter(Boolean).join(" "),
  }
}
