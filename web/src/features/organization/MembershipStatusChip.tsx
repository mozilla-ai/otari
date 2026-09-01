import { Chip } from "@heroui/react"

import { membershipLabel } from "./roles"

/**
 * A membership status as a chip, shared by the rosters that show one.
 *
 * `"blocked"` is not a membership status and no roster row carries it: it is the
 * gateway refusing this person's keys, which the organization roster passes in
 * place of the status because the membership is active while every request the
 * person makes fails.
 */
export function MembershipStatusChip({ status }: { status: string }) {
  if (status === "blocked") {
    return (
      <Chip size="sm" color="danger">
        Blocked
      </Chip>
    )
  }
  if (status === "active") {
    return (
      <Chip size="sm" color="accent">
        Active
      </Chip>
    )
  }
  return (
    <Chip size="sm" color={status === "suspended" ? "warning" : "default"}>
      {membershipLabel(status)}
    </Chip>
  )
}
