import type { OrganizationMember } from "@/client"
import { Dot } from "@/design-system/indicators/Dot"
import { MemberAvatar } from "@/features/account/avatar/MemberAvatar"
import { memberLabel } from "./roles"

/** A member's model access, resolved to its words and its ink. */
export interface MemberAccessLabel {
  text: string
  tone: "muted" | "normal" | "danger"
}

const ACCESS_INK: Record<MemberAccessLabel["tone"], string> = {
  danger: "text-danger",
  muted: "text-subtle",
  normal: "text-muted",
}

/**
 * The dot-and-word mark this surface uses in place of a chip.
 *
 * The dot is what reads as a mark rather than a value in the address slot. A
 * chip's vertical padding takes the row off the pitch the other rows hold.
 */
function SsoMark() {
  return (
    <span className="flex shrink-0 items-center gap-1.5 text-mono-caption text-subtle">
      <Dot className="bg-text-subtle" />
      SSO
    </span>
  )
}

/**
 * A member's name over the facts that qualify them.
 *
 * The second line leads with the member's address, or with the SSO mark where
 * the account holds none, and carries `access` after it. A member with no name
 * is already headed by their address, so the line does not repeat it.
 * `access` is omitted for a reader not entitled to it, leaving a member with no
 * address and no mark with one line.
 */
export function MemberIdentityCell({
  member,
  access,
}: {
  member: OrganizationMember
  access?: MemberAccessLabel
}) {
  // memberLabel falls back to the address, so a member with no name already
  // shows it as the heading and this line would repeat it.
  const repeatsTheHeading = !member.full_name
  // The mark stands in for an address, so it reads the account rather than
  // whether this line prints one.
  const lead = member.email ? (
    repeatsTheHeading ? null : (
      <span className="truncate" title={member.email}>
        {member.email}
      </span>
    )
  ) : member.is_sso_account ? (
    <SsoMark />
  ) : null
  return (
    <div className="flex items-center gap-3">
      <MemberAvatar userId={member.user_id} face={member.avatar_face} />
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="text-body">{memberLabel(member)}</span>
        {lead || access ? (
          // Does not wrap: wrapping takes the row off its 58px pitch. The
          // address is the only part that gives way, and keeps its full value
          // in the title.
          <span className="flex items-center gap-1.5 text-caption text-nowrap">
            {lead}
            {lead && access ? (
              <span aria-hidden className="text-subtle">
                ·
              </span>
            ) : null}
            {access ? (
              <span className={`shrink-0 ${ACCESS_INK[access.tone]}`}>
                {access.text}
              </span>
            ) : null}
          </span>
        ) : null}
      </div>
    </div>
  )
}
