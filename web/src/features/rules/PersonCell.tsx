import { MemberAvatar } from "@/features/account/avatar/MemberAvatar"
import type { RulePerson } from "@/features/rules/rules"

/** One person under a rule: their face and name, and "you" on the caller. */
export function PersonCell({ person }: { person: RulePerson }) {
  return (
    <span className="flex min-w-0 items-center gap-2.5">
      <MemberAvatar
        userId={person.avatarUserId}
        face={person.avatarFace}
        isCompact
      />
      <span className="min-w-0 truncate text-body">{person.name}</span>
      {person.isYou ? (
        <span className="shrink-0 text-sm text-subtle">you</span>
      ) : null}
    </span>
  )
}
