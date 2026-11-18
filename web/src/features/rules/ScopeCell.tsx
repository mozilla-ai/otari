import { MemberAvatar } from "@/features/account/avatar/MemberAvatar"
import type { RuleScope } from "@/features/rules/rules"

/** Who a rule reaches: the subject, then what kind of subject it is. */
export function ScopeCell({ scope }: { scope: RuleScope }) {
  const subject = (text: string) => (
    <span className="min-w-0 text-body md:truncate">{text}</span>
  )
  const kind = (text: string) => (
    <span className="shrink-0 text-sm text-subtle">{text}</span>
  )
  switch (scope.kind) {
    case "everyone":
      // The one word says who; the organization's name is what may give.
      return (
        <span className="flex min-w-0 flex-wrap items-center gap-x-2 md:flex-nowrap">
          <span className="shrink-0 text-body">Everyone</span>
          {scope.organization ? (
            <span className="min-w-0 text-sm text-subtle md:truncate">
              in {scope.organization}
            </span>
          ) : null}
        </span>
      )
    case "team":
      return (
        <span className="flex min-w-0 flex-wrap items-center gap-x-2 md:flex-nowrap">
          {subject(scope.team)}
          {kind("workspace")}
        </span>
      )
    case "teams":
      return (
        <span className="flex min-w-0 flex-wrap items-center gap-x-2 md:flex-nowrap">
          {scope.teams.length > 0
            ? subject(scope.teams.join(", "))
            : kind("Nobody yet")}
          {scope.teams.length > 0
            ? kind(scope.teams.length === 1 ? "workspace" : "workspaces")
            : null}
        </span>
      )
    case "each":
      return (
        <span className="flex min-w-0 flex-wrap items-center gap-x-2 md:flex-nowrap">
          {subject(`Each person in ${scope.teams.join(", ")}`)}
          <span className="shrink-0 text-mono-caption text-subtle tabular-nums">
            {scope.count}
          </span>
          {scope.includesYou ? kind("you included") : null}
        </span>
      )
    case "person":
      return (
        <span className="flex min-w-0 flex-wrap items-center gap-x-2.5 md:flex-nowrap">
          <MemberAvatar
            userId={scope.avatarUserId}
            face={scope.avatarFace}
            isCompact
          />
          {subject(scope.name)}
        </span>
      )
    case "key":
      return scope.name ? (
        <span className="flex min-w-0 flex-wrap items-center gap-x-2 md:flex-nowrap">
          {subject(scope.name)}
          {kind("API key")}
        </span>
      ) : (
        <span className="text-sm text-subtle">An API key</span>
      )
  }
}
