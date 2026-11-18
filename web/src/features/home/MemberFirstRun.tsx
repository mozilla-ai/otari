import { Button } from "@/design-system/actions/Button"
import { Dot } from "@/design-system/indicators/Dot"
import { useOpenCreateKey } from "@/features/keys/openCreateKey"

/**
 * In place of Needs attention while the caller has run nothing in the team:
 * the one path to a first session for someone who may start one, and what a
 * viewer may do instead. The figures and the empty table stay under it.
 */
export function MemberFirstRun({
  workspace,
  canCreateKey,
}: {
  workspace: string
  canCreateKey: boolean
}) {
  const openCreateKey = useOpenCreateKey()
  if (!canCreateKey) {
    return (
      <section className="flex items-baseline gap-3 border-t border-border pt-6">
        <Dot className="relative -top-0.5 shrink-0 bg-accent" />
        <p className="max-w-prose text-sm text-muted">
          You can see what runs through {workspace}. To run agents of your own,
          ask an owner or admin to make you a member.
        </p>
      </section>
    )
  }
  return (
    <section className="flex flex-wrap items-center justify-between gap-4 border-t border-border pt-6">
      <div className="flex items-start gap-3">
        <Dot className="mt-2.5 shrink-0 bg-accent" />
        <div className="flex flex-col gap-1">
          <h2 className="text-display-sub">Start your first agent</h2>
          <p className="max-w-prose text-sm text-muted">
            You have not run anything through {workspace} yet. You need one API
            key.
          </p>
        </div>
      </div>
      <Button
        variant="primary"
        onPress={() => openCreateKey("member_first_run")}
      >
        Create API key
      </Button>
    </section>
  )
}
