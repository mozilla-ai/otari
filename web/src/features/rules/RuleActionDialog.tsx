import { ChangeDefaultDialog } from "@/features/rules/ChangeDefaultDialog"
import { RemoveLimitDialog } from "@/features/rules/RemoveLimitDialog"
import type { Rule, RuleAction } from "@/features/rules/rules"
import { StopDefaultDialog } from "@/features/rules/StopDefaultDialog"
import { limitDialogCopy } from "@/features/spend/limitCopy"
import { SetLimitDialog } from "@/features/spend/SetLimitDialog"

export interface PendingAction {
  action: RuleAction
  rule: Rule
}

/** The dialog a rule's menu item opens. */
export function RuleActionDialog({
  pending,
  onClose,
}: {
  pending: PendingAction
  onClose: () => void
}) {
  const { action, rule } = pending
  const target = rule.spend?.target
  const teamDefault = rule.teamDefault
  if (action === "change-limit" && target) {
    return (
      <SetLimitDialog
        target={target}
        {...limitDialogCopy(target)}
        onClose={onClose}
      />
    )
  }
  if (action === "remove-limit" && target) {
    return (
      <RemoveLimitDialog name={rule.name} target={target} onClose={onClose} />
    )
  }
  if (action === "change-default" && teamDefault) {
    return <ChangeDefaultDialog teamDefault={teamDefault} onClose={onClose} />
  }
  if (action === "stop-default" && teamDefault) {
    return (
      <StopDefaultDialog
        teamDefault={teamDefault}
        peopleCount={rule.people?.length ?? 0}
        onClose={onClose}
      />
    )
  }
  return null
}
