import { FiDollarSign, FiTrash2, FiUsers } from "react-icons/fi"
import type { RowMenuAction } from "@/design-system/actions/RowMenu"
import {
  type Rule,
  type RuleAction,
  stopDefaultWords,
} from "@/features/rules/rules"
import { limitLine } from "@/features/spend/limits"

type MenuItem = Omit<RowMenuAction, "id" | "run">

const ITEM: Record<RuleAction, (rule: Rule) => MenuItem> = {
  "change-limit": (rule) => ({
    label: "Change spend limit…",
    hint: rule.spend?.target?.own
      ? limitLine(rule.spend.target.own)
      : undefined,
    icon: FiDollarSign,
  }),
  "remove-limit": () => ({
    label: "Remove spend limit…",
    hint: "Requests will no longer be limited by it",
    icon: FiTrash2,
    isDestructive: true,
  }),
  "change-default": (rule) => ({
    label: `Change the limit for everyone in ${rule.teamDefault?.team ?? "team"}…`,
    hint: "Applies to people who join later too",
    icon: FiUsers,
  }),
  "stop-default": (rule) => ({
    label: "Stop giving this limit to people who join later…",
    hint: stopDefaultWords(rule.people?.length ?? 0),
    icon: FiTrash2,
    isDestructive: true,
  }),
}

/** A rule's allowed actions as its row menu lists them. */
export function ruleMenuItems(
  rule: Rule,
  actions: readonly RuleAction[],
  run: (action: RuleAction) => void,
): RowMenuAction[] {
  return actions.map((action) => ({
    id: action,
    ...ITEM[action](rule),
    run: () => run(action),
  }))
}
