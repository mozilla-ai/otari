import type { OrganizationGuardrail } from "@/client"
import {
  type MandateConsequence,
  mandateConsequence,
} from "@/features/guardrails/buildState"
import {
  checksLabel,
  ifItCantRunLabel,
  mandatesFor,
  runsOnLabel,
} from "@/features/guardrails/guardrailRows"
import {
  moderationRuleId,
  moderationScope,
  type RuleScope,
} from "@/features/rules/rules"
import { useOrganizationGuardrailDefinitions } from "@/shared/api/guardrails"
import { useOrganizationContext } from "@/shared/api/organizations"
import { useOrganizationGuardrails } from "@/shared/api/tools"
import { useWorkspaces } from "@/shared/api/workspaces"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

/** One content moderation guardrail running in the selected workspace. */
export interface ModerationRow {
  /** The guardrail's page, as the registry names it. */
  id: string
  profile: string
  scope: RuleScope
  runsOn: string
  checks: string
  mode: "Block" | "Monitor"
  ifItCantRun: string
  /** What it is doing to requests when it is not what it says. */
  consequence: MandateConsequence
}

export interface Moderation {
  rows: ModerationRow[]
  /** The workspace the rows run in. */
  workspace?: { id: string; name: string }
  isLoading: boolean
  error: unknown
}

/**
 * The organization's content moderation that runs on the selected
 * workspace's requests. A paused one runs nowhere, so it is not listed. Owners
 * and admins only: the server refuses anyone else both lists, since they name
 * the vendors the organization has accounts with.
 */
export function useModeration(isManager: boolean): Moderation {
  const { selected } = useSelectedWorkspace()
  const context = useOrganizationContext().data
  const mandates = useOrganizationGuardrails(isManager)
  const definitions = useOrganizationGuardrailDefinitions(isManager)
  const workspaces = useWorkspaces(isManager)

  const defined = definitions.data ?? []
  const workspaceName = new Map(
    (workspaces.data ?? []).map((one) => [one.id, one.name]),
  )
  const teamName = (id: string) => workspaceName.get(id) ?? "A workspace"
  const running: OrganizationGuardrail[] = selected
    ? mandatesFor(selected.workspace_id, mandates.data ?? []).filter(
        (mandate) => mandate.enabled,
      )
    : []

  return {
    rows: running
      .map(
        (mandate): ModerationRow => ({
          id: moderationRuleId(mandate.id),
          profile: mandate.profile,
          scope: moderationScope(
            mandate,
            teamName,
            context?.organization?.name,
          ),
          runsOn: runsOnLabel(mandate, defined),
          checks: checksLabel(mandate),
          mode: mandate.mode === "block" ? "Block" : "Monitor",
          ifItCantRun: ifItCantRunLabel(mandate),
          consequence: mandateConsequence(mandate, defined),
        }),
      )
      .sort((a, b) => a.profile.localeCompare(b.profile)),
    workspace: selected
      ? { id: selected.workspace_id, name: selected.name }
      : undefined,
    isLoading:
      isManager &&
      (mandates.isLoading || definitions.isLoading || workspaces.isLoading),
    error: mandates.error ?? definitions.error ?? workspaces.error,
  }
}
