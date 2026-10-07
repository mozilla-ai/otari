/**
 * The entities a budget can apply to, as the budget form's picker offers them.
 *
 * An entity is a scope plus an optional provider and model, which is exactly what
 * one ceiling row stores, so a picked entity and a held ceiling compare by one key.
 * The form keeps a set of those keys and sends it whole as `applied_to`.
 *
 * Providers and models are entities of the organization: a provider entity is the
 * organization narrowed to that provider, and a model entity is the organization
 * narrowed to that model on its provider. That is how `appliedTo.ts` reads them
 * back, so what the picker offers and what the table names agree.
 */

import type {
  ApiKey,
  AppliedEntity,
  OrganizationMember,
  OrganizationSpendCeiling,
  Workspace,
} from "@/client"
import type { MultiSelectOption } from "@/design-system/forms/MultiSelect"
import { providerInstanceOf } from "@/features/models/modelKey"

/** One entity as a string, so a set of them is a set of strings. */
export function entityKey(entity: AppliedEntity): string {
  return JSON.stringify([
    entity.scope_type,
    entity.scope_id,
    entity.provider_key_id ?? null,
    entity.model ?? null,
  ])
}

export function entityFromKey(key: string): AppliedEntity {
  const [scope_type, scope_id, provider_key_id, model] = JSON.parse(key) as [
    AppliedEntity["scope_type"],
    string,
    string | null,
    string | null,
  ]
  return { scope_type, scope_id, provider_key_id, model }
}

export function ceilingKey(
  ceiling: Pick<
    OrganizationSpendCeiling,
    "scope_type" | "scope_id" | "provider_key_id" | "model"
  >,
): string {
  return entityKey({
    scope_type: ceiling.scope_type as AppliedEntity["scope_type"],
    scope_id: ceiling.scope_id,
    provider_key_id: ceiling.provider_key_id,
    model: ceiling.model,
  })
}

export type EntityGroupId =
  | "workspace"
  | "org_member"
  | "workspace_member"
  | "api_token"
  | "provider"
  | "model"

export interface EntityGroup {
  id: EntityGroupId
  label: string
  countNoun: { one: string; other: string }
  options: MultiSelectOption[]
}

export interface EntitySources {
  organizationId: string
  workspaces: readonly Pick<Workspace, "id" | "name">[]
  members: readonly OrganizationMember[]
  keys: readonly Pick<
    ApiKey,
    "id" | "key_name" | "key_prefix" | "workspace_id"
  >[]
  /** Catalog ids from `GET /v1/models`; only `instance:model` selectors become entities. */
  modelIds: readonly string[]
}

function personLabel(member: OrganizationMember): string {
  return member.full_name || member.email || "Unnamed member"
}

/**
 * Every pickable entity, grouped the way the form shows them.
 *
 * `takenBy` maps an entity's key to what already carries it; such an option is
 * listed but disabled, with the reason as its hint, because one entity carries
 * one budget and the list should say why something cannot be picked.
 */
export function entityGroups(
  sources: EntitySources,
  takenBy: ReadonlyMap<string, string>,
): EntityGroup[] {
  const org = sources.organizationId
  const workspaceName = new Map(sources.workspaces.map((w) => [w.id, w.name]))
  const option = (
    entity: AppliedEntity,
    label: string,
    hint?: string,
  ): MultiSelectOption => {
    const id = entityKey(entity)
    const taken = takenBy.get(id)
    const hints = [hint, taken].filter(Boolean).join(" · ")
    return {
      id,
      label,
      ...(hints ? { hint: hints } : {}),
      ...(taken ? { isDisabled: true } : {}),
    }
  }

  const selectors = [...new Set(sources.modelIds)].sort((a, b) =>
    a.localeCompare(b),
  )
  const instances = [
    ...new Set(selectors.flatMap((id) => providerInstanceOf(id) ?? [])),
  ]

  return [
    {
      id: "workspace",
      label: "Workspaces",
      countNoun: { one: "workspace", other: "workspaces" },
      options: sources.workspaces.map((workspace) =>
        option(
          { scope_type: "workspace", scope_id: workspace.id },
          workspace.name,
        ),
      ),
    },
    {
      id: "org_member",
      label: "Organization members",
      countNoun: { one: "member", other: "members" },
      // A pending invitation has no membership row to cap yet.
      options: sources.members.flatMap((member) =>
        member.organization_member_id
          ? [
              option(
                {
                  scope_type: "org_member",
                  scope_id: member.organization_member_id,
                },
                personLabel(member),
                member.full_name ? (member.email ?? undefined) : undefined,
              ),
            ]
          : [],
      ),
    },
    {
      id: "workspace_member",
      label: "Workspace members",
      countNoun: { one: "member", other: "members" },
      options: sources.members.flatMap((member) =>
        (member.workspaces ?? []).map((placement) =>
          option(
            {
              scope_type: "workspace_member",
              scope_id: placement.workspace_member_id,
            },
            personLabel(member),
            placement.workspace_name,
          ),
        ),
      ),
    },
    {
      id: "api_token",
      label: "API keys",
      countNoun: { one: "key", other: "keys" },
      options: sources.keys.map((key) =>
        option(
          { scope_type: "api_token", scope_id: key.id },
          key.key_name || `${key.key_prefix ?? "Key"}…`,
          workspaceName.get(key.workspace_id),
        ),
      ),
    },
    {
      id: "provider",
      label: "Providers",
      countNoun: { one: "provider", other: "providers" },
      options: instances.map((instance) =>
        option(
          {
            scope_type: "organization",
            scope_id: org,
            provider_key_id: instance,
          },
          instance,
        ),
      ),
    },
    {
      id: "model",
      label: "Models",
      countNoun: { one: "model", other: "models" },
      options: selectors.flatMap((selector) => {
        const instance = providerInstanceOf(selector)
        if (!instance) return []
        return [
          option(
            {
              scope_type: "organization",
              scope_id: org,
              provider_key_id: instance,
              model: selector.slice(instance.length + 1),
            },
            selector,
          ),
        ]
      }),
    },
  ]
}

/**
 * What already carries each entity, other than the budget being edited.
 *
 * `budgetName` answers for the organization's own budgets; a ceiling naming a
 * budget it does not know is one set at the deployment level.
 */
export function takenEntities(
  ceilings: readonly OrganizationSpendCeiling[],
  editingBudgetId: string | undefined,
  budgetName: (budgetId: string) => string | undefined,
): Map<string, string> {
  return new Map(
    ceilings
      .filter((ceiling) => ceiling.budget_id !== editingBudgetId)
      .map((ceiling) => {
        const name = budgetName(ceiling.budget_id)
        return [
          ceilingKey(ceiling),
          name ? `On ${name}` : "On a budget set at the deployment level",
        ]
      }),
  )
}

/**
 * A plain description of an entity no picker group offers, such as a workspace
 * narrowed to one provider, so it can be listed and removed rather than dropped.
 */
export function describeEntity(
  entity: AppliedEntity,
  names: { organizationName: string; workspaces: EntitySources["workspaces"] },
): string {
  const scope = (() => {
    switch (entity.scope_type) {
      case "organization":
        return names.organizationName
      case "workspace":
        return (
          names.workspaces.find((w) => w.id === entity.scope_id)?.name ??
          "A workspace"
        )
      case "org_member":
        return "An organization member"
      case "workspace_member":
        return "A workspace member"
      case "api_token":
        return "An API key"
    }
  })()
  if (entity.model) return `${scope}, ${entity.provider_key_id}:${entity.model}`
  if (entity.provider_key_id) return `${scope}, on ${entity.provider_key_id}`
  return scope
}
