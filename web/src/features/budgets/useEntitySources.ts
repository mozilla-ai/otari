import { useKeys } from "@/shared/api/apiKeys"
import { useModels } from "@/shared/api/models"
import { useOrganizationMembers } from "@/shared/api/organizations"
import { useWorkspaces } from "@/shared/api/workspaces"

import type { EntitySources } from "./appliedEntities"

/**
 * The reads an entity is named and offered from, and which of them failed.
 *
 * The failures are named so a page can say "could not load workspaces" rather
 * than show an empty list that reads as an organization with none.
 */
export function useEntitySources(organizationId: string): {
  sources: EntitySources
  failedLists: string[]
} {
  const workspaces = useWorkspaces()
  const members = useOrganizationMembers()
  const keys = useKeys()
  const models = useModels()
  return {
    sources: {
      organizationId,
      workspaces: workspaces.data ?? [],
      members: members.data ?? [],
      keys: keys.data ?? [],
      modelIds: (models.data?.data ?? []).map((model) => model.id),
    },
    failedLists: [
      workspaces.isError && "workspaces",
      members.isError && "members",
      keys.isError && "API keys",
      models.isError && "providers and models",
    ].filter((name): name is string => Boolean(name)),
  }
}
