import type { OrgProviderKey } from "@/client"
import { useModels } from "@/shared/api/models"

// One workspace's model allow-list on one of its organization's keys is per
// (workspace, key) and narrows rather than widens: no entries means every model
// that key serves. It is what `services/tenancy/organization_model_access.py`
// reads when it builds a session's catalog scope.

/** How a key is named on screen: its provider and the name the organization gave it. */
export function keyLabel(
  key: OrgProviderKey | undefined,
  keyId: string,
): string {
  // A key the organization list does not carry is not a state the gateway
  // produces (both reads cover the same non-archived set), so this is a
  // fallback rather than a case. It says what the row is before it says which
  // one: a bare UUID named nothing a person could act on, where "Unknown key"
  // plus a fragment is both readable and enough to find the row again (#2106).
  if (!key) return `Unknown key ${keyId.slice(0, 8)}`
  return `${key.provider} / ${key.name}`
}

/**
 * Why a typed model name cannot go on a key's list, or undefined when it can.
 *
 * The catalog spells a model `provider:model` and the list stores the bare
 * name, so the id somebody copies off the Models page is the one entry that
 * looks right and narrows the workspace to a model that does not exist. Matched
 * against the providers this deployment names rather than any text before a
 * colon, because `llama3:8b` is a whole model name on Ollama.
 */
export function keyModelProblem(
  name: string,
  provider: string | undefined,
  knownProviders: ReadonlySet<string>,
): string | undefined {
  const separator = name.indexOf(":")
  const prefix = separator === -1 ? undefined : name.slice(0, separator)
  if (prefix === undefined || !knownProviders.has(prefix)) return undefined
  return prefix === provider
    ? `Name the model without its "${prefix}:" prefix.`
    : `"${prefix}:" names another provider. Name a model this key serves.`
}

/** Every model the catalog lists for each provider, by bare name, and every provider name. */
export function useKeyModelSuggestions(keys: OrgProviderKey[]) {
  const catalog = useModels()
  const byProvider = (catalog.data?.data ?? []).reduce((groups, model) => {
    const separator = model.id.indexOf(":")
    if (separator === -1) return groups
    const provider = model.id.slice(0, separator)
    const name = model.id.slice(separator + 1)
    const entries = groups.get(provider)
    if (entries) entries.push(name)
    else groups.set(provider, [name])
    return groups
  }, new Map<string, string[]>())
  const knownProviders: ReadonlySet<string> = new Set([
    ...byProvider.keys(),
    ...keys.map((key) => key.provider),
  ])
  return { byProvider, knownProviders, isPending: catalog.isPending }
}
