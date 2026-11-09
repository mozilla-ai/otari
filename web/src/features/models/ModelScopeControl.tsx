import { type ReactNode, useMemo } from "react"
import {
  AccessListControl,
  type AccessListCopy,
  type AccessOption,
} from "@/features/models/AccessListControl"
import { providerInstanceOf, providerModelId } from "@/features/models/modelKey"
import { canManage } from "@/features/organization/roles"
import { useModels } from "@/shared/api/models"
import {
  useDeploymentOperator,
  useOrganizationContext,
  useOrgProviderKeys,
} from "@/shared/api/organizations"
import { useDeploymentProviderNames } from "@/shared/api/providers"
import { useAliases } from "@/shared/api/routing"
import { useWorkspaceProviderPolicy } from "@/shared/api/workspaces"
import { providerDisplayName } from "@/shared/helpers/providers"

// The control is reused for two layers of the same allow-list grammar: a user's
// default and a key's (narrower) override. The wording differs between them, so
// the heading, help text, and the "any" mode label are parameterized. "any" means
// `null` on the wire, converted where it arrives: unrestricted for a user, but
// "inherit the owner's default" for a key (a key with no list of its own falls
// back to its user).
export function ModelScopeControl({
  initial,
  onChange,
  title = "Model access",
  description,
  anyLabel = "Any model",
  workspaceId,
}: {
  initial: string[] | undefined
  onChange: (value: string[] | undefined, valid: boolean) => void
  title?: ReactNode
  description?: ReactNode
  anyLabel?: string
  /**
   * Offers only what this workspace may call; unset is the caller's whole
   * reach, and `null` holds the choices until the workspace is known.
   */
  workspaceId?: string | null
}) {
  const models = useModels(workspaceId !== null, workspaceId ?? "")
  // A workspace requiring zero data retention refuses every provider its own
  // listing leaves out, so nothing else may add a wildcard or an alias for one.
  // Until its policy answers, the listing alone decides.
  const policy = useWorkspaceProviderPolicy(workspaceId || null)
  const isNarrowed =
    workspaceId === null ||
    (!!workspaceId && policy.data?.require_zero_data_retention !== false)
  const { isOperator, isSettled } = useDeploymentOperator()
  // Operator-only: an organization admin is refused it.
  const providers = useDeploymentProviderNames(isSettled && isOperator)
  // Owner/admin-only, like the operator read above.
  const context = useOrganizationContext()
  const orgKeys = useOrgProviderKeys(canManage(context.data))
  const aliases = useAliases()

  // A wildcard per provider instance, every model, and each alias resolved to
  // its target, deduped by value. GET /models, scoped to the caller's
  // organization, lists nothing for a provider it cannot read or a key nobody
  // refreshed, so the operator's configured providers and an owner's or admin's
  // usable keys each add a wildcard. Every entry is a selector the backend
  // accepts, named for a reader.
  const catalog = useMemo<AccessOption[]>(() => {
    const ids = (models.data?.data ?? [])
      .map((model) => model.id)
      .filter((id) => providerInstanceOf(id) !== undefined)
      .sort((a, b) => a.localeCompare(b))
    const listed = new Set(ids.map((id) => providerInstanceOf(id) as string))
    const reachable = (instance: string) => !isNarrowed || listed.has(instance)
    const instances = [
      ...new Set([
        ...providers,
        // A key with a slug is reached as `<slug>:<model>`, which allowed_models
        // refuses to store, so it adds no wildcard.
        ...(orgKeys.data ?? [])
          .filter((key) => !key.archived_at && key.usable && !key.slug)
          .map((key) => key.provider),
        ...listed,
      ]),
    ]
      .filter(reachable)
      .sort((a, b) => a.localeCompare(b))
    const providerOf = (id: string) =>
      providerDisplayName(providerInstanceOf(id) ?? "")
    const candidates: AccessOption[] = [
      ...instances.map((instance) => ({
        id: `${instance}:*`,
        label: `All ${providerDisplayName(instance)} models`,
        hint: `${instance}:* · includes models added later`,
      })),
      ...ids.map((id) => ({
        id,
        label: providerModelId(id),
        hint: providerOf(id),
      })),
      ...(aliases.data ?? [])
        .filter((alias) => reachable(providerInstanceOf(alias.target) ?? ""))
        .map((alias) => ({
          id: alias.target,
          label: alias.name,
          hint: `Alias for ${providerModelId(alias.target)} · ${providerOf(alias.target)}`,
        })),
    ]
    // First label wins, so a model keeps its own name over the alias that
    // resolves to it.
    const seen = new Set<string>()
    const unique = candidates.filter((option) => {
      if (!option.id || seen.has(option.id)) return false
      seen.add(option.id)
      return true
    })
    // A chip shows the label alone, so a shared label names its provider.
    const shared = new Set(
      unique
        .map((option) => option.label)
        .filter((label, index, labels) => labels.indexOf(label) !== index),
    )
    return unique.map((option) => {
      if (!shared.has(option.label) || option.id.endsWith(":*")) return option
      const provider = providerOf(option.id)
      return {
        id: option.id,
        label: `${option.label} · ${provider}`,
        hint: option.hint === provider ? undefined : option.hint,
      }
    })
  }, [models.data, providers, orgKeys.data, aliases.data, isNarrowed])

  // A failed read is said even when what did load fills the list, which is then
  // incomplete. `isLoading`: a key read a plain member never makes stays pending.
  const isReading = models.isPending || context.isPending || orgKeys.isLoading
  const catalogHint =
    models.isError || orgKeys.isError
      ? "Could not load the model catalog. Reload to try again."
      : !isReading && catalog.length === 0
        ? "No providers or models discovered yet. Add a provider first."
        : null

  const copy: AccessListCopy = {
    inherit: anyLabel,
    only: "Only selected",
    block: "Block all",
    blocked: (
      <>
        <strong>every</strong> model until you change this access.
      </>
    ),
    pickPrompt: "Pick at least one model, or choose “Block all”.",
    inputLabel: "Allowed models",
    placeholder: "Search providers, models, aliases…",
  }

  return (
    <AccessListControl
      initial={initial}
      onChange={onChange}
      title={title}
      description={
        description ??
        "Which models this key may list and call. The master key is never restricted, so blocking a key cannot lock you out of the dashboard."
      }
      copy={copy}
      catalog={catalog}
      hint={catalogHint}
      loadingMessage={isReading ? "Loading the model catalog…" : null}
    />
  )
}

// A compact label describing a key's access, for the table row. Deliberately not
// a count: an entry like `openai:*` is one entry but many models, so a number
// would mislead. The exact entries are surfaced on hover / in the edit form.
export function accessLabel(allowedModels: string[] | undefined): {
  text: string
  tone: "muted" | "normal" | "danger"
} {
  // Absent is "not restricted", which an empty list cannot say: `[]` is a real
  // answer meaning no model at all. The wire spells absent `null`, so callers
  // convert it where it arrives.
  if (!allowedModels) return { text: "All models", tone: "muted" }
  if (allowedModels.length === 0) return { text: "No models", tone: "danger" }
  return { text: "Selected models", tone: "normal" }
}
