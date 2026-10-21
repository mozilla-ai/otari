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
}: {
  initial: string[] | undefined
  onChange: (value: string[] | undefined, valid: boolean) => void
  title?: ReactNode
  description?: ReactNode
  anyLabel?: string
}) {
  const models = useModels()
  const { isOperator, isSettled } = useDeploymentOperator()
  // Operator-only: an organization admin is refused it.
  const providers = useDeploymentProviderNames(isSettled && isOperator)
  // Owner/admin-only, like the operator read above.
  const orgKeys = useOrgProviderKeys(canManage(useOrganizationContext().data))
  const aliases = useAliases()

  // The pickable catalog: a wildcard per provider instance, every model, and
  // each alias resolved to its target. It comes from GET /models, which is
  // scoped to the caller's organization (its own providers and the deployment's
  // managed models), not from the operator-only discovery or provider-config
  // reads, which an organization admin cannot see. Deduped by value, so an alias
  // whose target is already listed does not appear twice. Every stored entry is
  // a canonical selector the backend will accept, so names that resolve through
  // no instance are left out. An operator also gets a wildcard for every
  // configured provider, which covers one whose models GET /models cannot list
  // (discovery failed or unsupported, nothing priced). An organization owner or
  // admin likewise gets one for each of its own provider keys: a key nobody has
  // refreshed offers no models, so GET /models lists nothing for it.
  // Each option is named for a reader (the model, or "All OpenAI models") and
  // keeps the stored selector findable in its hint, since that is what a key's
  // allow-list and the gateway's refusals spell.
  const catalog = useMemo<AccessOption[]>(() => {
    const ids = (models.data?.data ?? [])
      .map((model) => model.id)
      .filter((id) => providerInstanceOf(id) !== undefined)
      .sort((a, b) => a.localeCompare(b))
    const instances = [
      ...new Set([
        ...providers,
        ...(orgKeys.data ?? [])
          .filter((key) => !key.archived_at)
          .map((key) => key.provider),
        ...ids.map((id) => providerInstanceOf(id) as string),
      ]),
    ].sort((a, b) => a.localeCompare(b))
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
      ...(aliases.data ?? []).map((alias) => ({
        id: alias.target,
        label: alias.name,
        hint: `Alias for ${providerModelId(alias.target)} · ${providerOf(alias.target)}`,
      })),
    ]
    // First label wins, so a model keeps its own name over the alias that
    // resolves to it.
    const seen = new Set<string>()
    return candidates.filter((option) => {
      if (!option.id || seen.has(option.id)) return false
      seen.add(option.id)
      return true
    })
  }, [models.data, providers, orgKeys.data, aliases.data])

  // A failed read and an empty one both leave little or nothing to pick; only
  // the second means there is nothing to configure. A failed read is said even
  // when aliases or provider wildcards still fill the list, which is then
  // incomplete.
  const catalogHint = models.isError
    ? "Could not load the model catalog. Reload to try again."
    : !models.isPending && catalog.length === 0
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
      loadingMessage={models.isPending ? "Loading the model catalog…" : null}
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
