import { Link } from "@tanstack/react-router"
import { useMemo, useRef, useState } from "react"

import type {
  AliasResponse,
  PolicyGuardrail,
  PolicySpec,
  RoutingPolicyResponse,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { CopyButton } from "@/design-system/actions/CopyButton"
import { CopyableValue } from "@/design-system/actions/CopyField"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { FieldAction } from "@/design-system/forms/FieldAction"
import { ControlField } from "@/design-system/forms/FieldMessages"
import { Dot } from "@/design-system/indicators/Dot"
import { ListDetail, ListDetailRow } from "@/design-system/layout/ListDetail"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import { ModelComboBox } from "@/features/models/ModelComboBox"
import { canManage, isDeploymentOperator } from "@/features/organization/roles"
import { RouterReadiness } from "@/features/routing/RouterReadiness"
import { UserComboBox } from "@/features/users/UserComboBox"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useAliases,
  useCreateAlias,
  useCreateOrganizationAlias,
  useDeleteAlias,
  useDeleteOrganizationAlias,
  useDeleteOrganizationRoutingPolicy,
  useDeleteRoutingPolicy,
  useOrganizationAliases,
  useOrganizationRoutingPolicies,
  useRoutingPolicies,
  useSetOrganizationRoutingPolicy,
  useSetRoutingPolicy,
} from "@/shared/api/routing"
import { useToolSettings } from "@/shared/api/tools"
import { useUsers } from "@/shared/api/users"
import { useUrlValue } from "@/shared/helpers/urlState"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

/** A row on this page: either a routing policy or a stored/config alias.
 *
 *  An alias is the one-target case of a policy, so the two are listed together
 *  and this page is the single place either is managed. They still live in
 *  different tables behind different endpoints, so `kind` decides which API a
 *  write goes to; it is not cosmetic.
 */
type RoutingRow = RoutingPolicyResponse & { kind: "policy" | "alias" }

/** The router backends the form can write. Any other is shown read-only rather
 *  than rewritten as one of these on save. */
const KNN_BACKEND = "knn"
const WEIGHTED_BACKEND = "weighted"
type RouterBackend = typeof KNN_BACKEND | typeof WEIGHTED_BACKEND

/** Server-side cap on a compiled plan (`MAX_CANDIDATES` in models/routing.py). */
const MAX_CANDIDATES = 5

/** Present an alias as the one-target policy it is. */
function aliasAsRow(alias: AliasResponse): RoutingRow {
  return {
    kind: "alias",
    name: alias.name,
    spec: { select: [{ default: alias.target }] },
    source: alias.source,
    user_id: alias.user_id,
    // Carried, not dropped: an edit or a delete goes back to the workspace the
    // row lives in, and a row without it falls back to the *selected* workspace,
    // which for an admin whose list spans the organization is a different one.
    workspace_id: alias.workspace_id,
    is_dynamic: false,
    created_at: alias.created_at,
    updated_at: alias.updated_at,
  }
}

// Scope is part of the identity, so it is part of the row key: the same policy
// name can exist globally and per user, and keying on the name alone would
// collapse those rows into one. Same reasoning (and encoding) as the alias table.
const rowKeyOf = (row: RoutingRow): string =>
  JSON.stringify([row.kind, row.user_id, row.name])

/** Whether a guardrails service is configured for this gateway.
 *
 *  A policy guardrail is a request to a separate service (`guardrails_url`). With
 *  no service configured there is nothing to call, so mandating a check would
 *  either fail every request through the policy (mode block, on_unavailable block)
 *  or silently do nothing. Neither is a state to let an operator build by accident,
 *  so the affordance is disabled until a service exists.
 *
 *  While the settings are still loading this returns `true`: a control that starts
 *  enabled and stays enabled is better than one that flickers from disabled to
 *  enabled, which reads as a bug.
 */
function useGuardrailsConfigured(): {
  configured: boolean
  isLoading: boolean
} {
  const settings = useToolSettings()
  const field = settings.data?.fields.find(
    (entry) => entry.key === "guardrails_url",
  )
  const value = typeof field?.value === "string" ? field.value.trim() : ""
  return {
    configured: settings.isLoading || value !== "",
    isLoading: settings.isLoading,
  }
}

/** Whether this form can represent a spec without losing part of it.
 *
 *  The editor reconstructs a spec from four pieces of state, so anything it does
 *  not model (a `user_id`/`key_id` condition, a comparator other than `gte`, a
 *  `budget_remaining_usd` threshold, a router entry) would be silently dropped on
 *  save. Offering Edit on such a policy would quietly destroy the operator's
 *  config, so those are shown read-only until the form covers them. Refusing to
 *  edit is recoverable; a silent lossy save is not.
 */
function isEditableInForm(spec: PolicySpec): boolean {
  // The form re-emits `select` as conditions, then the router, then the default.
  // Selection is order-sensitive server-side (the first matching entry wins), so a
  // spec whose router sits *before* its conditions would come back with different
  // behavior than it went in with. Refusing to edit is recoverable; a silent
  // semantic change on Save is not.
  const routerIndex = spec.select.findIndex(
    (entry) => entry.router !== undefined,
  )
  const lastConditionIndex = spec.select.reduce(
    (last, entry, index) => (entry.when !== undefined ? index : last),
    -1,
  )
  if (
    routerIndex !== -1 &&
    lastConditionIndex !== -1 &&
    routerIndex < lastConditionIndex
  )
    return false
  return spec.select.every((entry) => {
    if (entry.default !== undefined) return entry.when === undefined
    // A router entry is editable: the form models the backend, its pool and (for
    // the weighted backend) the split, which is the whole entry. An unknown backend
    // is still shown read-only, because saving it back through one of these
    // controls would silently rewrite it as a backend the operator did not choose.
    if (entry.router !== undefined) {
      if ((entry.candidates?.length ?? 0) === 0) return false
      const backend = normalizedBackend(entry.router)
      if (backend === KNN_BACKEND) return true
      // A weighted entry without weights cannot be saved back (the API refuses it),
      // so the form would have to invent a split. Read-only says so instead.
      return (
        backend === WEIGHTED_BACKEND &&
        Object.keys(entry.weights ?? {}).length > 0
      )
    }
    const when = entry.when
    if (when === undefined || entry.target === undefined) return false
    const keys = Object.keys(when)
    return (
      keys.length === 1 &&
      keys[0] === "budget_used_pct" &&
      when.budget_used_pct?.gte !== undefined
    )
  })
}

/** The fallthrough target of a spec, which every valid spec has exactly one of. */
function defaultTargetOf(spec: PolicySpec): string {
  return spec.select.find((entry) => entry.default !== undefined)?.default ?? ""
}

/** The router's candidate pool, or an empty list for a policy with no router. */
function candidatesOf(spec: PolicySpec): string[] {
  return (
    spec.select.find((entry) => entry.router !== undefined)?.candidates ?? []
  )
}

/** The pool the form edits: the router's candidates, with the default target in it.
 *
 *  The gateway appends the default target to the pool when a policy omits it, so a
 *  spec written through the API can list it or not. Normalizing here means the form
 *  shows the models that will actually be dispatched, in the order they were
 *  written, rather than a pool that is missing its own fallback.
 */
function initialPool(spec: PolicySpec): string[] {
  const candidates = candidatesOf(spec)
  if (candidates.length === 0) return []
  const fallthrough = defaultTargetOf(spec)
  return candidates.includes(fallthrough)
    ? candidates
    : [...candidates, fallthrough]
}

/** Which entry of `initialPool` serves when the router declines. */
function initialSafeIndex(spec: PolicySpec): number {
  const index = initialPool(spec).indexOf(defaultTargetOf(spec))
  return index === -1 ? 0 : index
}

/** A backend name as the server reads it.
 *
 *  The resolver matches on `name.strip().lower()`, so `" KNN "` selects the learned
 *  router. Comparing the raw string here would show a policy the gateway routes
 *  perfectly well as an unrecognized backend, read-only and mislabelled.
 */
function normalizedBackend(name: string | undefined): string | undefined {
  return name?.trim().toLowerCase()
}

/** The router backend a policy names, or undefined for a policy with no router. */
function routerBackendOf(spec: PolicySpec): string | undefined {
  return normalizedBackend(
    spec.select.find((entry) => entry.router !== undefined)?.router,
  )
}

/** What to call the backend that decides, for a chip or a one-line summary.
 *
 *  Named per backend rather than "Dynamic", because the backend's name is what tells
 *  the reader what to do next (teach it, or move the shares). A backend this build
 *  does not know gets the neutral word: it is routed, and claiming it learns would be
 *  a guess about a backend added after this line was written.
 */
function routerLabelOf(spec: PolicySpec): string {
  const backend = routerBackendOf(spec)
  if (backend === WEIGHTED_BACKEND) return "Weighted"
  if (backend === KNN_BACKEND) return "Learned"
  return "Routed"
}

/** The declared traffic split, empty unless the policy is weighted. */
function weightsOf(spec: PolicySpec): Record<string, number> {
  return spec.select.find((entry) => entry.router !== undefined)?.weights ?? {}
}

/** Each candidate's percentage of the traffic, normalized like the server does.
 *
 *  Weights are relative, so the form shows what the operator actually gets: 7 and 3
 *  read as 70% and 30%. A candidate with no weight takes none of the traffic and
 *  stays in the plan as a failover target, which is how a provider is drained.
 */
function sharesOf(weights: number[]): number[] {
  const total = weights.reduce((sum, weight) => sum + Math.max(0, weight), 0)
  if (total <= 0) return weights.map(() => 0)
  return weights.map((weight) => (Math.max(0, weight) * 100) / total)
}

/** The conditional entries, i.e. everything that is not the fallthrough. */
function conditionsOf(
  spec: PolicySpec,
): { threshold: number; target: string }[] {
  return spec.select
    .filter(
      (entry) =>
        entry.when?.budget_used_pct?.gte !== undefined &&
        entry.target !== undefined,
    )
    .map((entry) => ({
      threshold: entry.when!.budget_used_pct!.gte!,
      target: entry.target!,
    }))
}

/** One line summarizing what a policy serves, for a row and the column it opens. */
function servesSummary(policy: RoutingPolicyResponse): string {
  const chain = policy.spec.on_failure ?? []
  const pool = candidatesOf(policy.spec)
  if (pool.length > 0 && routerBackendOf(policy.spec) === WEIGHTED_BACKEND) {
    // The split shape, not the model names: two provider:model strings do not fit
    // one line of a row, and the shares are what distinguishes one weighted
    // policy from another. The pool is spelled out in the editor and in explain.
    const declared = weightsOf(policy.spec)
    const target = defaultTargetOf(policy.spec)
    const full = pool.includes(target) ? pool : [...pool, target]
    const split = sharesOf(full.map((selector) => declared[selector] ?? 0))
      .map((share) => `${Math.round(share)}%`)
      .join(" / ")
    return `Weighted · ${split} across ${full.length} models`
  }
  if (pool.length > 0) {
    return `${routerLabelOf(policy.spec)} · ${pool.length} candidates, ${defaultTargetOf(policy.spec)} by default`
  }
  if (policy.is_dynamic) {
    const total = 1 + chain.length
    return `Chosen per request · ${total} candidate${total === 1 ? "" : "s"}`
  }
  const target = defaultTargetOf(policy.spec)
  return chain.length > 0 ? `${target}  +${chain.length} on failure` : target
}

// ---------------------------------------------------------------------------
// Editor
// ---------------------------------------------------------------------------

/** Who a policy applies to. Same control and wording as the alias scope picker,
 *  because it is the same decision. */
function ScopePicker({
  userId,
  onChange,
}: {
  userId: string | null
  onChange: (userId: string | null) => void
}) {
  const users = useUsers()
  const scoped = userId !== null

  const modeButton = (value: boolean, label: string) => (
    <Tab
      key={label}
      isActive={scoped === value}
      onPress={() => onChange(value ? "" : null)}
    >
      {label}
    </Tab>
  )

  return (
    <div className="flex flex-col gap-3">
      <ControlField
        label="Applies to"
        description="A global policy resolves for every caller. A user-scoped one resolves only for that user, and takes precedence over a global policy of the same name."
      />
      <TabRow>
        {modeButton(false, "Every caller")}
        {modeButton(true, "One user")}
      </TabRow>
      {scoped ? (
        <UserComboBox
          label="User"
          value={userId ?? ""}
          onChange={onChange}
          users={users.data ?? []}
          placeholder="Pick a user…"
          description="Only this user resolves the policy."
          unknownHint={
            <span className="text-danger">
              No such user. Pick an existing one.
            </span>
          }
        />
      ) : null}
    </div>
  )
}

const MODE_VALUES = ["block", "monitor"] as const

/** A two-value mode switch. The codebase has no Select component and four
 *  hand-rolled `aria-pressed` groups, so this follows that pattern rather than
 *  introducing a fifth idiom. */
function ModeToggle({
  label,
  hint,
  value,
  onChange,
}: {
  label: string
  hint?: string
  value: "block" | "monitor"
  onChange: (value: "block" | "monitor") => void
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-body">{label}</span>
      <TabRow>
        {MODE_VALUES.map((mode) => (
          <Tab
            key={mode}
            isActive={value === mode}
            onPress={() => onChange(mode)}
          >
            {mode}
          </Tab>
        ))}
      </TabRow>
      {hint === undefined ? null : <span className="text-caption">{hint}</span>}
    </div>
  )
}

/** Create or edit a policy.
 *
 *  Reading order mirrors the schema so the form and the YAML teach the same
 *  model: name and scope, then what serves a normal request, then what happens on
 *  failure, then what always runs. The failure and guardrail sections are absent
 *  until summoned rather than collapsed-and-empty, which keeps naming one model a
 *  three-field task.
 */
export function PolicyForm({
  existing,
  initialTarget = "",
  workspaceId,
  onClose,
}: {
  existing: RoutingRow | null
  initialTarget?: string
  /**
   * The workspace a tenant admin's write lands in, or null for an operator.
   *
   * Null is the deployment-wide surface, which defaults the workspace itself;
   * a string is the tenant-scoped one, which requires it named and refuses a
   * user scope. Both mutation pairs are always created, as hooks must be, and
   * only the pair this says is ever mutated.
   */
  workspaceId: string | null
  onClose: () => void
}) {
  const save = useSetRoutingPolicy()
  const saveAlias = useCreateAlias()
  const saveOrgPolicy = useSetOrganizationRoutingPolicy()
  const saveOrgAlias = useCreateOrganizationAlias()
  const tenantScoped = workspaceId !== null
  const editing = existing !== null
  // Editing an alias writes back through the alias API: it is still a row in
  // model_aliases, and silently rewriting it as a policy would leave the original
  // behind under the same name.
  const editingAlias = existing?.kind === "alias"
  const guardrails_ = useGuardrailsConfigured()

  const [name, setName] = useState(existing?.name ?? "")
  const [userId, setUserId] = useState<string | null>(existing?.user_id ?? null)
  const [target, setTarget] = useState(
    existing ? defaultTargetOf(existing.spec) : initialTarget,
  )
  const [chain, setChain] = useState<string[]>(existing?.spec.on_failure ?? [])
  const [conditions, setConditions] = useState(
    existing ? conditionsOf(existing.spec) : [],
  )
  const [guardrails, setGuardrails] = useState<PolicyGuardrail[]>(
    existing?.spec.guardrails ?? [],
  )
  // The learned router's pool, and which of its models serves when the router
  // declines. One list rather than a pool plus a separate "Serves" field: the
  // fallback is always one of the models the router may choose, so asking for it
  // twice made an operator name the strong model in two places and invited them to
  // disagree with themselves. This mirrors what the gateway does with the spec,
  // where the default target joins the pool if it was left out.
  const [candidates, setCandidates] = useState<string[]>(
    existing ? initialPool(existing.spec) : [],
  )
  const [safeIndex, setSafeIndex] = useState<number>(
    existing ? initialSafeIndex(existing.spec) : 0,
  )
  // Which backend orders the pool. The two share the pool control, because both are
  // "these models, one of them per request"; they differ in what decides and in
  // whether a share sits next to each entry.
  const [backend, setBackend] = useState<RouterBackend>(
    existing && routerBackendOf(existing.spec) === WEIGHTED_BACKEND
      ? WEIGHTED_BACKEND
      : KNN_BACKEND,
  )
  // Parallel to `candidates`, so a weight follows its model when one is removed.
  // Held as the text the operator typed rather than as a number: re-rendering a
  // parsed number swallows a half-typed decimal ("7." parses to 7 and renders back
  // as "7") and turns a cleared field into a silent 0. Parsed once, below.
  const [weights, setWeights] = useState<string[]>(() => {
    if (existing === null) return []
    const declared = weightsOf(existing.spec)
    return initialPool(existing.spec).map((selector) =>
      String(declared[selector] ?? 0),
    )
  })
  const routed = candidates.length > 0
  const weighted = routed && backend === WEIGHTED_BACKEND
  // An empty field parses to NaN rather than 0, so a share the operator cleared is
  // unfinished rather than a drain they did not ask for. "Infinity" and a negative
  // are rejected here too, matching what the API refuses.
  const weightValues = weights.map((text) =>
    text.trim() === "" ? Number.NaN : Number(text),
  )
  const weightsWellFormed = weightValues.every(
    (value) => Number.isFinite(value) && value >= 0,
  )
  const shares = sharesOf(
    weightValues.map((value) =>
      Number.isFinite(value) ? Math.max(0, value) : 0,
    ),
  )
  // With a router, the fallthrough is the marked model; without one it is the single
  // "Serves" field.
  const effectiveTarget = routed ? (candidates[safeIndex] ?? "") : target

  const nameHasDelimiter = /[:/]/.test(name)
  // A policy's name is its key, so a rename is a move rather than an edit: the API
  // takes it as `rename_from` on the same write as the spec. Aliases have no such
  // verb, so their name stays fixed here.
  const previousName = existing?.name ?? ""
  const renaming =
    editing &&
    !editingAlias &&
    name.trim() !== "" &&
    name.trim() !== previousName
  const scopeReady = userId === null || userId.trim() !== ""
  const conditionsReady = conditions.every(
    (c) => c.target.trim() !== "" && c.threshold > 0 && c.threshold < 100,
  )
  const guardrailsReady = guardrails.every((g) => g.profile.trim() !== "")
  // A model named twice is refused by the API, and on a weighted policy it would
  // also collapse in the weight map: two rows, one key, so the split submitted is
  // not the split the form showed. Checked over the named rows only, so a pair of
  // still-empty rows reads as unfinished rather than as a duplicate.
  const namedCandidates = candidates
    .map((entry) => entry.trim())
    .filter((entry) => entry !== "")
  const duplicateCandidate =
    new Set(namedCandidates).size !== namedCandidates.length
  // Two, not one: ranking a single model is not a decision, and the API refuses it.
  const candidatesReady =
    !routed ||
    (candidates.length >= 2 &&
      candidates.every((entry) => entry.trim() !== "") &&
      !duplicateCandidate &&
      effectiveTarget.trim() !== "")
  // An all-zero split would select nothing and the policy would always serve its
  // default, so the API refuses it. Caught here so the form cannot author it.
  const splitReady =
    !weighted || (weightsWellFormed && weightValues.some((value) => value > 0))
  // The server caps the compiled plan at MAX_CANDIDATES, counting the routed pool
  // plus the failure chain. Enforced here too so the form cannot author a policy it
  // then fails to save: a rule the UI knows about should not arrive as a 400.
  const plannedCandidates = (candidates.length || 1) + chain.length
  const atCandidateCap = plannedCandidates >= MAX_CANDIDATES
  const overCandidateCap = plannedCandidates > MAX_CANDIDATES
  const canSubmit =
    name.trim() !== "" &&
    effectiveTarget.trim() !== "" &&
    !nameHasDelimiter &&
    scopeReady &&
    conditionsReady &&
    guardrailsReady &&
    candidatesReady &&
    splitReady &&
    !overCandidateCap &&
    chain.every((entry) => entry.trim() !== "")

  // Built in plan order, with the fallthrough last, which is what the schema
  // requires: an entry after the default could never be reached.
  const spec: PolicySpec = useMemo(
    () => ({
      select: [
        ...conditions.map((condition) => ({
          when: { budget_used_pct: { gte: condition.threshold } },
          target: condition.target.trim(),
        })),
        // After the conditions, before the fallthrough: an explicit tier-down is
        // the operator overriding the router, and the router is what runs when no
        // condition applies.
        ...(routed
          ? [
              {
                router: backend,
                candidates: candidates.map((entry) => entry.trim()),
                // Keyed by selector, which is how the server reads it. Only for the
                // weighted backend: a weight map on a knn entry is refused, because
                // it would read as a split and do nothing.
                ...(weighted
                  ? {
                      weights: Object.fromEntries(
                        candidates.map((entry, index) => {
                          const value = weightValues[index] ?? 0
                          return [
                            entry.trim(),
                            Number.isFinite(value) ? Math.max(0, value) : 0,
                          ]
                        }),
                      ),
                    }
                  : {}),
              },
            ]
          : []),
        { default: effectiveTarget.trim() },
      ],
      ...(chain.length > 0
        ? { on_failure: chain.map((entry) => entry.trim()) }
        : {}),
      ...(guardrails.length > 0 ? { guardrails } : {}),
    }),
    [
      conditions,
      candidates,
      routed,
      backend,
      weighted,
      weightValues,
      effectiveTarget,
      chain,
      guardrails,
    ],
  )

  // Everything the operator can change, against what it was seeded with. A guard
  // that watched only the name would be worse than none on a form this long,
  // where a stray Escape can land ten minutes into building a fallback chain.
  const draft = JSON.stringify({
    name,
    userId,
    target,
    chain,
    conditions,
    guardrails,
    candidates,
    safeIndex,
    backend,
    weights,
  })
  const seeded = useRef(draft)
  const isDirty = draft !== seeded.current

  // An alias has exactly one target, so growing one a chain, a condition, or a
  // guardrail makes it a policy. Saving it as a policy alone would leave the alias
  // row in place under the same name, and the API refuses that collision, so the
  // form keeps an alias an alias and points the operator at the way across.
  const outgrewAlias =
    editingAlias &&
    (chain.length > 0 ||
      conditions.length > 0 ||
      guardrails.length > 0 ||
      candidates.length > 0)
  const pending =
    save.isPending ||
    saveAlias.isPending ||
    saveOrgPolicy.isPending ||
    saveOrgAlias.isPending

  const submit = () => {
    if (!canSubmit || outgrewAlias) return
    const scope = userId === null ? null : userId.trim()
    if (editingAlias) {
      if (workspaceId !== null) {
        saveOrgAlias.mutate(
          {
            name: name.trim(),
            target: effectiveTarget.trim(),
            workspace_id: workspaceId,
          },
          { onSuccess: onClose },
        )
        return
      }
      saveAlias.mutate(
        { name: name.trim(), target: effectiveTarget.trim(), user_id: scope },
        { onSuccess: onClose },
      )
      return
    }
    if (workspaceId !== null) {
      saveOrgPolicy.mutate(
        {
          name: name.trim(),
          spec,
          workspace_id: workspaceId,
          ...(renaming ? { rename_from: previousName } : {}),
        },
        { onSuccess: onClose },
      )
      return
    }
    save.mutate(
      {
        name: name.trim(),
        spec,
        user_id: scope,
        ...(renaming ? { rename_from: previousName } : {}),
      },
      { onSuccess: onClose },
    )
  }

  return (
    <FormDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      // `lg` on every routing dialog: this form grows a fallback chain, a
      // condition tier, a weighted split and a guardrail list as they are asked
      // for, and a frame that changed width while an operator built one up
      // would read as a different dialog each time.
      size="lg"
      title={
        editing
          ? `Edit ${existing.kind === "alias" ? "alias" : "policy"}`
          : "New policy"
      }
      // The policy's identity, in the mono face that says it is a value rather
      // than prose. Here rather than in the title because `title` is a string.
      description={
        editing ? (
          <>
            <code>{existing.name}</code>
            {existing.user_id ? (
              <>
                {" "}
                for user <code>{existing.user_id}</code>
              </>
            ) : null}
          </>
        ) : (
          <>
            What callers send as <code>model</code>, and the model that answers
            it.
          </>
        )
      }
      submitLabel={editing ? "Save" : "Create policy"}
      onSubmit={submit}
      isPending={pending}
      isSubmitDisabled={!canSubmit || outgrewAlias}
      isDirty={isDirty}
      // All four writers, not two: an organization-scoped save fails through its
      // own mutation, and with those two missing the refusal was swallowed and
      // the panel just sat there.
      error={
        save.error ??
        saveAlias.error ??
        saveOrgPolicy.error ??
        saveOrgAlias.error
      }
      footerStart={
        <p className="text-caption">In effect for new requests within 30s.</p>
      }
    >
      <div className="grid gap-4 sm:grid-cols-2">
        {editingAlias ? (
          <div className="flex flex-col gap-1">
            <span className="text-body">Alias name</span>
            <code className="text-sm text-muted">{previousName}</code>
            <span className="text-xs text-muted">
              An alias name is its key and cannot be changed here. Delete and
              recreate to change it.
            </span>
          </div>
        ) : (
          <Field
            label="Policy name"
            value={name}
            onChange={setName}
            placeholder="fast"
            isRequired
            // Only on create. Dropping an operator who clicked Edit to change a
            // target into the name box invites a typo in the one field that is
            // the policy's identity.
            autoFocus={!editing}
            description={
              nameHasDelimiter ? (
                <span className="text-danger">
                  A policy name cannot contain “:” or “/”.
                </span>
              ) : renaming ? (
                <span>
                  Renames <code>{previousName}</code> on save. Callers have to
                  send the new name from then on, and usage already recorded
                  keeps the old one.
                </span>
              ) : editing ? (
                <>
                  What callers send as <code>model</code>. Change it to rename
                  the policy.
                </>
              ) : (
                <>
                  What callers send as <code>model</code>.
                </>
              )
            }
          />
        )}
        {routed ? (
          <div className="flex flex-col gap-1">
            <span className="text-body">Serves</span>
            <span className="text-sm text-foreground">
              {effectiveTarget.trim() === "" ? (
                <span className="text-muted">
                  whichever model you mark below
                </span>
              ) : (
                <code>{effectiveTarget}</code>
              )}
            </span>
            <span className="text-xs text-muted">
              {weighted
                ? "The split picks per request, so this policy has no single target. The model marked below is what serves a caller who opts out."
                : "A router picks per request, so this policy has no single target. The model marked below is what serves when the router does not choose."}
            </span>
          </div>
        ) : (
          <ModelComboBox
            label="Serves"
            value={target}
            onChange={setTarget}
            isRequired
            description="The model that serves a normal request. Callers never see it."
          />
        )}
      </div>

      {editing ? (
        <p className="text-caption">
          Who this applies to is the other half of the key. It cannot be changed
          here: delete and recreate to move it between scopes.
        </p>
      ) : tenantScoped ? (
        // Withheld rather than disabled: a user id is a deployment-wide
        // identifier, so the tenant-scoped writer refuses one outright and
        // an organization's entries are workspace-wide. Offering the picker
        // here would take a value the API is going to reject.
        <p className="text-caption">
          This applies to everyone in the selected workspace.
        </p>
      ) : (
        <ScopePicker userId={userId} onChange={setUserId} />
      )}

      {/* Conditional tier-down */}
      {conditions.length > 0 ? (
        <div className="flex flex-col gap-3 border border-control-border p-3">
          <ControlField
            label="Instead, when the budget fills up"
            description="Checked before the model above. A threshold must be under 100: the budget gate refuses a request before selection once the cap is reached, so a rule at 100 could never fire."
          />
          {conditions.map((condition, index) => (
            <div key={index} className="flex flex-wrap items-end gap-3">
              <Field
                label="Budget used at least (%)"
                value={String(condition.threshold)}
                onChange={(value) =>
                  setConditions((prev) =>
                    prev.map((c, i) =>
                      i === index ? { ...c, threshold: Number(value) || 0 } : c,
                    ),
                  )
                }
                description={
                  condition.threshold >= 100 ? (
                    <span className="text-danger">Must be under 100.</span>
                  ) : undefined
                }
              />
              <div className="min-w-56 flex-1">
                <ModelComboBox
                  label="Use instead"
                  value={condition.target}
                  onChange={(value) =>
                    setConditions((prev) =>
                      prev.map((c, i) =>
                        i === index ? { ...c, target: value } : c,
                      ),
                    )
                  }
                  isRequired
                />
              </div>
              <FieldAction>
                <Button
                  variant="ghost"
                  onPress={() =>
                    setConditions((prev) => prev.filter((_, i) => i !== index))
                  }
                >
                  Remove
                </Button>
              </FieldAction>
            </div>
          ))}
        </div>
      ) : null}

      {/* The routed pool: one control for both backends, because both are "these
              models, one of them per request". What differs is who decides, and
              whether a share sits next to each entry. */}
      {candidates.length > 0 ? (
        <div className="flex flex-col gap-3 border border-control-border p-3">
          <ControlField
            label={
              weighted ? "Split traffic between" : "The router chooses between"
            }
            description={
              weighted
                ? "Each request goes to one of these, drawn in proportion to its share. Shares are relative, so 70 and 30 mean the same as 7 and 3. No pricing needed."
                : "For each request, the cheapest of these that past scoring says is good enough. Every model here needs pricing, because the router weighs quality against cost."
            }
          />
          {candidates.map((entry, index) => (
            <div key={index} className="flex flex-wrap items-end gap-3">
              <div className="min-w-56 flex-1">
                <ModelComboBox
                  label={`Model ${index + 1}`}
                  value={entry}
                  onChange={(value) =>
                    setCandidates((prev) =>
                      prev.map((c, i) => (i === index ? value : c)),
                    )
                  }
                  isRequired
                />
              </div>
              {weighted ? (
                <div className="flex items-end gap-2">
                  <Field
                    label="Share"
                    value={weights[index] ?? ""}
                    onChange={(value) =>
                      setWeights((prev) =>
                        prev.map((weight, i) => (i === index ? value : weight)),
                      )
                    }
                    // The percentage, not the number they typed: relative weights
                    // are easy to write and hard to read, and this is the line
                    // that says a zero-weight model is drained rather than gone.
                    description={
                      !Number.isFinite(weightValues[index] ?? Number.NaN) ||
                      (weightValues[index] ?? 0) < 0
                        ? "A number, zero or more"
                        : (weightValues[index] ?? 0) > 0
                          ? `${Math.round(shares[index] ?? 0)}% of requests`
                          : "No weighted traffic; still tried if another fails"
                    }
                  />
                </div>
              ) : null}
              <FieldAction>
                <label className="flex items-center gap-2 text-body">
                  <input
                    type="radio"
                    name="router-safe-choice"
                    checked={safeIndex === index}
                    onChange={() => setSafeIndex(index)}
                  />
                  {weighted ? "Serves on opt-out" : "Serves when unsure"}
                </label>
              </FieldAction>
              <FieldAction>
                <Button
                  variant="ghost"
                  onPress={() => {
                    setCandidates((prev) => prev.filter((_, i) => i !== index))
                    setWeights((prev) => prev.filter((_, i) => i !== index))
                    // Keep the mark on the same model where possible; if the marked
                    // one went, fall back to the first, never to nothing.
                    setSafeIndex((prev) =>
                      index < prev ? prev - 1 : index === prev ? 0 : prev,
                    )
                  }}
                >
                  Remove
                </Button>
              </FieldAction>
            </div>
          ))}
          <p className="text-caption">
            {weighted ? (
              <>
                The marked model serves a caller who sends{" "}
                <code>Otari-Router: off</code>, which is the way to pin traffic
                to one provider during an incident. A model that fails before
                responding moves the request to another model in this pool, by
                the same shares, before any fallback below.
              </>
            ) : (
              <>
                The marked model serves whenever the router does not choose: too
                few scored examples, a weakly supported pick, a request carrying
                tools, or a caller sending <code>Otari-Router: off</code>. Mark
                the one you would have picked without a router.
              </>
            )}
          </p>
          {candidates.length < 2 ? (
            <p className="text-caption text-danger">
              Name at least two models.{" "}
              {weighted ? "Splitting traffic one way" : "Ranking one"} is not a
              routing decision.
            </p>
          ) : null}
          {duplicateCandidate ? (
            <p className="text-caption text-danger">
              Name each model once.{" "}
              {weighted
                ? "A model listed twice has one share, not two, so the split saved would not be the one shown."
                : "A pool that repeats a model is refused."}
            </p>
          ) : null}
          {weighted && !weightsWellFormed ? (
            <p className="text-caption text-danger">
              Every share is a number of zero or more. Use 0 to drain a model
              without removing it.
            </p>
          ) : weighted && !splitReady ? (
            <p className="text-caption text-danger">
              Give at least one model a share above zero, or this policy can
              never send traffic anywhere but its marked model.
            </p>
          ) : null}
          <div className="flex flex-wrap items-baseline gap-2">
            <button
              type="button"
              disabled={atCandidateCap}
              className={
                atCandidateCap
                  ? "cursor-not-allowed text-sm text-muted opacity-60"
                  : "text-sm text-link hover:underline"
              }
              onClick={() => {
                setCandidates((prev) => [...prev, ""])
                // Zero, not an invented share: adding a provider must not move
                // traffic onto it before the operator says how much.
                setWeights((prev) => [...prev, "0"])
              }}
            >
              + Another model
            </button>
            {atCandidateCap ? (
              <span className="text-xs text-muted">
                A policy dispatches at most {MAX_CANDIDATES} models, counting
                the fallback chain. Remove a fallback to add another.
              </span>
            ) : null}
          </div>
        </div>
      ) : null}

      {/* Failure chain */}
      {chain.length > 0 ? (
        <div className="flex flex-col gap-3 border border-control-border p-3">
          <ControlField
            label="If that fails, try"
            description="Tried in order after a retryable failure. Not tried once tokens have started streaming, or after a 400/401/403, which every provider would reject the same way."
          />
          {chain.map((entry, index) => (
            <div key={index} className="flex flex-wrap items-end gap-3">
              <div className="min-w-56 flex-1">
                <ModelComboBox
                  label={`Fallback ${index + 1}`}
                  value={entry}
                  onChange={(value) =>
                    setChain((prev) =>
                      prev.map((e, i) => (i === index ? value : e)),
                    )
                  }
                  isRequired
                />
              </div>
              <FieldAction>
                <Button
                  variant="ghost"
                  onPress={() =>
                    setChain((prev) => prev.filter((_, i) => i !== index))
                  }
                >
                  Remove
                </Button>
              </FieldAction>
            </div>
          ))}
          <div className="flex flex-wrap items-baseline gap-2">
            <button
              type="button"
              disabled={atCandidateCap}
              className={
                atCandidateCap
                  ? "cursor-not-allowed text-sm text-muted opacity-60"
                  : "text-sm text-link hover:underline"
              }
              onClick={() => setChain((prev) => [...prev, ""])}
            >
              + Another fallback
            </button>
            {atCandidateCap ? (
              <span className="text-xs text-muted">
                A policy dispatches at most {MAX_CANDIDATES} models in total.
              </span>
            ) : null}
          </div>
        </div>
      ) : null}

      {/* Guardrails */}
      {guardrails.length > 0 ? (
        <div className="flex flex-col gap-3 border border-control-border p-3">
          <div>
            <span className="text-body">Always check</span>
            <p className="text-caption">
              Runs on every request through this policy. Callers can add their
              own guardrails but cannot weaken these.
            </p>
            {guardrails_.configured ? null : (
              <p className="mt-1 text-caption text-warning">
                No guardrails service is configured, so these cannot run. With
                `if the service is down` set to block, every request through
                this policy is refused until one is configured.{" "}
                <Link to="/tools" className="underline">
                  Set one up
                </Link>
                , or remove the guardrail.
              </p>
            )}
          </div>
          {guardrails.map((guardrail, index) => (
            <div key={index} className="flex flex-col gap-3">
              <div className="flex flex-wrap items-end gap-3">
                <Field
                  label="Profile"
                  value={guardrail.profile}
                  onChange={(value) =>
                    setGuardrails((prev) =>
                      prev.map((g, i) =>
                        i === index ? { ...g, profile: value } : g,
                      ),
                    )
                  }
                  placeholder="prompt-injection"
                  isRequired
                  description="A profile configured on the guardrails service."
                />
                <ModeToggle
                  label="Mode"
                  value={guardrail.mode}
                  onChange={(mode) =>
                    setGuardrails((prev) =>
                      prev.map((g, i) => (i === index ? { ...g, mode } : g)),
                    )
                  }
                  hint="block rejects a flagged request; monitor records it and serves anyway."
                />
                <ModeToggle
                  label="If the service is down"
                  value={guardrail.on_unavailable ?? "block"}
                  onChange={(mode) =>
                    setGuardrails((prev) =>
                      prev.map((g, i) =>
                        i === index ? { ...g, on_unavailable: mode } : g,
                      ),
                    )
                  }
                  hint="block fails closed, so a guardrails outage refuses every request through this policy."
                />
                <FieldAction>
                  <Button
                    variant="ghost"
                    onPress={() =>
                      setGuardrails((prev) =>
                        prev.filter((_, i) => i !== index),
                      )
                    }
                  >
                    Remove
                  </Button>
                </FieldAction>
              </div>
              {guardrail.mode === "block" &&
              (guardrail.on_unavailable ?? "block") === "block" ? (
                <div className="text-caption text-warning">
                  With both set to block, a guardrails-service outage rejects
                  every request through this policy, ahead of any fallback
                  above.
                </div>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}

      {/* Complexity is summoned, never presented: naming one model stays a
              three-field task. */}
      <div className="flex flex-wrap gap-3 text-sm">
        {conditions.length === 0 ? (
          <button
            type="button"
            className="text-link hover:underline"
            onClick={() => setConditions([{ threshold: 80, target: "" }])}
          >
            + Tier down when the budget fills up
          </button>
        ) : null}
        {chain.length === 0 ? (
          <button
            type="button"
            className="text-link hover:underline"
            onClick={() => setChain([""])}
          >
            + Add a fallback chain
          </button>
        ) : null}
        {candidates.length === 0 ? (
          <button
            type="button"
            className="text-link hover:underline"
            // Seeded with the policy's own target, marked as the safe choice, so
            // the pool starts from the model this policy already serves and the
            // operator adds the cheaper one rather than restating everything.
            onClick={() => {
              setBackend(KNN_BACKEND)
              setCandidates([target.trim() || "", ""])
              setWeights([])
              setSafeIndex(0)
            }}
          >
            + Let a router pick the cheapest good-enough model
          </button>
        ) : null}
        {candidates.length === 0 ? (
          <button
            type="button"
            className="text-link hover:underline"
            // An even split of the policy's own target with one more provider:
            // the neutral starting point for load balancing, which the operator
            // then skews. Seeding 90/10 would be guessing at a canary.
            onClick={() => {
              setBackend(WEIGHTED_BACKEND)
              setCandidates([target.trim() || "", ""])
              setWeights(["50", "50"])
              setSafeIndex(0)
            }}
          >
            + Split traffic across providers by weight
          </button>
        ) : null}
        {guardrails.length === 0 ? (
          // Disabled rather than hidden, and never disabled silently: a hidden
          // control teaches nothing, and a greyed-out one with no explanation
          // is worse. The reason sits next to it with the route to fixing it,
          // as text rather than a tooltip so it is readable on touch and by a
          // screen reader.
          <span className="flex flex-wrap items-baseline gap-2">
            <button
              type="button"
              disabled={!guardrails_.configured}
              aria-describedby={
                guardrails_.configured ? undefined : "guardrails-unavailable"
              }
              className={
                guardrails_.configured
                  ? "text-link hover:underline"
                  : "cursor-not-allowed text-muted opacity-60"
              }
              onClick={() =>
                setGuardrails([
                  { profile: "", mode: "block", on_unavailable: "block" },
                ])
              }
            >
              + Add guardrails
            </button>
            {guardrails_.configured ? null : (
              <span id="guardrails-unavailable" className="text-caption">
                No guardrails service is configured, so there would be nothing
                to call.{" "}
                <Link to="/tools" className="text-link hover:underline">
                  Set one up in Tools &amp; Guardrails
                </Link>
                .
              </span>
            )}
          </span>
        ) : null}
      </div>

      {/* Each of these explains a mode chosen above it, so it belongs beside
          that choice. The footer's caption is the one sentence about the save
          itself. */}
      {routed && !weighted ? (
        <p className="text-caption">
          A new router serves the model above until it has scored examples.
          Recording them is an API job for now (
          <code>POST /api/v1/routing/preferences/rank</code>); open{" "}
          <b>Examples</b> on the row afterwards to watch it warm up.
        </p>
      ) : null}
      {weighted ? (
        <p className="text-caption">
          Each request is drawn independently, so the shares hold over traffic
          rather than over any ten requests, and they behave the same behind any
          number of replicas.
        </p>
      ) : null}
      {outgrewAlias ? (
        <p className="text-warning text-xs">
          An alias holds one target. To add a fallback, a condition, or a
          guardrail, delete this alias and create a policy with the same name.
        </p>
      ) : null}
    </FormDialog>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

/** The affirmative kind mark: an accent dot and the word, in mono. */
function KindMark({ label }: { label: string }) {
  return (
    <span className="flex items-center gap-2 text-mono-caption text-foreground">
      <Dot className="bg-accent" />
      {label.toUpperCase()}
    </span>
  )
}

/** A `sm` control at the touch floor below `md`: 32px where there is a pointer,
 *  44px where there is not. */
const ACTION_CLASS = "min-h-11 md:min-h-0"

/** A list row's second line: what it serves, then what changes what can be done
 *  with it. One string rather than marks, so a narrow column ellipsizes it. */
function rowSummary(row: RoutingRow): string {
  return [
    servesSummary(row),
    row.user_id ? `for ${row.user_id}` : "",
    row.kind === "alias" ? "alias" : "",
    row.source === "config" ? "config" : "",
  ]
    .filter((part) => part !== "")
    .join(" · ")
}

/** One policy read-only, in the column its row opens.
 *
 *  The list column holds a name and a line, so every fact the table spread
 *  across lanes is here, and so are the controls that act on the row. Nothing
 *  the table showed became unreachable.
 */
function PolicyDetail({
  row,
  canEdit,
  isOperator,
  isReadinessShown,
  onToggleReadiness,
  onEdit,
  onDelete,
}: {
  row: RoutingRow
  canEdit: boolean
  isOperator: boolean
  isReadinessShown: boolean
  onToggleReadiness: () => void
  onEdit: () => void
  onDelete: () => void
}) {
  const chain = row.spec.on_failure ?? []
  const guardrails = row.spec.guardrails ?? []
  const pool = candidatesOf(row.spec)
  const isConfig = row.source === "config"
  const isEditable = isEditableInForm(row.spec)
  // Teaching is data, not configuration, so the panel is offered even for a
  // policy defined in config.yml: an operator can score examples for a policy
  // they cannot edit here, and without this that policy could never route.
  // Operator-only, because it reads `/routing/status`, which is
  // deployment-wide: an admin has no readiness to be shown.
  //
  // "Examples" rather than "Router": on a Routing page full of routing
  // policies, "Router" names the thing rather than what opens, and the count of
  // scored examples is the one number in there that changes.
  const hasReadiness = isOperator && routerBackendOf(row.spec) === KNN_BACKEND

  return (
    <div className="flex flex-col">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
        {/* Copy beside the heading rather than inside it: nested in the `h2`
            its name joins the heading's, which is then read as the policy name
            plus "Copy policy name". */}
        <div className="flex min-w-0 items-center gap-1">
          <h2 className="truncate text-title">{row.name}</h2>
          <CopyButton value={row.name} label="policy name" />
        </div>
        {/* No control at all for a caller who cannot act: each of these is
            either a write or the operator-only Examples read. Dense on a desk
            and at the touch floor on a phone, where they are the only controls
            in the column that is on screen. */}
        <div className="flex flex-wrap items-center gap-2">
          {hasReadiness ? (
            <Button
              size="sm"
              className={ACTION_CLASS}
              onPress={onToggleReadiness}
            >
              {isReadinessShown ? "Hide examples" : "Examples"}
            </Button>
          ) : null}
          {canEdit && !isConfig && isEditable ? (
            <Button size="sm" className={ACTION_CLASS} onPress={onEdit}>
              Edit
            </Button>
          ) : null}
          {canEdit && !isConfig ? (
            <Button size="sm" className={ACTION_CLASS} onPress={onDelete}>
              Delete
            </Button>
          ) : null}
        </div>
      </div>

      <div className="flex flex-col gap-2 border-b border-border px-4 py-4">
        <span className="text-overline">Serves</span>
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-body">{servesSummary(row)}</span>
          {/* The kind of routing, as an affirmative mark: a learned router or a
              weighted split is a decision somebody made about this policy,
              where a plain single-target policy is just the default shape. */}
          {pool.length > 0 ? (
            <KindMark label={routerLabelOf(row.spec)} />
          ) : row.is_dynamic ? (
            <KindMark label="Dynamic" />
          ) : null}
        </div>
        {chain.length > 0 ? (
          <span className="text-caption">If it fails: {chain.join(", ")}</span>
        ) : null}
      </div>

      <div className="grid gap-4 border-b border-border px-4 py-4 sm:grid-cols-3">
        <div className="flex flex-col gap-1">
          <span className="text-overline">Guards</span>
          <span className="text-body">
            {guardrails.length === 0
              ? "None"
              : guardrails
                  .map(
                    (guardrail) => `${guardrail.profile} (${guardrail.mode})`,
                  )
                  .join(", ")}
          </span>
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <span className="text-overline">Applies to</span>
          <span className="text-body">
            {(row.user_id ?? null) === null ? (
              "Every caller"
            ) : (
              <CopyableValue value={row.user_id ?? ""} label="user id" />
            )}
          </span>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-overline">Source</span>
          <span className="flex flex-wrap items-center gap-3 text-mono-caption text-muted">
            <span className="flex items-center gap-2">
              <Dot className={isConfig ? "bg-text-subtle" : "bg-accent"} />
              {row.source.toUpperCase()}
            </span>
            {row.kind === "alias" ? (
              <span className="text-mono-overline text-subtle">alias</span>
            ) : null}
          </span>
        </div>
      </div>

      {isConfig ? (
        <p className="max-w-prose px-4 py-3 text-caption">
          Set in config.yml, so it cannot be edited or deleted here.
        </p>
      ) : isEditable ? null : (
        <p className="max-w-prose px-4 py-3 text-caption">
          Uses options this form cannot show yet. Edit it through the API so
          nothing is lost.
        </p>
      )}

      {isReadinessShown ? (
        <div className="border-t border-border">
          <RouterReadiness
            policyName={row.name}
            candidates={pool}
            defaultTarget={defaultTargetOf(row.spec)}
            backend={routerBackendOf(row.spec) ?? KNN_BACKEND}
            scopedUserId={row.user_id ?? null}
            onClose={onToggleReadiness}
          />
        </div>
      ) : null}
    </div>
  )
}

/** The detail column with nothing open, which is where a wide viewport lands
 *  before the first click and where an empty page says what to do.
 *
 *  Three readings, the same three the page has: pick one, or the steps for a
 *  caller who can write and has no policies, or who defines these for one who
 *  cannot.
 */
function NothingOpen({
  canEdit,
  hasRows,
  isLoading,
}: {
  canEdit: boolean
  hasRows: boolean
  isLoading: boolean
}) {
  // Nothing rather than a guess: until the lists answer, "no policies yet" and
  // "pick one" are both claims about a page that has not loaded.
  if (isLoading) return null
  if (hasRows)
    return (
      <p className="px-4 py-10 text-center text-caption">
        Pick a policy to see what it serves and what it applies to.
      </p>
    )
  if (!canEdit)
    return (
      <div className="flex flex-col gap-2 px-4 py-5">
        <h2 className="text-title">No routing policies yet</h2>
        <p className="text-body">
          Policies that apply in your workspaces will be listed here once your
          organization&apos;s admins define them.
        </p>
      </div>
    )
  return (
    <div className="flex flex-col gap-3 px-4 py-5">
      <h2 className="text-title">No routing policies yet</h2>
      <ol className="flex list-decimal flex-col gap-1 pl-5 text-body">
        <li>
          Create a policy and point it at the model that should normally serve.
        </li>
        <li>
          Add a fallback chain so a provider outage does not become a failed
          request.
        </li>
        <li>
          Or split the traffic across two providers by weight, and move the
          shares as you learn.
        </li>
        <li>
          Or let a router choose per request between a cheap and a strong model,
          then teach it with a few scored examples.
        </li>
        <li>Have your callers send the policy name as their `model`.</li>
      </ol>
    </div>
  )
}

export function RoutingPage() {
  // Deliberately unscoped, unlike keys and usage. The gateway stores every
  // policy and alias in the default workspace on purpose, because resolution
  // reads a process-wide name-keyed cache: one stored elsewhere would be listed
  // as scoped while it resolved for everyone. Filtering this list by the
  // selected workspace would therefore show an empty page while those policies
  // were live for that workspace's traffic, and hide a policy the moment it was
  // created. Scope this when resolution is scoped, not before.
  //
  // Which list is asked depends on who is signed in (otari-ai#1942): an
  // operator reads the deployment-wide management view, and anyone else reads
  // the tenant-scoped `/organizations/me/*` pair. Both reads wait for the
  // context to settle rather than taking "not yet an operator" as "member", so
  // an operator's page does not fire a read it is about to drop.
  const organization = useOrganizationContext()
  const isOperator = isDeploymentOperator(organization.data)
  const isContextSettled =
    organization.data !== undefined || organization.isError
  const policies = useRoutingPolicies(isOperator)
  const memberPolicies = useOrganizationRoutingPolicies(
    isContextSettled && !isOperator,
  )
  const aliases = useAliases(isOperator)
  const memberAliases = useOrganizationAliases(isContextSettled && !isOperator)
  const deletePolicy = useDeleteRoutingPolicy()
  const deleteAlias = useDeleteAlias()
  const deleteOrgPolicy = useDeleteOrganizationRoutingPolicy()
  const deleteOrgAlias = useDeleteOrganizationAlias()
  // Where a tenant admin's write lands, and null for an operator, who writes
  // deployment-wide. The tenant surface requires the workspace named, so an
  // admin who belongs to none has nowhere to write and is shown the read-only
  // page: the switcher is seeded from their own memberships, not the
  // organization's whole list (otari-ai#1969).
  const { selected: selectedWorkspace } = useSelectedWorkspace()
  const writeWorkspaceId = isOperator
    ? null
    : (selectedWorkspace?.workspace_id ?? null)
  const canEdit =
    isOperator || (canManage(organization.data) && writeWorkspaceId !== null)
  // An admin's list spans every workspace of the organization, not just the
  // selected one, so a write to an existing row goes back to the workspace that
  // row lives in (`deleteWorkspaceFor` below, and the Edit dialog's
  // `workspaceId`). Using the selection would create a second policy of the
  // same name in the selected workspace and leave the edited one untouched.
  // A deep link may pre-fill the new-policy dialog with ?target=provider:model.
  const initialTarget = useUrlValue("target")
  const [adding, setAdding] = useState(initialTarget !== "")
  const [editing, setEditing] = useState<RoutingRow | null>(null)
  const [pendingDelete, setPendingDelete] = useState<RoutingRow>()
  // The open row, held as its key rather than as the row itself, so the detail
  // column reads from the list rather than from a snapshot taken at the click.
  const [openKey, setOpenKey] = useState<string | null>(null)
  // Readiness opens under the control that opened it, in the column showing the
  // policy it describes. It belongs to that policy, so opening another closes
  // it.
  const [isReadinessShown, setIsReadinessShown] = useState(false)
  // `adding` is seeded from ?target= before the membership context settles, so
  // the role is applied here rather than in the initializer: gating the
  // initializer would drop an operator's deep link, since `isOperator` is still
  // false at the moment it runs. A member arriving on that link gets the
  // read-only empty column instead of a form whose only outcome is a refusal.
  const isAdding = adding && canEdit

  // Aliases and policies are listed together: an alias is the one-target case,
  // and this page is the only place either is managed.
  //
  // Both operator lists are read through `isOperator` rather than relied on to
  // be empty because their hooks are disabled: a disabled query still hands
  // back whatever sits in the cache under its key, so a caller who was an
  // operator earlier in the session would keep seeing the deployment-wide rows
  // after being demoted. The gate belongs where the data is rendered.
  const rows: RoutingRow[] = [
    ...((isOperator ? policies.data : memberPolicies.data) ?? []).map(
      (policy) => ({
        ...policy,
        kind: "policy" as const,
      }),
    ),
    ...((isOperator ? aliases.data : memberAliases.data) ?? []).map(aliasAsRow),
  ].sort(
    (a, b) =>
      a.name.localeCompare(b.name) ||
      (a.user_id ?? "").localeCompare(b.user_id ?? ""),
  )
  // The context counts as loading too: until it settles, neither list has been
  // asked, and an empty column would read as "no policies" rather than "not
  // yet".
  const isListLoading =
    !isContextSettled ||
    (isOperator
      ? policies.isLoading || aliases.isLoading
      : memberPolicies.isLoading || memberAliases.isLoading)

  // A rename moves the row and a delete removes it, so the key can name a row
  // the list no longer has. Resolving it every render rather than holding the
  // row is what makes that a return to the prompt instead of a column
  // describing something that does not resolve any more.
  const openRow = rows.find((row) => rowKeyOf(row) === openKey)
  // Read by both halves of the empty column, its message and its press target,
  // so the two cannot disagree about whether the list has answered yet.
  const isListEmpty = !isListLoading && rows.length === 0

  // Which of the four delete surfaces a row goes to. The tenant one names the
  // workspace and has no user scope; the deployment-wide one defaults the
  // workspace and keeps it.
  const deleteWorkspaceFor = (row: RoutingRow) =>
    isOperator ? null : (row.workspace_id ?? writeWorkspaceId)
  const deleteMutationFor = (row: RoutingRow) =>
    deleteWorkspaceFor(row) !== null
      ? row.kind === "alias"
        ? deleteOrgAlias
        : deleteOrgPolicy
      : row.kind === "alias"
        ? deleteAlias
        : deletePolicy
  // Resolved for the pending row alone, not as a chain over all four: a refusal
  // stays on its mutation until the next call, so reading every one of them
  // would report the last row's failure over this row's confirm.
  const pendingDeleteMutation = pendingDelete
    ? deleteMutationFor(pendingDelete)
    : undefined

  const startCreate = () => {
    setEditing(null)
    setAdding(true)
  }

  return (
    <div className="flex flex-col gap-6">
      <PageIntro title="Routing">
        {/* Three readings of the same page, because what a caller may do here
            differs: an operator sees the deployment-wide capabilities, an
            admin sees what their own workspace writes reach, and a member is
            told who manages these rather than offered a control they would
            be refused. */}
        {isOperator
          ? "Named models your callers send as `model`. A policy decides which real model serves each request, what is tried if that fails, and which guardrails always run. It can also split traffic across providers by weight, or let a router learn which prompts a cheaper model handles just as well."
          : canEdit
            ? "Named models your callers send as `model`. A policy decides which real model serves each request, what is tried if that fails, and which guardrails always run. What you create here applies in the selected workspace, and can name any model your organization has a provider key for."
            : "Named models your callers send as `model`. A policy decides which real model serves each request, what is tried if that fails, and which guardrails always run. These are the ones in force in your workspaces; your organization's admins manage them."}
      </PageIntro>

      {/* The reads only. A delete reports inside its own confirm dialog and a
          write inside the form dialog that made it, which is where the operator
          is looking in each case. */}
      <ErrorBanner
        error={
          policies.error ??
          memberPolicies.error ??
          aliases.error ??
          memberAliases.error
        }
      />

      <ListDetail
        listLabel="Policies"
        backLabel="All policies"
        detailLabel="Policy detail"
        listAction={
          canEdit ? (
            // In the column it adds to rather than beside the page title: the
            // dialog it opens sits over the page, so the control belongs to the
            // list it lengthens.
            <Button
              size="sm"
              variant="primary"
              className={ACTION_CLASS}
              onPress={startCreate}
            >
              Create policy
            </Button>
          ) : undefined
        }
        isDetailShown={openRow !== undefined}
        onShowList={() => setOpenKey(null)}
        empty={
          isListLoading
            ? "Loading…"
            : !isListEmpty
              ? undefined
              : canEdit
                ? "Create your first policy"
                : "No routing policies yet."
        }
        // Only once the lists have answered, and only for a caller who can
        // write: a press whose one outcome is a refusal is not an invitation.
        onEmptyPress={canEdit && isListEmpty ? startCreate : undefined}
        list={rows.map((row) => (
          <ListDetailRow
            key={rowKeyOf(row)}
            label={row.name}
            isSelected={rowKeyOf(row) === openKey}
            onSelect={() => {
              setIsReadinessShown(false)
              setOpenKey(rowKeyOf(row))
            }}
          >
            {rowSummary(row)}
          </ListDetailRow>
        ))}
        detail={
          openRow === undefined ? (
            <NothingOpen
              canEdit={canEdit}
              hasRows={rows.length > 0}
              isLoading={isListLoading}
            />
          ) : (
            <PolicyDetail
              row={openRow}
              canEdit={canEdit}
              isOperator={isOperator}
              isReadinessShown={isReadinessShown}
              onToggleReadiness={() => setIsReadinessShown((shown) => !shown)}
              onEdit={() => {
                setAdding(false)
                setEditing(openRow)
              }}
              onDelete={() => setPendingDelete(openRow)}
            />
          )
        }
      />

      {isAdding ? (
        <PolicyForm
          existing={null}
          initialTarget={initialTarget}
          workspaceId={writeWorkspaceId}
          onClose={() => setAdding(false)}
        />
      ) : null}
      {editing !== null ? (
        <PolicyForm
          existing={editing}
          workspaceId={
            isOperator ? null : (editing.workspace_id ?? writeWorkspaceId)
          }
          onClose={() => setEditing(null)}
        />
      ) : null}

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        // Cleared on the way out: a refusal otherwise sits on the mutation and
        // greets the next row's confirm as if that row had failed.
        onOpenChange={(open) => {
          if (open) return
          setPendingDelete(undefined)
          deletePolicy.reset()
          deleteAlias.reset()
          deleteOrgPolicy.reset()
          deleteOrgAlias.reset()
        }}
        heading={
          pendingDelete?.kind === "alias" ? "Delete alias" : "Delete policy"
        }
        body={
          pendingDelete
            ? `${pendingDelete.name} stops resolving. A request that still sends it as its model is refused, so update the callers that name it.`
            : null
        }
        confirmLabel={
          pendingDelete?.kind === "alias" ? "Delete alias" : "Delete policy"
        }
        isPending={pendingDeleteMutation?.isPending ?? false}
        error={pendingDeleteMutation?.error}
        onConfirm={() => {
          if (!pendingDelete) return
          const onSuccess = () => setPendingDelete(undefined)
          const rowWorkspace = deleteWorkspaceFor(pendingDelete)
          if (rowWorkspace !== null) {
            const scoped = {
              name: pendingDelete.name,
              workspaceId: rowWorkspace,
            }
            if (pendingDelete.kind === "alias")
              deleteOrgAlias.mutate(scoped, { onSuccess })
            else deleteOrgPolicy.mutate(scoped, { onSuccess })
            return
          }
          const deployment = {
            name: pendingDelete.name,
            userId: pendingDelete.user_id,
          }
          if (pendingDelete.kind === "alias")
            deleteAlias.mutate(deployment, { onSuccess })
          else deletePolicy.mutate(deployment, { onSuccess })
        }}
      />
    </div>
  )
}
