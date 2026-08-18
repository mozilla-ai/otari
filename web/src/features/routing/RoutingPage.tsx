import { Button, Card, Chip } from "@heroui/react"
import { Link } from "@tanstack/react-router"
import { useCallback, useMemo, useState } from "react"

import type {
  AliasResponse,
  PolicyGuardrail,
  PolicySpec,
  RoutingPolicyResponse,
} from "@/client"
import { ModelComboBox } from "@/features/models/ModelComboBox"
import { RouterReadiness } from "@/features/routing/RouterReadiness"
import { UserComboBox } from "@/features/users/UserComboBox"
import {
  useAliases,
  useCreateAlias,
  useDeleteAlias,
  useDeleteRoutingPolicy,
  useRoutingPolicies,
  useSetRoutingPolicy,
  useToolSettings,
  useUsers,
} from "@/shared/api/hooks"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import { Field } from "@/shared/components/Field"
import {
  ConfirmButton,
  CopyableValue,
  EmptyState,
  ErrorBanner,
  PageHeader,
} from "@/shared/components/ui"
import { useUrlValue } from "@/shared/helpers/urlState"

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

/** One line summarising what a policy serves, for the table. */
function servesSummary(policy: RoutingPolicyResponse): string {
  const chain = policy.spec.on_failure ?? []
  const pool = candidatesOf(policy.spec)
  if (pool.length > 0 && routerBackendOf(policy.spec) === WEIGHTED_BACKEND) {
    // The split shape, not the model names: two provider:model strings do not fit a
    // table cell, and the shares are what distinguishes one weighted policy from
    // another. The pool is spelled out in the editor and in explain.
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
    <button
      type="button"
      aria-pressed={scoped === value}
      onClick={() => onChange(value ? "" : null)}
      className={
        scoped === value
          ? "rounded-md bg-surface px-3 py-1.5 text-sm font-medium text-foreground shadow-sm"
          : "rounded-md px-3 py-1.5 text-sm text-muted hover:text-foreground"
      }
    >
      {label}
    </button>
  )

  return (
    <div className="flex flex-col gap-3">
      <div>
        <span className="text-sm font-medium text-foreground">Applies to</span>
        <p className="text-xs text-muted">
          A global policy resolves for every caller. A user-scoped one resolves
          only for that user, and takes precedence over a global policy of the
          same name.
        </p>
      </div>
      <div className="flex w-fit items-center gap-1 rounded-lg bg-surface-alt p-1">
        {modeButton(false, "Every caller")}
        {modeButton(true, "One user")}
      </div>
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
      <span className="text-sm font-medium text-foreground">{label}</span>
      <div className="flex w-fit items-center gap-1 rounded-lg bg-surface-alt p-1">
        {MODE_VALUES.map((mode) => (
          <button
            key={mode}
            type="button"
            aria-pressed={value === mode}
            onClick={() => onChange(mode)}
            className={
              value === mode
                ? "rounded-md bg-surface px-3 py-1 text-sm font-medium text-foreground shadow-sm"
                : "rounded-md px-3 py-1 text-sm text-muted hover:text-foreground"
            }
          >
            {mode}
          </button>
        ))}
      </div>
      {hint === undefined ? null : (
        <span className="text-xs text-muted">{hint}</span>
      )}
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
function PolicyForm({
  existing,
  initialTarget = "",
  onClose,
}: {
  existing: RoutingRow | null
  initialTarget?: string
  onClose: () => void
}) {
  const save = useSetRoutingPolicy()
  const saveAlias = useCreateAlias()
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
  const pending = save.isPending || saveAlias.isPending

  const submit = () => {
    if (!canSubmit || outgrewAlias) return
    const scope = userId === null ? null : userId.trim()
    if (editingAlias) {
      saveAlias.mutate(
        { name: name.trim(), target: effectiveTarget.trim(), user_id: scope },
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
    <div className="flex flex-col gap-4">
      <Card>
        <Card.Content className="flex flex-col gap-5 p-5">
          <div className="text-sm font-semibold text-foreground">
            {editing ? (
              <>
                Edit {existing.kind === "alias" ? "alias" : "policy"}{" "}
                <code>{existing.name}</code>
                {existing.user_id ? (
                  <>
                    {" "}
                    for user <code>{existing.user_id}</code>
                  </>
                ) : null}
              </>
            ) : (
              "New routing policy"
            )}
          </div>
          <ErrorBanner error={save.error ?? saveAlias.error} />

          <div className="grid gap-4 sm:grid-cols-2">
            {editingAlias ? (
              <div className="flex flex-col gap-1">
                <span className="text-sm font-medium text-foreground">
                  Alias name
                </span>
                <code className="text-sm text-muted">{previousName}</code>
                <span className="text-xs text-muted">
                  An alias name is its key and cannot be changed here. Delete
                  and recreate to change it.
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
                      Renames <code>{previousName}</code> on save. Callers have
                      to send the new name from then on, and usage already
                      recorded keeps the old one.
                    </span>
                  ) : editing ? (
                    "What callers send as `model`. Change it to rename the policy."
                  ) : (
                    "What callers send as `model`."
                  )
                }
              />
            )}
            {routed ? (
              <div className="flex flex-col gap-1">
                <span className="text-sm font-medium text-foreground">
                  Serves
                </span>
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
            <p className="text-xs text-muted">
              Who this applies to is the other half of the key. It cannot be
              changed here: delete and recreate to move it between scopes.
            </p>
          ) : (
            <ScopePicker userId={userId} onChange={setUserId} />
          )}

          {/* Conditional tier-down */}
          {conditions.length > 0 ? (
            <div className="flex flex-col gap-3 rounded-lg border border-border p-3">
              <div>
                <span className="text-sm font-medium text-foreground">
                  Instead, when the budget fills up
                </span>
                <p className="text-xs text-muted">
                  Checked before the model above. A threshold must be under 100:
                  the budget gate refuses a request before selection once the
                  cap is reached, so a rule at 100 could never fire.
                </p>
              </div>
              {conditions.map((condition, index) => (
                <div key={index} className="flex flex-wrap items-end gap-3">
                  <Field
                    label="Budget used at least (%)"
                    value={String(condition.threshold)}
                    onChange={(value) =>
                      setConditions((prev) =>
                        prev.map((c, i) =>
                          i === index
                            ? { ...c, threshold: Number(value) || 0 }
                            : c,
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
                  <Button
                    variant="ghost"
                    onPress={() =>
                      setConditions((prev) =>
                        prev.filter((_, i) => i !== index),
                      )
                    }
                  >
                    Remove
                  </Button>
                </div>
              ))}
            </div>
          ) : null}

          {/* The routed pool: one control for both backends, because both are "these
              models, one of them per request". What differs is who decides, and
              whether a share sits next to each entry. */}
          {candidates.length > 0 ? (
            <div className="flex flex-col gap-3 rounded-lg border border-border p-3">
              <div>
                <span className="text-sm font-medium text-foreground">
                  {weighted
                    ? "Split traffic between"
                    : "The router chooses between"}
                </span>
                <p className="text-xs text-muted">
                  {weighted
                    ? "Each request goes to one of these, drawn in proportion to its share. Shares are relative, so 70 and 30 mean the same as 7 and 3. No pricing needed."
                    : "For each request, the cheapest of these that past scoring says is good enough. Every model here needs pricing, because the router weighs quality against cost."}
                </p>
              </div>
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
                            prev.map((weight, i) =>
                              i === index ? value : weight,
                            ),
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
                  <label className="flex items-center gap-2 pb-2 text-xs text-foreground">
                    <input
                      type="radio"
                      name="router-safe-choice"
                      checked={safeIndex === index}
                      onChange={() => setSafeIndex(index)}
                    />
                    {weighted ? "Serves on opt-out" : "Serves when unsure"}
                  </label>
                  <Button
                    variant="ghost"
                    onPress={() => {
                      setCandidates((prev) =>
                        prev.filter((_, i) => i !== index),
                      )
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
                </div>
              ))}
              <p className="text-xs text-muted">
                {weighted ? (
                  <>
                    The marked model serves a caller who sends{" "}
                    <code>Otari-Router: off</code>, which is the way to pin
                    traffic to one provider during an incident. A model that
                    fails before responding moves the request to another model
                    in this pool, by the same shares, before any fallback below.
                  </>
                ) : (
                  <>
                    The marked model serves whenever the router does not choose:
                    too few scored examples, a weakly supported pick, a request
                    carrying tools, or a caller sending{" "}
                    <code>Otari-Router: off</code>. Mark the one you would have
                    picked without a router.
                  </>
                )}
              </p>
              {candidates.length < 2 ? (
                <p className="text-xs text-danger">
                  Name at least two models.{" "}
                  {weighted ? "Splitting traffic one way" : "Ranking one"} is
                  not a routing decision.
                </p>
              ) : null}
              {duplicateCandidate ? (
                <p className="text-xs text-danger">
                  Name each model once.{" "}
                  {weighted
                    ? "A model listed twice has one share, not two, so the split saved would not be the one shown."
                    : "A pool that repeats a model is refused."}
                </p>
              ) : null}
              {weighted && !weightsWellFormed ? (
                <p className="text-xs text-danger">
                  Every share is a number of zero or more. Use 0 to drain a
                  model without removing it.
                </p>
              ) : weighted && !splitReady ? (
                <p className="text-xs text-danger">
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
                      : "text-sm text-accent hover:underline"
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
                    A policy dispatches at most {MAX_CANDIDATES} models,
                    counting the fallback chain. Remove a fallback to add
                    another.
                  </span>
                ) : null}
              </div>
            </div>
          ) : null}

          {/* Failure chain */}
          {chain.length > 0 ? (
            <div className="flex flex-col gap-3 rounded-lg border border-border p-3">
              <div>
                <span className="text-sm font-medium text-foreground">
                  If that fails, try
                </span>
                <p className="text-xs text-muted">
                  Tried in order after a retryable failure. Not tried once
                  tokens have started streaming, or after a 400/401/403, which
                  every provider would reject the same way.
                </p>
              </div>
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
                  <Button
                    variant="ghost"
                    onPress={() =>
                      setChain((prev) => prev.filter((_, i) => i !== index))
                    }
                  >
                    Remove
                  </Button>
                </div>
              ))}
              <div className="flex flex-wrap items-baseline gap-2">
                <button
                  type="button"
                  disabled={atCandidateCap}
                  className={
                    atCandidateCap
                      ? "cursor-not-allowed text-sm text-muted opacity-60"
                      : "text-sm text-accent hover:underline"
                  }
                  onClick={() => setChain((prev) => [...prev, ""])}
                >
                  + Another fallback
                </button>
                {atCandidateCap ? (
                  <span className="text-xs text-muted">
                    A policy dispatches at most {MAX_CANDIDATES} models in
                    total.
                  </span>
                ) : null}
              </div>
            </div>
          ) : null}

          {/* Guardrails */}
          {guardrails.length > 0 ? (
            <div className="flex flex-col gap-3 rounded-lg border border-border p-3">
              <div>
                <span className="text-sm font-medium text-foreground">
                  Always check
                </span>
                <p className="text-xs text-muted">
                  Runs on every request through this policy. Callers can add
                  their own guardrails but cannot weaken these.
                </p>
                {guardrails_.configured ? null : (
                  <p className="mt-1 text-xs text-warning">
                    No guardrails service is configured, so these cannot run.
                    With `if the service is down` set to block, every request
                    through this policy is refused until one is configured.{" "}
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
                          prev.map((g, i) =>
                            i === index ? { ...g, mode } : g,
                          ),
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
                  </div>
                  {guardrail.mode === "block" &&
                  (guardrail.on_unavailable ?? "block") === "block" ? (
                    <div className="text-xs text-warning">
                      With both set to block, a guardrails-service outage
                      rejects every request through this policy, ahead of any
                      fallback above.
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
                className="text-accent hover:underline"
                onClick={() => setConditions([{ threshold: 80, target: "" }])}
              >
                + Tier down when the budget fills up
              </button>
            ) : null}
            {chain.length === 0 ? (
              <button
                type="button"
                className="text-accent hover:underline"
                onClick={() => setChain([""])}
              >
                + Add a fallback chain
              </button>
            ) : null}
            {candidates.length === 0 ? (
              <button
                type="button"
                className="text-accent hover:underline"
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
                className="text-accent hover:underline"
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
                    guardrails_.configured
                      ? undefined
                      : "guardrails-unavailable"
                  }
                  className={
                    guardrails_.configured
                      ? "text-accent hover:underline"
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
                  <span
                    id="guardrails-unavailable"
                    className="text-xs text-muted"
                  >
                    No guardrails service is configured, so there would be
                    nothing to call.{" "}
                    <Link to="/tools" className="text-accent hover:underline">
                      Set one up in Tools &amp; Guardrails
                    </Link>
                    .
                  </span>
                )}
              </span>
            ) : null}
          </div>

          <div className="flex items-center gap-3">
            <Button
              variant="primary"
              isDisabled={!canSubmit || pending || outgrewAlias}
              onPress={submit}
            >
              {pending ? "Saving…" : editing ? "Save" : "Create policy"}
            </Button>
            <Button variant="ghost" onPress={onClose}>
              Cancel
            </Button>
            <span className="text-xs text-muted">
              In effect for new requests within 30s.
            </span>
            {routed && !weighted ? (
              <span className="text-xs text-muted">
                A new router serves the model above until it has scored
                examples. Recording them is an API job for now (
                <code>POST /v1/routing/preferences/rank</code>); open{" "}
                <b>Examples</b> on the row afterwards to watch it warm up.
              </span>
            ) : null}
            {weighted ? (
              <span className="text-xs text-muted">
                Each request is drawn independently, so the shares hold over
                traffic rather than over any ten requests, and they behave the
                same behind any number of replicas.
              </span>
            ) : null}
            {outgrewAlias ? (
              <span className="text-xs text-warning">
                An alias holds one target. To add a fallback, a condition, or a
                guardrail, delete this alias and create a policy with the same
                name.
              </span>
            ) : null}
          </div>
        </Card.Content>
      </Card>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function RoutingPage() {
  // Deliberately unscoped, unlike keys and usage. The gateway stores every
  // policy and alias in the default workspace on purpose, because resolution
  // reads a process-wide name-keyed cache: one stored elsewhere would be listed
  // as scoped while it resolved for everyone. Filtering this list by the
  // selected workspace would therefore show an empty page while those policies
  // were live for that workspace's traffic, and hide a policy the moment it was
  // created. Scope this when resolution is scoped, not before.
  const policies = useRoutingPolicies()
  const aliases = useAliases()
  const deletePolicy = useDeleteRoutingPolicy()
  const deleteAlias = useDeleteAlias()
  // A deep link may pre-fill the add form with ?target=provider:model.
  const initialTarget = useUrlValue("target")
  const [adding, setAdding] = useState(initialTarget !== "")
  const [editing, setEditing] = useState<RoutingRow | null>(null)
  // Readiness opens inline under its own row (DataTable's accordion), because it
  // describes one policy and the operator clicked that policy. A card above the
  // table would put the panel nowhere near the control that opened it.
  const [expanded, setExpanded] = useState<string | null>(null)

  // Aliases and policies are listed together: an alias is the one-target case,
  // and this page is the only place either is managed.
  const rows: RoutingRow[] = [
    ...(policies.data ?? []).map((policy) => ({
      ...policy,
      kind: "policy" as const,
    })),
    ...(aliases.data ?? []).map(aliasAsRow),
  ].sort(
    (a, b) =>
      a.name.localeCompare(b.name) ||
      (a.user_id ?? "").localeCompare(b.user_id ?? ""),
  )

  // Stable so DataTable's row cache holds; see its docstring.
  const renderDetail = useCallback(
    (row: RoutingRow) => (
      <RouterReadiness
        policyName={row.name}
        candidates={candidatesOf(row.spec)}
        defaultTarget={defaultTargetOf(row.spec)}
        backend={routerBackendOf(row.spec) ?? KNN_BACKEND}
        scopedUserId={row.user_id ?? null}
        onClose={() => setExpanded(null)}
      />
    ),
    [],
  )

  const columns = useMemo<DataTableColumn<RoutingRow>[]>(
    () => [
      {
        id: "name",
        header: "Policy",
        isRowHeader: true,
        cell: (policy) => (
          <CopyableValue value={policy.name} label="policy name" />
        ),
      },
      {
        id: "serves",
        header: "Serves",
        cell: (policy) => (
          <div className="flex items-center gap-2">
            <span className="text-sm text-foreground">
              {servesSummary(policy)}
            </span>
            {candidatesOf(policy.spec).length > 0 ? (
              <Chip size="sm" color="accent">
                {routerLabelOf(policy.spec)}
              </Chip>
            ) : policy.is_dynamic ? (
              <Chip size="sm" color="accent">
                Dynamic
              </Chip>
            ) : null}
          </div>
        ),
      },
      {
        id: "guards",
        header: "Guards",
        cell: (policy) => {
          const guardrails = policy.spec.guardrails ?? []
          if (guardrails.length === 0)
            return <span className="text-muted">–</span>
          return (
            <span className="text-sm text-foreground">
              {guardrails
                .map((guardrail) => `${guardrail.profile} (${guardrail.mode})`)
                .join(", ")}
            </span>
          )
        },
      },
      {
        id: "scope",
        header: "Applies to",
        cell: (policy) =>
          (policy.user_id ?? null) === null ? (
            <span className="text-muted">Every caller</span>
          ) : (
            <CopyableValue value={policy.user_id ?? ""} label="user id" />
          ),
      },
      {
        id: "source",
        header: "Source",
        cell: (row) => (
          <div className="flex items-center gap-1">
            <Chip
              size="sm"
              color={row.source === "config" ? "default" : "accent"}
            >
              {row.source}
            </Chip>
            {row.kind === "alias" ? (
              <Chip size="sm" color="default">
                alias
              </Chip>
            ) : null}
          </div>
        ),
      },
      {
        id: "actions",
        header: "",
        cell: (policy) => {
          // Teaching is data, not configuration, so it is offered even for a policy
          // defined in config.yml: an operator can score examples for a policy they
          // cannot edit here, and without this that policy could never route.
          // "Examples" rather than "Router": on a Routing page full of routing
          // policies, "Router" names the thing rather than what opens, and the count
          // of scored examples is the one number in there that changes. Outlined
          // rather than ghost so it reads as the row's distinct affordance next to
          // Edit and Delete.
          // Only for a backend that learns: a weighted policy has nothing to teach,
          // so offering Examples on one would promise a screen that cannot help it.
          const readiness = routerBackendOf(policy.spec) === KNN_BACKEND && (
            <Button
              size="sm"
              variant="outline"
              onPress={() =>
                setExpanded((current) =>
                  current === rowKeyOf(policy) ? null : rowKeyOf(policy),
                )
              }
            >
              {expanded === rowKeyOf(policy) ? "Hide examples" : "Examples"}
            </Button>
          )
          return policy.source === "config" ? (
            <div className="flex items-center justify-end gap-2">
              {readiness}
              <span className="text-xs text-muted">set in config.yml</span>
            </div>
          ) : (
            <div className="flex items-center justify-end gap-2">
              {readiness}
              {isEditableInForm(policy.spec) ? (
                <Button
                  size="sm"
                  variant="ghost"
                  onPress={() => {
                    // The table stays mounted while the create form is open, so
                    // Edit is still reachable from it. Closing the other panels
                    // keeps this to one form: two stacked forms do not recover on
                    // their own, since each only closes when cancelled.
                    setAdding(false)
                    setEditing(policy)
                  }}
                >
                  Edit
                </Button>
              ) : (
                <span className="text-xs text-muted">
                  Uses options this form cannot show yet. Edit it through the
                  API so nothing is lost.
                </span>
              )}
              <ConfirmButton
                confirmLabel="Confirm"
                isPending={deletePolicy.isPending || deleteAlias.isPending}
                onConfirm={() =>
                  policy.kind === "alias"
                    ? deleteAlias.mutate({
                        name: policy.name,
                        userId: policy.user_id,
                      })
                    : deletePolicy.mutate({
                        name: policy.name,
                        userId: policy.user_id,
                      })
                }
              >
                Delete
              </ConfirmButton>
            </div>
          )
        },
      },
    ],
    [deletePolicy, deleteAlias, expanded],
  )

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Routing"
        description="Named models your callers send as `model`. A policy decides which real model serves each request, what is tried if that fails, and which guardrails always run. It can also split traffic across providers by weight, or let a router learn which prompts a cheaper model handles just as well."
        action={
          adding || editing !== null ? undefined : (
            <Button
              variant="primary"
              onPress={() => {
                setEditing(null)
                setAdding(true)
              }}
            >
              New policy
            </Button>
          )
        }
      />

      <ErrorBanner
        error={
          policies.error ??
          aliases.error ??
          deletePolicy.error ??
          deleteAlias.error
        }
      />

      {adding ? (
        <PolicyForm
          existing={null}
          initialTarget={initialTarget}
          onClose={() => setAdding(false)}
        />
      ) : null}
      {editing !== null ? (
        <PolicyForm existing={editing} onClose={() => setEditing(null)} />
      ) : null}

      {rows.length === 0 &&
      !policies.isLoading &&
      !aliases.isLoading &&
      !adding ? (
        <EmptyState title="No routing policies yet">
          <ol className="flex list-decimal flex-col gap-1 pl-5 text-sm text-muted">
            <li>
              Create a policy and point it at the model that should normally
              serve.
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
              Or let a router choose per request between a cheap and a strong
              model, then teach it with a few scored examples.
            </li>
            <li>Have your callers send the policy name as their `model`.</li>
          </ol>
        </EmptyState>
      ) : (
        <DataTable
          ariaLabel="Routing policies"
          columns={columns}
          rows={rows}
          getRowKey={rowKeyOf}
          detailKey={expanded}
          renderDetail={renderDetail}
          isLoading={policies.isLoading || aliases.isLoading}
          emptyContent="No routing policies yet."
        />
      )}
    </div>
  )
}
