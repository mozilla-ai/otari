import { Button, Card, Chip } from "@heroui/react";
import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import type { AliasResponse, PolicyGuardrail, PolicySpec, RoutingPolicyResponse } from "@/api/types";
import {
  useAliases,
  useCreateAlias,
  useDeleteAlias,
  useDeleteRoutingPolicy,
  useRoutingPolicies,
  useSetRoutingPolicy,
  useToolSettings,
  useUsers,
} from "@/api/hooks";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { Field } from "@/components/Field";
import { ModelComboBox } from "@/components/ModelComboBox";
import { UserComboBox } from "@/components/UserComboBox";
import { ConfirmButton, CopyableValue, EmptyState, ErrorBanner, PageHeader } from "@/components/ui";

/** A row on this page: either a routing policy or a stored/config alias.
 *
 *  An alias is the one-target case of a policy, so the two are listed together
 *  and this page is the single place either is managed. They still live in
 *  different tables behind different endpoints, so `kind` decides which API a
 *  write goes to; it is not cosmetic.
 */
type RoutingRow = RoutingPolicyResponse & { kind: "policy" | "alias" };

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
  };
}

// Scope is part of the identity, so it is part of the row key: the same policy
// name can exist globally and per user, and keying on the name alone would
// collapse those rows into one. Same reasoning (and encoding) as the alias table.
const rowKeyOf = (row: RoutingRow): string => JSON.stringify([row.kind, row.user_id, row.name]);

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
function useGuardrailsConfigured(): { configured: boolean; isLoading: boolean } {
  const settings = useToolSettings();
  const field = settings.data?.fields.find((entry) => entry.key === "guardrails_url");
  const value = typeof field?.value === "string" ? field.value.trim() : "";
  return { configured: settings.isLoading || value !== "", isLoading: settings.isLoading };
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
  return spec.select.every((entry) => {
    if (entry.default !== undefined) return entry.when === undefined;
    if (entry.router !== undefined) return false;
    const when = entry.when;
    if (when === undefined || entry.target === undefined) return false;
    const keys = Object.keys(when);
    return keys.length === 1 && keys[0] === "budget_used_pct" && when.budget_used_pct?.gte !== undefined;
  });
}

/** The fallthrough target of a spec, which every valid spec has exactly one of. */
function defaultTargetOf(spec: PolicySpec): string {
  return spec.select.find((entry) => entry.default !== undefined)?.default ?? "";
}

/** The conditional entries, i.e. everything that is not the fallthrough. */
function conditionsOf(spec: PolicySpec): { threshold: number; target: string }[] {
  return spec.select
    .filter((entry) => entry.when?.budget_used_pct?.gte !== undefined && entry.target !== undefined)
    .map((entry) => ({ threshold: entry.when!.budget_used_pct!.gte!, target: entry.target! }));
}

/** One line summarising what a policy serves, for the table. */
function servesSummary(policy: RoutingPolicyResponse): string {
  const chain = policy.spec.on_failure ?? [];
  if (policy.is_dynamic) {
    const total = 1 + chain.length;
    return `Chosen per request · ${total} candidate${total === 1 ? "" : "s"}`;
  }
  const target = defaultTargetOf(policy.spec);
  return chain.length > 0 ? `${target}  +${chain.length} on failure` : target;
}

// ---------------------------------------------------------------------------
// Editor
// ---------------------------------------------------------------------------

/** Who a policy applies to. Same control and wording as the alias scope picker,
 *  because it is the same decision. */
function ScopePicker({ userId, onChange }: { userId: string | null; onChange: (userId: string | null) => void }) {
  const users = useUsers();
  const scoped = userId !== null;

  const modeButton = (value: boolean, label: string) => (
    <button
      type="button"
      aria-pressed={scoped === value}
      onClick={() => onChange(value ? "" : null)}
      className={
        scoped === value
          ? "rounded-md bg-white px-3 py-1.5 text-sm font-medium text-[var(--otari-ink)] shadow-sm"
          : "rounded-md px-3 py-1.5 text-sm text-[var(--otari-muted)] hover:text-[var(--otari-ink)]"
      }
    >
      {label}
    </button>
  );

  return (
    <div className="flex flex-col gap-3">
      <div>
        <span className="text-sm font-medium text-[var(--otari-ink)]">Applies to</span>
        <p className="text-xs text-[var(--otari-muted)]">
          A global policy resolves for every caller. A user-scoped one resolves only for that user, and takes
          precedence over a global policy of the same name.
        </p>
      </div>
      <div className="flex w-fit items-center gap-1 rounded-lg bg-[var(--otari-bg)] p-1">
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
          unknownHint={<span className="text-red-700">No such user. Pick an existing one.</span>}
        />
      ) : null}
    </div>
  );
}

const MODE_VALUES = ["block", "monitor"] as const;

/** A two-value mode switch. The codebase has no Select component and four
 *  hand-rolled `aria-pressed` groups, so this follows that pattern rather than
 *  introducing a fifth idiom. */
function ModeToggle({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint?: string;
  value: "block" | "monitor";
  onChange: (value: "block" | "monitor") => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-sm font-medium text-[var(--otari-ink)]">{label}</span>
      <div className="flex w-fit items-center gap-1 rounded-lg bg-[var(--otari-bg)] p-1">
        {MODE_VALUES.map((mode) => (
          <button
            key={mode}
            type="button"
            aria-pressed={value === mode}
            onClick={() => onChange(mode)}
            className={
              value === mode
                ? "rounded-md bg-white px-3 py-1 text-sm font-medium text-[var(--otari-ink)] shadow-sm"
                : "rounded-md px-3 py-1 text-sm text-[var(--otari-muted)] hover:text-[var(--otari-ink)]"
            }
          >
            {mode}
          </button>
        ))}
      </div>
      {hint === undefined ? null : <span className="text-xs text-[var(--otari-muted)]">{hint}</span>}
    </div>
  );
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
  existing: RoutingRow | null;
  initialTarget?: string;
  onClose: () => void;
}) {
  const save = useSetRoutingPolicy();
  const saveAlias = useCreateAlias();
  const editing = existing !== null;
  // Editing an alias writes back through the alias API: it is still a row in
  // model_aliases, and silently rewriting it as a policy would leave the original
  // behind under the same name.
  const editingAlias = existing?.kind === "alias";
  const guardrails_ = useGuardrailsConfigured();

  const [name, setName] = useState(existing?.name ?? "");
  const [userId, setUserId] = useState<string | null>(existing?.user_id ?? null);
  const [target, setTarget] = useState(existing ? defaultTargetOf(existing.spec) : initialTarget);
  const [chain, setChain] = useState<string[]>(existing?.spec.on_failure ?? []);
  const [conditions, setConditions] = useState(existing ? conditionsOf(existing.spec) : []);
  const [guardrails, setGuardrails] = useState<PolicyGuardrail[]>(existing?.spec.guardrails ?? []);

  const nameHasDelimiter = /[:/]/.test(name);
  const scopeReady = userId === null || userId.trim() !== "";
  const conditionsReady = conditions.every((c) => c.target.trim() !== "" && c.threshold > 0 && c.threshold < 100);
  const guardrailsReady = guardrails.every((g) => g.profile.trim() !== "");
  const canSubmit =
    name.trim() !== "" &&
    target.trim() !== "" &&
    !nameHasDelimiter &&
    scopeReady &&
    conditionsReady &&
    guardrailsReady &&
    chain.every((entry) => entry.trim() !== "");

  // Built in plan order, with the fallthrough last, which is what the schema
  // requires: an entry after the default could never be reached.
  const spec: PolicySpec = useMemo(
    () => ({
      select: [
        ...conditions.map((condition) => ({
          when: { budget_used_pct: { gte: condition.threshold } },
          target: condition.target.trim(),
        })),
        { default: target.trim() },
      ],
      ...(chain.length > 0 ? { on_failure: chain.map((entry) => entry.trim()) } : {}),
      ...(guardrails.length > 0 ? { guardrails } : {}),
    }),
    [conditions, target, chain, guardrails],
  );

  // An alias has exactly one target, so growing one a chain, a condition, or a
  // guardrail makes it a policy. Saving it as a policy alone would leave the alias
  // row in place under the same name, and the API refuses that collision, so the
  // form keeps an alias an alias and points the operator at the way across.
  const outgrewAlias = editingAlias && (chain.length > 0 || conditions.length > 0 || guardrails.length > 0);
  const pending = save.isPending || saveAlias.isPending;

  const submit = () => {
    if (!canSubmit || outgrewAlias) return;
    const scope = userId === null ? null : userId.trim();
    if (editingAlias) {
      saveAlias.mutate({ name: name.trim(), target: target.trim(), user_id: scope }, { onSuccess: onClose });
      return;
    }
    save.mutate({ name: name.trim(), spec, user_id: scope }, { onSuccess: onClose });
  };

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <Card.Content className="flex flex-col gap-5 p-5">
          <div className="text-sm font-semibold text-[var(--otari-ink)]">
            {editing ? (
              <>
                Edit {existing.kind === "alias" ? "alias" : "policy"} <code>{existing.name}</code>
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
            {editing ? (
              <div className="flex flex-col gap-1">
                <span className="text-sm font-medium text-[var(--otari-ink)]">Policy name</span>
                <code className="text-sm text-[var(--otari-muted)]">{existing.name}</code>
                <span className="text-xs text-[var(--otari-muted)]">
                  The name and who it applies to are the key and cannot be changed here. Delete and recreate to
                  change either.
                </span>
              </div>
            ) : (
              <Field
                label="Policy name"
                value={name}
                onChange={setName}
                placeholder="fast"
                isRequired
                autoFocus
                description={
                  nameHasDelimiter ? (
                    <span className="text-red-700">A policy name cannot contain “:” or “/”.</span>
                  ) : (
                    "What callers send as `model`."
                  )
                }
              />
            )}
            <ModelComboBox
              label="Serves"
              value={target}
              onChange={setTarget}
              isRequired
              description="The model that serves a normal request. Callers never see it."
            />
          </div>

          {editing ? null : <ScopePicker userId={userId} onChange={setUserId} />}

          {/* Conditional tier-down */}
          {conditions.length > 0 ? (
            <div className="flex flex-col gap-3 rounded-lg border border-[var(--otari-line)] p-3">
              <div>
                <span className="text-sm font-medium text-[var(--otari-ink)]">Instead, when the budget fills up</span>
                <p className="text-xs text-[var(--otari-muted)]">
                  Checked before the model above. A threshold must be under 100: the budget gate refuses a
                  request before selection once the cap is reached, so a rule at 100 could never fire.
                </p>
              </div>
              {conditions.map((condition, index) => (
                <div key={index} className="flex flex-wrap items-end gap-3">
                  <Field
                    label="Budget used at least (%)"
                    value={String(condition.threshold)}
                    onChange={(value) =>
                      setConditions((prev) =>
                        prev.map((c, i) => (i === index ? { ...c, threshold: Number(value) || 0 } : c)),
                      )
                    }
                    description={
                      condition.threshold >= 100 ? (
                        <span className="text-red-700">Must be under 100.</span>
                      ) : undefined
                    }
                  />
                  <div className="min-w-56 flex-1">
                    <ModelComboBox
                      label="Use instead"
                      value={condition.target}
                      onChange={(value) =>
                        setConditions((prev) => prev.map((c, i) => (i === index ? { ...c, target: value } : c)))
                      }
                      isRequired
                    />
                  </div>
                  <Button
                    variant="ghost"
                    onPress={() => setConditions((prev) => prev.filter((_, i) => i !== index))}
                  >
                    Remove
                  </Button>
                </div>
              ))}
            </div>
          ) : null}

          {/* Failure chain */}
          {chain.length > 0 ? (
            <div className="flex flex-col gap-3 rounded-lg border border-[var(--otari-line)] p-3">
              <div>
                <span className="text-sm font-medium text-[var(--otari-ink)]">If that fails, try</span>
                <p className="text-xs text-[var(--otari-muted)]">
                  Tried in order after a retryable failure. Not tried once tokens have started streaming, or
                  after a 400/401/403, which every provider would reject the same way.
                </p>
              </div>
              {chain.map((entry, index) => (
                <div key={index} className="flex flex-wrap items-end gap-3">
                  <div className="min-w-56 flex-1">
                    <ModelComboBox
                      label={`Fallback ${index + 1}`}
                      value={entry}
                      onChange={(value) => setChain((prev) => prev.map((e, i) => (i === index ? value : e)))}
                      isRequired
                    />
                  </div>
                  <Button variant="ghost" onPress={() => setChain((prev) => prev.filter((_, i) => i !== index))}>
                    Remove
                  </Button>
                </div>
              ))}
              <div>
                <button
                  type="button"
                  className="text-sm text-[var(--otari-brand)] hover:underline"
                  onClick={() => setChain((prev) => [...prev, ""])}
                >
                  + Another fallback
                </button>
              </div>
            </div>
          ) : null}

          {/* Guardrails */}
          {guardrails.length > 0 ? (
            <div className="flex flex-col gap-3 rounded-lg border border-[var(--otari-line)] p-3">
              <div>
                <span className="text-sm font-medium text-[var(--otari-ink)]">Always check</span>
                <p className="text-xs text-[var(--otari-muted)]">
                  Runs on every request through this policy. Callers can add their own guardrails but cannot
                  weaken these.
                </p>
                {guardrails_.configured ? null : (
                  <p className="mt-1 text-xs text-amber-700">
                    No guardrails service is configured, so these cannot run. With `if the service is down`
                    set to block, every request through this policy is refused until one is configured.{" "}
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
                        setGuardrails((prev) => prev.map((g, i) => (i === index ? { ...g, profile: value } : g)))
                      }
                      placeholder="prompt-injection"
                      isRequired
                      description="A profile configured on the guardrails service."
                    />
                    <ModeToggle
                      label="Mode"
                      value={guardrail.mode}
                      onChange={(mode) =>
                        setGuardrails((prev) => prev.map((g, i) => (i === index ? { ...g, mode } : g)))
                      }
                      hint="block rejects a flagged request; monitor records it and serves anyway."
                    />
                    <ModeToggle
                      label="If the service is down"
                      value={guardrail.on_unavailable ?? "block"}
                      onChange={(mode) =>
                        setGuardrails((prev) =>
                          prev.map((g, i) => (i === index ? { ...g, on_unavailable: mode } : g)),
                        )
                      }
                      hint="block fails closed, so a guardrails outage refuses every request through this policy."
                    />
                    <Button
                      variant="ghost"
                      onPress={() => setGuardrails((prev) => prev.filter((_, i) => i !== index))}
                    >
                      Remove
                    </Button>
                  </div>
                  {guardrail.mode === "block" && (guardrail.on_unavailable ?? "block") === "block" ? (
                    <div className="text-xs text-amber-700">
                      With both set to block, a guardrails-service outage rejects every request through this
                      policy, ahead of any fallback above.
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
                className="text-[var(--otari-brand)] hover:underline"
                onClick={() => setConditions([{ threshold: 80, target: "" }])}
              >
                + Tier down when the budget fills up
              </button>
            ) : null}
            {chain.length === 0 ? (
              <button
                type="button"
                className="text-[var(--otari-brand)] hover:underline"
                onClick={() => setChain([""])}
              >
                + Add a fallback chain
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
                  aria-describedby={guardrails_.configured ? undefined : "guardrails-unavailable"}
                  className={
                    guardrails_.configured
                      ? "text-[var(--otari-brand)] hover:underline"
                      : "cursor-not-allowed text-[var(--otari-muted)] opacity-60"
                  }
                  onClick={() => setGuardrails([{ profile: "", mode: "block", on_unavailable: "block" }])}
                >
                  + Add guardrails
                </button>
                {guardrails_.configured ? null : (
                  <span id="guardrails-unavailable" className="text-xs text-[var(--otari-muted)]">
                    No guardrails service is configured, so there would be nothing to call.{" "}
                    <Link to="/tools" className="text-[var(--otari-brand)] hover:underline">
                      Set one up in Tools &amp; Guardrails
                    </Link>
                    .
                  </span>
                )}
              </span>
            ) : null}
          </div>

          <div className="flex items-center gap-3">
            <Button variant="primary" isDisabled={!canSubmit || pending || outgrewAlias} onPress={submit}>
              {pending ? "Saving…" : editing ? "Save" : "Create policy"}
            </Button>
            <Button variant="ghost" onPress={onClose}>
              Cancel
            </Button>
            <span className="text-xs text-[var(--otari-muted)]">In effect for new requests within 30s.</span>
            {outgrewAlias ? (
              <span className="text-xs text-amber-700">
                An alias holds one target. To add a fallback, a condition, or a guardrail, delete this alias
                and create a policy with the same name.
              </span>
            ) : null}
          </div>
        </Card.Content>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function RoutingPage() {
  const policies = useRoutingPolicies();
  const aliases = useAliases();
  const deletePolicy = useDeleteRoutingPolicy();
  const deleteAlias = useDeleteAlias();
  // A "Make an alias" link from the Models page arrives as ?target=provider:model.
  const [searchParams] = useSearchParams();
  const initialTarget = searchParams.get("target") ?? "";
  const [adding, setAdding] = useState(initialTarget !== "");
  const [editing, setEditing] = useState<RoutingRow | null>(null);

  // Aliases and policies are listed together: an alias is the one-target case,
  // and this page is the only place either is managed.
  const rows: RoutingRow[] = [
    ...(policies.data ?? []).map((policy) => ({ ...policy, kind: "policy" as const })),
    ...(aliases.data ?? []).map(aliasAsRow),
  ].sort((a, b) => a.name.localeCompare(b.name) || (a.user_id ?? "").localeCompare(b.user_id ?? ""));

  const columns = useMemo<DataTableColumn<RoutingRow>[]>(
    () => [
      {
        id: "name",
        header: "Policy",
        isRowHeader: true,
        cell: (policy) => <CopyableValue value={policy.name} label="policy name" />,
      },
      {
        id: "serves",
        header: "Serves",
        cell: (policy) => (
          <div className="flex items-center gap-2">
            <span className="text-sm text-[var(--otari-ink)]">{servesSummary(policy)}</span>
            {policy.is_dynamic ? (
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
          const guardrails = policy.spec.guardrails ?? [];
          if (guardrails.length === 0) return <span className="text-[var(--otari-muted)]">–</span>;
          return (
            <span className="text-sm text-[var(--otari-ink)]">
              {guardrails.map((guardrail) => `${guardrail.profile} (${guardrail.mode})`).join(", ")}
            </span>
          );
        },
      },
      {
        id: "scope",
        header: "Applies to",
        cell: (policy) =>
          policy.user_id === null ? (
            <span className="text-[var(--otari-muted)]">Every caller</span>
          ) : (
            <CopyableValue value={policy.user_id} label="user id" />
          ),
      },
      {
        id: "source",
        header: "Source",
        cell: (row) => (
          <div className="flex items-center gap-1">
            <Chip size="sm" color={row.source === "config" ? "default" : "accent"}>
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
        cell: (policy) =>
          policy.source === "config" ? (
            <span className="text-xs text-[var(--otari-muted)]">set in config.yml</span>
          ) : (
            <div className="flex items-center justify-end gap-2">
              {isEditableInForm(policy.spec) ? (
                <Button
                  size="sm"
                  variant="ghost"
                  onPress={() => {
                    // The table stays mounted while the create form is open, so
                    // Edit is still reachable from it. Closing the other editor
                    // keeps this to one form: two stacked forms do not recover on
                    // their own, since each only closes when cancelled.
                    setAdding(false);
                    setEditing(policy);
                  }}
                >
                  Edit
                </Button>
              ) : (
                <span className="text-xs text-[var(--otari-muted)]">
                  Uses options this form cannot show yet. Edit it through the API so nothing is lost.
                </span>
              )}
              <ConfirmButton
                confirmLabel="Confirm"
                isPending={deletePolicy.isPending || deleteAlias.isPending}
                onConfirm={() =>
                  policy.kind === "alias"
                    ? deleteAlias.mutate({ name: policy.name, userId: policy.user_id })
                    : deletePolicy.mutate({ name: policy.name, userId: policy.user_id })
                }
              >
                Delete
              </ConfirmButton>
            </div>
          ),
      },
    ],
    [deletePolicy, deleteAlias],
  );

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Routing"
        description="Named models your callers send as `model`. A policy decides which real model serves each request, what is tried if that fails, and which guardrails always run."
        action={
          adding || editing !== null ? undefined : (
            <Button
              variant="primary"
              onPress={() => {
                setEditing(null);
                setAdding(true);
              }}
            >
              New policy
            </Button>
          )
        }
      />

      <ErrorBanner error={policies.error ?? aliases.error ?? deletePolicy.error ?? deleteAlias.error} />

      {adding ? (
        <PolicyForm existing={null} initialTarget={initialTarget} onClose={() => setAdding(false)} />
      ) : null}
      {editing !== null ? <PolicyForm existing={editing} onClose={() => setEditing(null)} /> : null}

      {rows.length === 0 && !policies.isLoading && !aliases.isLoading && !adding ? (
        <EmptyState title="No routing policies yet">
          <ol className="flex list-decimal flex-col gap-1 pl-5 text-sm text-[var(--otari-muted)]">
            <li>Create a policy and point it at the model that should normally serve.</li>
            <li>Add a fallback chain so a provider outage does not become a failed request.</li>
            <li>Have your callers send the policy name as their `model`.</li>
          </ol>
        </EmptyState>
      ) : (
        <DataTable
          ariaLabel="Routing policies"
          columns={columns}
          rows={rows}
          getRowKey={rowKeyOf}
          isLoading={policies.isLoading || aliases.isLoading}
          emptyContent="No routing policies yet."
        />
      )}
    </div>
  );
}
