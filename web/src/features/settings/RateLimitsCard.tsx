import { useState } from "react"
import type { RateLimitRule } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import {
  useCreateRateLimitRule,
  useDeleteRateLimitRule,
  useRateLimitRules,
  useUpdateRateLimitRule,
} from "@/shared/api/rateLimits"
import { docsSourceHref } from "@/shared/helpers/docs"
import { formatNumber } from "@/shared/helpers/format"

type Scope = RateLimitRule["per"]

const SCOPES: { value: Scope; label: string }[] = [
  { value: "key", label: "Each API key" },
  { value: "user", label: "Each user" },
  { value: "deployment", label: "The whole deployment" },
]

const SCOPE_LABEL: Record<Scope, string> = {
  key: "per key",
  user: "per user",
  deployment: "deployment",
}

// The pattern the gateway holds a name to, since it is part of the counter's key.
const NAME_PATTERN = /^[A-Za-z0-9_.-]+$/
const DEFAULT_LEASE_SEC = 900

/** The lanes every line shares, so the columns read down. */
const NAME_LANE = "w-full shrink-0 font-mono text-xs md:w-[9rem]"
const SCOPE_LANE = "w-full shrink-0 text-caption text-subtle md:w-[5.5rem]"

function limitsSummary(rule: RateLimitRule): string {
  const parts = [
    rule.rpm ? `${formatNumber(rule.rpm)} requests/min` : "",
    rule.tpm ? `${formatNumber(rule.tpm)} tokens/min` : "",
    rule.max_concurrent ? `${formatNumber(rule.max_concurrent)} in flight` : "",
  ]
  return parts.filter((part) => part !== "").join(" · ")
}

/** A whole number above zero, `""` for a blank box, or `undefined` when the text is neither. */
function parseLimit(text: string): number | "" | undefined {
  const trimmed = text.trim()
  if (trimmed === "") return ""
  if (!/^\d+$/.test(trimmed)) return undefined
  const value = Number(trimmed)
  return value >= 1 ? value : undefined
}

function asText(value: number | null | undefined): string {
  return value ? String(value) : ""
}

interface RuleDraft {
  name: string
  per: Scope
  rpm: string
  tpm: string
  maxConcurrent: string
  leaseSec: string
}

function draftOf(rule: RateLimitRule | undefined): RuleDraft {
  return {
    name: rule?.name ?? "",
    per: rule?.per ?? "key",
    rpm: asText(rule?.rpm),
    tpm: asText(rule?.tpm),
    maxConcurrent: asText(rule?.max_concurrent),
    leaseSec: String(rule?.lease_sec ?? DEFAULT_LEASE_SEC),
  }
}

function RuleDialog({
  rule,
  isOpen,
  onClose,
}: {
  /** The stored rule to edit, or `undefined` to add one. */
  rule: RateLimitRule | undefined
  isOpen: boolean
  onClose: () => void
}) {
  const create = useCreateRateLimitRule()
  const update = useUpdateRateLimitRule()
  const [draft, setDraft] = useState(() => draftOf(rule))
  const { isDirty } = useDirtySnapshot(draft)
  const isEdit = rule !== undefined
  const mutation = isEdit ? update : create

  const set = (patch: Partial<RuleDraft>) =>
    setDraft((current) => ({ ...current, ...patch }))

  const rpm = parseLimit(draft.rpm)
  const tpm = parseLimit(draft.tpm)
  const maxConcurrent = parseLimit(draft.maxConcurrent)
  const leaseSec = Number(draft.leaseSec)
  const nameError =
    draft.name !== "" && !NAME_PATTERN.test(draft.name)
      ? "Letters, digits, dots, dashes and underscores only."
      : ""
  const leaseError =
    maxConcurrent !== "" && !(leaseSec > 0)
      ? "A number of seconds above zero."
      : ""
  const hasLimit = [rpm, tpm, maxConcurrent].some(
    (value) => typeof value === "number",
  )
  const isReady =
    draft.name.trim() !== "" &&
    nameError === "" &&
    leaseError === "" &&
    rpm !== undefined &&
    tpm !== undefined &&
    maxConcurrent !== undefined &&
    hasLimit

  const submit = () => {
    const limits = {
      per: draft.per,
      rpm: rpm === "" || rpm === undefined ? null : rpm,
      tpm: tpm === "" || tpm === undefined ? null : tpm,
      max_concurrent:
        maxConcurrent === "" || maxConcurrent === undefined
          ? null
          : maxConcurrent,
      lease_sec: maxConcurrent === "" ? DEFAULT_LEASE_SEC : leaseSec,
    }
    if (rule) {
      update.mutate({ name: rule.name, body: limits }, { onSuccess: onClose })
    } else {
      create.mutate(
        { name: draft.name.trim(), ...limits },
        { onSuccess: onClose },
      )
    }
  }

  const limitError = (value: number | "" | undefined) =>
    value === undefined ? "A whole number above zero." : ""

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title={rule ? `Edit ${rule.name}` : "New rate limit rule"}
      description="Set any of the three limits; leave a box blank for no limit of that kind."
      submitLabel={isEdit ? "Save rule" : "Add rule"}
      onSubmit={submit}
      isPending={mutation.isPending}
      isSubmitDisabled={!isReady}
      isDirty={isDirty}
      error={mutation.error}
    >
      <Field
        label="Name"
        value={draft.name}
        onChange={(name) => set({ name })}
        isRequired
        isDisabled={isEdit}
        autoFocus={!isEdit}
        placeholder="keys"
        isInvalid={nameError !== ""}
        errorMessage={nameError}
        description="Named in the 429 a refused request gets. Fixed once the rule exists."
      />
      <Select
        label="Counted for"
        value={draft.per}
        onChange={(per) => set({ per: per as Scope })}
        options={SCOPES}
        description="Each API key and each user get their own count; the whole deployment shares one."
        shouldReserveMessage={false}
      />
      <Field
        label="Requests per minute"
        value={draft.rpm}
        onChange={(value) => set({ rpm: value })}
        placeholder="no limit"
        isInvalid={rpm === undefined}
        errorMessage={limitError(rpm)}
      />
      <Field
        label="Tokens per minute"
        value={draft.tpm}
        onChange={(value) => set({ tpm: value })}
        placeholder="no limit"
        isInvalid={tpm === undefined}
        errorMessage={limitError(tpm)}
        description="A request is admitted on its estimate and charged the tokens it used."
      />
      <Field
        label="Requests in flight"
        value={draft.maxConcurrent}
        onChange={(value) => set({ maxConcurrent: value })}
        placeholder="no limit"
        isInvalid={maxConcurrent === undefined}
        errorMessage={limitError(maxConcurrent)}
      />
      {maxConcurrent !== "" ? (
        <Field
          label="Slot lease (seconds)"
          value={draft.leaseSec}
          onChange={(value) => set({ leaseSec: value })}
          isInvalid={leaseError !== ""}
          errorMessage={leaseError}
          description="The longest a slot stays taken if a gateway process dies holding it."
        />
      ) : null}
      {!hasLimit && rpm !== undefined && tpm !== undefined ? (
        <p className="text-caption text-subtle">Set at least one limit.</p>
      ) : null}
    </FormDialog>
  )
}

function RuleLine({
  rule,
  onEdit,
}: {
  rule: RateLimitRule
  onEdit: (rule: RateLimitRule) => void
}) {
  const remove = useDeleteRateLimitRule()
  const [isDeleteOpen, setDeleteOpen] = useState(false)
  const isStored = rule.source === "dashboard"

  return (
    <div className="flex flex-col gap-2 py-4 md:flex-row md:flex-wrap md:items-center">
      <code className={NAME_LANE}>{rule.name}</code>
      <span className={SCOPE_LANE}>{SCOPE_LABEL[rule.per]}</span>
      <span className="min-w-0 flex-1 text-caption tabular-nums">
        {limitsSummary(rule)}
      </span>
      {isStored ? (
        <div className="flex shrink-0 gap-2">
          <Button
            variant="ghost"
            aria-label={`Edit ${rule.name}`}
            onPress={() => onEdit(rule)}
          >
            Edit
          </Button>
          <Button
            variant="ghost"
            aria-label={`Remove ${rule.name}`}
            isDisabled={remove.isPending}
            onPress={() => setDeleteOpen(true)}
          >
            Remove
          </Button>
        </div>
      ) : (
        <span className="shrink-0 text-mono-overline text-subtle">
          config.yml
        </span>
      )}
      <ConfirmDialog
        isOpen={isDeleteOpen}
        // Cleared on the way out, so a refusal does not greet the next open.
        onOpenChange={(open) => {
          setDeleteOpen(open)
          if (!open) remove.reset()
        }}
        heading="Remove rate limit rule"
        body={`${rule.name} stops limiting requests on this replica at once, and on every replica within 30 seconds.`}
        confirmLabel="Remove rule"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          remove.mutate(rule.name, { onSuccess: () => setDeleteOpen(false) })
        }}
      />
    </div>
  )
}

/**
 * The deployment's ``rate_limits`` rules. Stored rules are added and edited
 * here; config.yml rules are listed read-only, since that file is where they
 * change.
 */
export function RateLimitsCard() {
  const rules = useRateLimitRules()
  const [editing, setEditing] = useState<RateLimitRule | undefined>()
  const [isDialogOpen, setDialogOpen] = useState(false)
  const [openCount, setOpenCount] = useState(0)

  const all = rules.data?.rules ?? []
  const open = (rule: RateLimitRule | undefined) => {
    setEditing(rule)
    setOpenCount((count) => count + 1)
    setDialogOpen(true)
  }

  return (
    <>
      {/* Outside the group, whose rows are a divide-y container; keyed on the
          open count so each open starts from the rule as it is now. */}
      <RuleDialog
        key={openCount}
        rule={editing}
        isOpen={isDialogOpen}
        onClose={() => setDialogOpen(false)}
      />
      <SettingsGroup
        title="Rate limit rules"
        count={rules.data ? all.length : undefined}
        description="Requests per minute, tokens per minute and requests in flight, per API key, per user or for the whole deployment. A change applies on this replica at once and on every replica within 30 seconds."
        docsHref={docsSourceHref("configuration.md", "rate-limit-rules")}
        action={
          <Button variant="primary" onPress={() => open(undefined)}>
            Add rule
          </Button>
        }
      >
        {rules.error ? (
          <div className="py-4">
            <ErrorBanner error={rules.error} />
          </div>
        ) : null}
        {rules.data && all.length === 0 ? (
          <p className="py-4 text-caption text-subtle">
            No rules, so requests are limited only by rate_limit_rpm.
          </p>
        ) : null}
        {all.map((rule) => (
          <RuleLine key={rule.name} rule={rule} onEdit={open} />
        ))}
      </SettingsGroup>
    </>
  )
}
