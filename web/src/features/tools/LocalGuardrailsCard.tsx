import { useState } from "react"

import type {
  BuiltInGuardrailSpec,
  ConfigGuardrail,
  StoredGuardrail,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Toggle } from "@/design-system/forms/Toggle"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { DisclosureRow } from "@/design-system/navigation/DisclosureRow"
import { taskLabel } from "@/features/tools/guardrailTasks"
import { LocalGuardrailDialog } from "@/features/tools/LocalGuardrailDialog"
import { LocalGuardrailTestDialog } from "@/features/tools/LocalGuardrailTestDialog"
import {
  useBuiltInGuardrails,
  useDeleteGuardrail,
  useGuardrailCredentials,
  useUpdateGuardrail,
} from "@/shared/api/tools"

// Guardrails Otari builds and runs in its own process, rather than sending to a
// sidecar. A row here is what a caller's `profile` field names, so this card and
// the organization mandates below it are two halves of one thing: this defines
// a guardrail, that one decides who it runs for.
//
// A drill-in rather than a fifth band of rows, for the reason the search-tools
// card above is one: it is a list of things rather than a set of settings, and
// it is empty on most deployments.

/** The lanes every line shares, so the columns read down. */
const NAME_LANE = "w-full shrink-0 font-mono text-xs md:w-[9rem]"
const GUARDRAIL_LANE = "w-full shrink-0 text-caption md:w-[10rem]"

function guardrailCount(count: number): string {
  return `${count} ${count === 1 ? "guardrail" : "guardrails"}`
}

function StoredGuardrailLine({
  guardrail,
  guardrails,
  onEdit,
  onTest,
}: {
  guardrail: StoredGuardrail
  guardrails: readonly BuiltInGuardrailSpec[]
  onEdit: () => void
  onTest: () => void
}) {
  const update = useUpdateGuardrail()
  const remove = useDeleteGuardrail()
  const [isDeleteOpen, setDeleteOpen] = useState(false)

  // Named in the words the form asked in. A row whose class this build no longer
  // describes still reads: it falls back to the wire name rather than blank.
  const spec = guardrails.find(
    (entry) => entry.guardrail_name === guardrail.guardrail_name,
  )
  const busy = remove.isPending || update.isPending

  return (
    <div className="flex flex-col gap-2 px-4 py-3">
      <div className="flex flex-col gap-2 md:flex-row md:flex-wrap md:items-center">
        <code className={NAME_LANE}>{guardrail.name}</code>
        <span className={GUARDRAIL_LANE}>
          {spec?.display_name ?? guardrail.guardrail_name}
        </span>
        <span className="min-w-0 flex-1 text-caption text-subtle">
          {spec ? taskLabel(spec.primary_category) : "Not in this build"}
        </span>
        <Toggle
          isSelected={guardrail.enabled}
          isDisabled={busy}
          label={`Run ${guardrail.name}`}
          onChange={(next) =>
            update.mutate({
              name: guardrail.name,
              body: {
                enabled: next,
                expected_updated_at: guardrail.updated_at,
              },
            })
          }
        />
        {/* Named per row, as the search-tools rows are: the card is a list, so a
            bare "Edit" is the same name on every one of them. */}
        <Button
          variant="ghost"
          aria-label={`Edit ${guardrail.name}`}
          isDisabled={busy}
          onPress={onEdit}
        >
          Edit
        </Button>
        <Button
          variant="ghost"
          aria-label={`Test ${guardrail.name}`}
          isDisabled={busy}
          onPress={onTest}
        >
          Test
        </Button>
        <Button
          variant="ghost"
          aria-label={`Remove ${guardrail.name}`}
          isDisabled={busy}
          onPress={() => setDeleteOpen(true)}
        >
          Remove
        </Button>
      </div>
      {guardrail.decryptable ? null : (
        <p className="text-caption text-warning">
          Secrets unreadable: check OTARI_SECRET_KEY
        </p>
      )}
      {guardrail.shadows_config ? (
        <p className="text-caption text-warning">
          Overrides the config-file guardrail of this name
        </p>
      ) : null}
      <ErrorBanner error={update.error} />

      <ConfirmDialog
        isOpen={isDeleteOpen}
        // Cleared on the way out: a refusal otherwise sits on the mutation and
        // greets the next open as if it had just happened.
        onOpenChange={(open) => {
          setDeleteOpen(open)
          if (!open) remove.reset()
        }}
        heading="Remove guardrail"
        body={`${guardrail.name} and the credentials stored with it are removed. A request that still names it as a profile is refused, so update the callers that use it.`}
        confirmLabel="Remove permanently"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          remove.mutate(guardrail.name, {
            onSuccess: () => setDeleteOpen(false),
          })
        }}
      />
    </div>
  )
}

// A config-file guardrail is a fact, not a control: it is editable only where
// the file is defined, so it reads as a line of values.
function ConfigGuardrailLine({ guardrail }: { guardrail: ConfigGuardrail }) {
  return (
    <div className="flex flex-col gap-2 px-4 py-3 md:flex-row md:flex-wrap md:items-center">
      <code className={NAME_LANE}>{guardrail.name}</code>
      <span className={GUARDRAIL_LANE}>{guardrail.guardrail_name}</span>
      <span className="min-w-0 flex-1 text-caption text-subtle">
        config file, editable where it is defined
      </span>
      <span className="shrink-0 text-mono-overline text-subtle">
        {guardrail.enabled ? "On" : "Off"}
      </span>
      {guardrail.shadowed ? (
        <p className="text-caption text-warning">
          Overridden by the stored guardrail of this name
        </p>
      ) : null}
    </div>
  )
}

/**
 * The guardrails this gateway runs itself, as a row that drills in.
 *
 * Sits under the guardrail service settings because it is the alternative to
 * them: a guardrail defined here needs no sidecar, and the profile it publishes
 * is the same field an organization mandate names.
 */
export function LocalGuardrailsCard({ docsHref }: { docsHref: string }) {
  const catalog = useBuiltInGuardrails()
  const stored = useGuardrailCredentials()
  const [isOpen, setIsOpen] = useState(false)
  const [adding, setAdding] = useState(false)
  const [openCount, setOpenCount] = useState(0)
  const [editing, setEditing] = useState<StoredGuardrail | undefined>()
  const [testing, setTesting] = useState<StoredGuardrail | undefined>()

  const known = catalog.data?.guardrails ?? []
  const rows = stored.data?.stored ?? []
  const fromConfig = stored.data?.config ?? []
  const count = rows.length + fromConfig.length
  // Nothing is known until the read answers, and "0 guardrails" is a claim
  // rather than a placeholder. A failed read is not an empty deployment either:
  // `isLoading` goes false with no data behind it.
  const failed = Boolean(stored.error)
  const answered = !stored.isLoading && !failed

  // The row's own copy is what the dialog seeds from, so it has to be the fresh
  // one after a save rather than the object the button was pressed with.
  const editingRow = rows.find((row) => row.name === editing?.name)
  const testingRow = rows.find((row) => row.name === testing?.name)

  return (
    <>
      {/* Outside the group, not inside it: `FormDialog` renders a real element,
          and a group's rows are a `divide-y` container where one more child
          changes which row is last. */}
      {/* Keyed on the open count, so each open remounts a blank form. Clearing
          the draft on close instead would blank the fields while the dialog is
          still animating away. */}
      {adding ? (
        <LocalGuardrailDialog
          key={openCount}
          isOpen={adding}
          onClose={() => setAdding(false)}
          guardrails={known}
          takenNames={[
            ...rows.map((row) => row.name),
            ...fromConfig.map((row) => row.name),
          ]}
        />
      ) : null}
      {editingRow ? (
        // Keyed on the row, so one row's draft cannot reach another's dialog.
        <LocalGuardrailDialog
          key={editingRow.name}
          isOpen
          onClose={() => setEditing(undefined)}
          guardrails={known}
          takenNames={[]}
          editing={editingRow}
        />
      ) : null}
      {testingRow ? (
        <LocalGuardrailTestDialog
          key={testingRow.name}
          isOpen
          onClose={() => setTesting(undefined)}
          guardrail={testingRow}
        />
      ) : null}
      <SettingsGroup
        bounded
        title="Guardrails Otari runs itself"
        description="Defined here and run in this process, with no separate guardrails service. A caller names one in the same profile field."
        docsHref={docsHref}
        action={
          <Button
            variant="primary"
            isDisabled={known.length === 0}
            onPress={() => {
              setOpenCount((current) => current + 1)
              setAdding(true)
            }}
          >
            Add guardrail
          </Button>
        }
      >
        {stored.error || catalog.error ? (
          // Outside the disclosure: a read that failed is the thing the operator
          // most needs to see, and the row is collapsed by default.
          <div className="px-4 py-3">
            <ErrorBanner error={stored.error ?? catalog.error} />
          </div>
        ) : null}
        <DisclosureRow
          label="Configure local guardrails"
          help={
            failed
              ? "Could not read the guardrails this deployment defines."
              : !answered
                ? "Reading the guardrails this deployment defines."
                : count === 0
                  ? "None defined. Add one to check traffic without running a second service."
                  : "Callers name one in a guardrail entry's profile field."
          }
          isOpen={isOpen}
          onToggle={() => setIsOpen((open) => !open)}
          trailing={
            <span className="text-caption text-subtle tabular-nums">
              {answered ? guardrailCount(count) : ""}
            </span>
          }
        >
          <div className="flex flex-col divide-y divide-border-subtle">
            {rows.map((row) => (
              <StoredGuardrailLine
                key={row.name}
                guardrail={row}
                guardrails={known}
                onEdit={() => setEditing(row)}
                onTest={() => setTesting(row)}
              />
            ))}
            {fromConfig.map((row) => (
              <ConfigGuardrailLine key={row.name} guardrail={row} />
            ))}
          </div>
        </DisclosureRow>
      </SettingsGroup>
    </>
  )
}
