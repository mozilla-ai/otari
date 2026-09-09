import { Button, Chip } from "@heroui/react"
import { useState } from "react"

import type { AlertRule, CreateAlertRuleRequest } from "@/client"
import {
  useAlertRules,
  useCreateAlertRule,
  useDeleteAlertRule,
  useTestAlertRule,
  useUpdateAlertRule,
} from "@/shared/api/alerts"
import { useOrganizationContext } from "@/shared/api/organizations"
import { ConfirmButton } from "@/shared/components/actions/ConfirmButton"
import {
  DataTable,
  type DataTableColumn,
} from "@/shared/components/data/DataTable"
import { ErrorBanner } from "@/shared/components/feedback/ErrorBanner"
import { InfoBanner } from "@/shared/components/feedback/InfoBanner"
import { Checkbox } from "@/shared/components/forms/Checkbox"
import { Field } from "@/shared/components/forms/Field"
import { SecretField } from "@/shared/components/forms/SecretField"
import { PageIntro } from "@/shared/components/layout/PageIntro"
import { Section } from "@/shared/components/layout/Section"
import { TableScrollFrame } from "@/shared/components/layout/TableScrollFrame"
import { FilterSelect } from "@/shared/components/navigation/FilterSelect"
import { formatRelative } from "@/shared/helpers/format"

import { canManage } from "./roles"

// Where this organization's budget alerts go.
//
// The page is built around one asymmetry: creating a rule is free and proves
// nothing, and what decides whether an alert ever arrives is whether the
// destination actually accepts a delivery. So "Send test" is a first-class
// action on every row. A destination that silently accepts nothing looks
// exactly like a budget that never crossed a threshold.
//
// The destination is write-only in both directions. The server stores it
// encrypted and returns only a redaction, because an Apprise URL carries its
// credentials in the path. The table shows the redaction, the create form takes
// the URL once through a `SecretField`, and the edit form never prefills it: an
// empty destination box means "keep the stored one", which is what the API's
// omit-to-keep contract expects.
//
// What a rule watches is not configurable, and that is deliberate: a rule covers
// every one of the organization's ceilings. Picking ceilings per rule would be a
// second scoping model on top of the one `budgets` already has.

/**
 * The warning thresholds offered, plus the "no warning" case.
 *
 * A picker rather than a free number field: the server accepts 1 to 99 and
 * nobody needs 63. `""` is the absent case, which the API takes as an explicit
 * null and reads as "alert only when a ceiling starts refusing requests".
 */
const WARN_THRESHOLD_OPTIONS = [
  { value: "50", label: "50% of the limit" },
  { value: "75", label: "75% of the limit" },
  { value: "80", label: "80% of the limit" },
  { value: "90", label: "90% of the limit" },
  { value: "95", label: "95% of the limit" },
  { value: "", label: "No early warning" },
]

/** A short, non-technical hint at what a redacted destination will reach. */
function destinationKind(destination: string): string {
  const scheme = destination.split(":")[0]?.toLowerCase() ?? ""
  if (scheme.startsWith("slack")) return "Slack"
  if (scheme.startsWith("discord")) return "Discord"
  if (scheme.startsWith("pagerduty") || scheme.startsWith("pagertree"))
    return "PagerDuty"
  if (scheme.startsWith("tgram") || scheme.startsWith("telegram"))
    return "Telegram"
  if (scheme.startsWith("mailto")) return "Email"
  if (scheme.startsWith("json") || scheme.startsWith("form")) return "Webhook"
  return "Custom"
}

/**
 * The create and edit form, which are the same fields over two verbs.
 *
 * Editing prefills everything except the destination, which cannot be prefilled
 * because the server never returns it. An empty box therefore means "keep the
 * stored one", and the PATCH omits the field entirely.
 */
function RuleForm({
  rule,
  onClose,
}: {
  rule?: AlertRule
  onClose: () => void
}) {
  const create = useCreateAlertRule()
  const update = useUpdateAlertRule()
  const [name, setName] = useState(rule?.name ?? "")
  const [destination, setDestination] = useState("")
  const [warnAt, setWarnAt] = useState(
    rule ? (rule.warn_at_percent?.toString() ?? "") : "80",
  )
  const [notifyOnExceeded, setNotifyOnExceeded] = useState(
    rule?.notify_on_exceeded ?? true,
  )

  const editing = rule !== undefined
  const mutation = editing ? update : create

  // A rule with neither threshold can never produce a message, which the server
  // refuses too. Caught here so the button is simply unavailable rather than
  // the operator submitting into a 422.
  const inert = warnAt === "" && !notifyOnExceeded
  const incomplete =
    name.trim() === "" || (!editing && destination.trim() === "")

  const submit = () => {
    const shared = {
      name: name.trim(),
      warn_at_percent: warnAt === "" ? null : Number(warnAt),
      notify_on_exceeded: notifyOnExceeded,
    }
    if (editing) {
      update.mutate(
        {
          ruleId: rule.id,
          // Omitted, not sent empty: the server reads an absent destination as
          // "keep the stored one".
          body: destination.trim()
            ? { ...shared, destination: destination.trim() }
            : shared,
        },
        { onSuccess: onClose },
      )
      return
    }
    const body: CreateAlertRuleRequest = {
      ...shared,
      destination: destination.trim(),
      enabled: true,
    }
    create.mutate(body, {
      onSuccess: () => {
        setName("")
        setDestination("")
        onClose()
      },
    })
  }

  return (
    <Section
      className="border-y border-border py-5"
      contentClassName="flex flex-col gap-4"
    >
      <h2 className="text-title">
        {editing ? `Edit ${rule.name}` : "Add an alert destination"}
      </h2>
      <ErrorBanner error={mutation.error} />
      <Field
        label="Name"
        value={name}
        onChange={setName}
        isRequired
        autoFocus
        placeholder="Platform team Slack"
        description="How this rule is identified in the alert itself, so whoever reads the message knows which rule sent it."
      />
      <SecretField
        label="Destination"
        value={destination}
        onChange={setDestination}
        placeholder="slack://token/channel"
        description={
          editing
            ? "Leave empty to keep the stored destination. Type a new Apprise URL to replace it."
            : "An Apprise URL. slack://token/channel, discord://webhook_id/webhook_token, pagerduty://key@apikey, mailto://user:password@smtp.example.com, or json://host/path for a plain webhook. Stored encrypted and never shown again."
        }
      />
      <FilterSelect
        label="Warn early at"
        value={warnAt}
        onChange={setWarnAt}
        options={WARN_THRESHOLD_OPTIONS}
      />
      <Checkbox isSelected={notifyOnExceeded} onChange={setNotifyOnExceeded}>
        Also alert when a budget starts refusing requests
      </Checkbox>
      {inert ? (
        <InfoBanner>
          With no early warning and no alert on refusal, this rule would never
          send anything. Pick a threshold, or leave the refusal alert on.
        </InfoBanner>
      ) : null}
      <p className="text-caption">
        The early warning is the useful one: by the time a budget starts
        refusing requests, the overspend has already happened.
      </p>
      <div className="flex gap-2">
        <Button
          variant="primary"
          isDisabled={incomplete || inert}
          isPending={mutation.isPending}
          onPress={submit}
        >
          {editing ? "Save changes" : "Add destination"}
        </Button>
        <Button variant="ghost" onPress={onClose}>
          Cancel
        </Button>
      </div>
    </Section>
  )
}

/**
 * The test action for one row, holding its own outcome.
 *
 * Per-row state rather than page state: two rows tested in a row must not show
 * each other's result, and the mutation is deliberately not invalidating the
 * list (a test writes nothing), so there is no refetch to hang the answer off.
 *
 * Three outcomes rather than two, and the third is the one that matters. If the
 * request itself fails (a network error, a 403, a 5xx) then `isSuccess` is
 * false, `data` is undefined, and a chip keyed on success alone renders nothing
 * once the spinner clears: the operator presses the button and gets silence,
 * which is exactly the state this button exists to rule out. The page-level
 * `ErrorBanner` does not cover it either, because this mutation belongs to one
 * row rather than to the page.
 *
 * "Test failed" and "Couldn't send test" are kept apart on purpose. The first
 * means the gateway reached the destination and the destination refused, which
 * is a bad Apprise URL. The second means the gateway never got that far, which
 * is a session, network or server problem. They need different fixes, so
 * collapsing them into one label would hide the difference.
 */
function TestRuleButton({ rule }: { rule: AlertRule }) {
  const test = useTestAlertRule()
  const result = test.data

  return (
    <div className="flex items-center justify-end gap-1.5">
      {test.isError ? (
        <Chip size="sm" color="danger">
          Couldn't send test
        </Chip>
      ) : test.isSuccess ? (
        <Chip size="sm" color={result?.delivered ? "accent" : "warning"}>
          {result?.delivered ? "Test delivered" : "Test failed"}
        </Chip>
      ) : null}
      <Button
        size="sm"
        variant="ghost"
        isPending={test.isPending}
        onPress={() => test.mutate(rule.id)}
      >
        Send test
      </Button>
    </div>
  )
}

export function OrganizationAlertsPage() {
  const context = useOrganizationContext()
  const canEdit = canManage(context.data)
  const rules = useAlertRules(canEdit)
  const update = useUpdateAlertRule()
  const remove = useDeleteAlertRule()
  const [adding, setAdding] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)

  const rows = rules.data?.data ?? []
  const editingRule = rows.find((row) => row.id === editingId)

  const columns: DataTableColumn<AlertRule>[] = [
    {
      id: "name",
      header: "Name",
      isRowHeader: true,
      cell: (row) => <span className="font-medium">{row.name}</span>,
    },
    {
      id: "destination",
      header: "Destination",
      cell: (row) => (
        <div className="flex flex-col">
          <span>{destinationKind(row.destination)}</span>
          {/* The redaction, not the URL: scheme and host only. Shown small
              because it is for recognizing the row, not for reading. */}
          <span className="text-caption break-all">{row.destination}</span>
        </div>
      ),
    },
    {
      id: "thresholds",
      header: "Alerts on",
      cell: (row) => (
        <div className="flex flex-wrap gap-1.5">
          {row.warn_at_percent != null ? (
            <Chip size="sm" color="default">{`${row.warn_at_percent}%`}</Chip>
          ) : null}
          {row.notify_on_exceeded ? (
            <Chip size="sm" color="default">
              Limit reached
            </Chip>
          ) : null}
        </div>
      ),
    },
    {
      id: "status",
      header: "Status",
      cell: (row) =>
        row.enabled ? (
          <Chip size="sm" color="accent">
            Active
          </Chip>
        ) : (
          <Chip size="sm" color="default">
            Paused
          </Chip>
        ),
    },
    {
      id: "added",
      header: "Added",
      cell: (row) => formatRelative(row.created_at),
    },
  ]

  if (canEdit) {
    columns.push({
      id: "test",
      header: "Test",
      align: "end",
      cell: (row) => <TestRuleButton rule={row} />,
    })
    columns.push({
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (row) => (
        <div className="flex items-center justify-end gap-1.5">
          <Button
            size="sm"
            variant="ghost"
            onPress={() => {
              setAdding(false)
              setEditingId(row.id)
            }}
          >
            Edit
          </Button>
          <Button
            size="sm"
            variant="ghost"
            isDisabled={update.isPending}
            onPress={() =>
              update.mutate({
                ruleId: row.id,
                body: { enabled: !row.enabled },
              })
            }
          >
            {row.enabled ? "Pause" : "Resume"}
          </Button>
          <ConfirmButton
            confirmLabel="Remove rule"
            isPending={remove.isPending}
            onConfirm={() => remove.mutate(row.id)}
          >
            Remove
          </ConfirmButton>
        </div>
      ),
    })
  }

  return (
    <div className="flex flex-col">
      <PageIntro
        title="Budget alerts"
        action={
          canEdit && !adding && editingId === null ? (
            <Button variant="primary" onPress={() => setAdding(true)}>
              Add destination
            </Button>
          ) : null
        }
      >
        Get told when this organization's budgets are running out, instead of
        finding out when requests start being refused. Each destination is
        checked against every budget ceiling the organization owns, and an alert
        is sent once per budget period.
      </PageIntro>

      <ErrorBanner
        error={context.error ?? rules.error ?? update.error ?? remove.error}
      />

      {/* Held back until the context has answered, so an admin is not told for
          one paint that they may not be here. */}
      {context.data && !canEdit ? (
        <InfoBanner>
          Only organization owners and admins can manage budget alerts.
        </InfoBanner>
      ) : null}

      {adding ? <RuleForm onClose={() => setAdding(false)} /> : null}
      {editingRule ? (
        <RuleForm
          key={editingRule.id}
          rule={editingRule}
          onClose={() => setEditingId(null)}
        />
      ) : null}

      {canEdit || context.isPending ? (
        <TableScrollFrame className="otari-alert-rules-table">
          <DataTable
            ariaLabel="Budget alert rules"
            columns={columns}
            rows={rows}
            getRowKey={(row) => row.id}
            isLoading={context.isPending || rules.isLoading}
            emptyContent="No alert destinations yet. Add one so a budget running out reaches you instead of waiting to be noticed."
          />
        </TableScrollFrame>
      ) : null}
    </div>
  )
}
