import { Link } from "@tanstack/react-router"
import { useState } from "react"
import type {
  HealthCheckProvider,
  HealthCheckResult,
  HealthCheckSettings,
  HealthCheckStatus,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmButton } from "@/design-system/actions/ConfirmButton"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import {
  PAGE_SIZE_OPTIONS,
  TablePagination,
} from "@/design-system/data/TablePagination"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Field } from "@/design-system/forms/Field"
import { TextArea } from "@/design-system/forms/TextArea"
import { Toggle } from "@/design-system/forms/Toggle"
import { Dot } from "@/design-system/indicators/Dot"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import {
  useDisableFailingModels,
  useHealthCheckStatus,
  useRunHealthChecks,
  useUpdateHealthCheckSettings,
} from "@/shared/api/healthChecks"
import { looksLikeAddress, parseAddresses } from "@/shared/helpers/addresses"
import { formatDateTimeMinutes } from "@/shared/helpers/format"
import { useDeployment } from "@/shared/hooks/useDeployment"

// The server's cap on the alert list.
const MAX_ALERT_EMAILS = 20

const STATUS_LABELS = {
  ok: "OK",
  degraded: "Some failing",
  down: "Down",
} as const

function ProviderStatus({ status }: { status: HealthCheckProvider["status"] }) {
  return (
    <span
      className={`flex items-center gap-2 text-mono-caption ${status === "ok" ? "text-muted" : "text-danger"}`}
    >
      <Dot className={status === "ok" ? "bg-success" : "bg-danger"} />
      {STATUS_LABELS[status]}
    </span>
  )
}

// One bar per run in the last 24 hours, its height the share that answered.
function History({ provider }: { provider: HealthCheckProvider }) {
  const points = provider.history
  if (points.length < 2) {
    return <span className="text-caption">—</span>
  }
  const healthy = points.filter((point) => point.answered === point.total)
  return (
    <span
      role="img"
      aria-label={`${healthy.length} of ${points.length} runs in the last 24 hours had every model answering`}
      className="flex h-4 items-end gap-px"
    >
      {points.map((point) => (
        <span
          key={point.started_at}
          title={`${formatDateTimeMinutes(point.started_at)}: ${point.answered} of ${point.total}`}
          className={`w-1 ${point.answered === point.total ? "bg-success" : "bg-danger"}`}
          style={{
            height: `${Math.max(15, (point.answered / Math.max(point.total, 1)) * 100)}%`,
          }}
        />
      ))}
    </span>
  )
}

const PROVIDER_COLUMNS: DataTableColumn<HealthCheckProvider>[] = [
  {
    id: "provider",
    header: "Provider",
    isRowHeader: true,
    cell: (row) => <span className="text-body">{row.provider}</span>,
  },
  {
    id: "status",
    header: "Status",
    cell: (row) => <ProviderStatus status={row.status} />,
  },
  {
    id: "answered",
    header: "Answered",
    align: "end",
    cell: (row) => (
      <span className="text-mono-caption">
        {row.answered}/{row.total}
      </span>
    ),
  },
  {
    id: "history",
    header: "Last 24h",
    cell: (row) => <History provider={row} />,
  },
  {
    id: "error",
    header: "Most common error",
    cell: (row) =>
      row.top_error ? (
        <span className="line-clamp-2 text-caption" title={row.top_error}>
          {row.top_error_count}× {row.top_error}
        </span>
      ) : null,
  },
]

const RESULT_COLUMNS: DataTableColumn<HealthCheckResult>[] = [
  {
    id: "selector",
    header: "Model",
    isRowHeader: true,
    cell: (row) => <span className="text-mono-caption">{row.selector}</span>,
  },
  {
    id: "status",
    header: "Status",
    cell: (row) => (
      <span
        className={`flex items-center gap-2 text-mono-caption ${row.status === "fail" ? "text-danger" : "text-muted"}`}
      >
        <Dot
          className={
            row.status === "pass"
              ? "bg-success"
              : row.status === "fail"
                ? "bg-danger"
                : "bg-text-subtle"
          }
        />
        {row.status === "fail" && row.status_code
          ? `HTTP ${row.status_code}`
          : row.status === "pass"
            ? "Answered"
            : row.status === "fail"
              ? "Failed"
              : "Skipped"}
      </span>
    ),
  },
  {
    id: "since",
    header: "Failing since",
    cell: (row) =>
      row.failing_since ? (
        <span className="text-mono-caption">
          {formatDateTimeMinutes(row.failing_since)}
          {row.failed_runs > 1 ? ` · ${row.failed_runs} runs` : ""}
        </span>
      ) : null,
  },
  {
    id: "latency",
    header: "Time",
    align: "end",
    cell: (row) =>
      row.status === "skip" ? null : (
        <span className="text-mono-caption">{row.latency_ms} ms</span>
      ),
  },
  {
    id: "error",
    header: "Error",
    cell: (row) =>
      row.error ? (
        <span className="line-clamp-2 text-caption" title={row.error}>
          {row.error}
        </span>
      ) : null,
  },
]

function RunSummary({ status }: { status: HealthCheckStatus }) {
  const { last_run: last, running, settings } = status
  const next = settings.schedule_enabled
    ? settings.next_run_at
      ? `Next run ${formatDateTimeMinutes(settings.next_run_at)}.`
      : "Next run within a minute."
    : "The schedule is off."
  return (
    <p className="text-sm text-muted" role="status" aria-live="polite">
      {running
        ? `Running since ${formatDateTimeMinutes(running.started_at)}. `
        : settings.run_requested
          ? "Starting a run. "
          : null}
      {last
        ? `Last run ${formatDateTimeMinutes(last.started_at)}: ${last.passed} of ${last.total} answered${last.included_reasoning ? ", reasoning models included" : ""}. `
        : "No run has finished yet. "}
      {next}
    </p>
  )
}

// The server's bounds on each count, so a value it would refuse, or one a
// lenient parse would quietly shorten ("30m"), is refused here before a save.
const COUNT_RANGES = {
  interval: [15, 1440],
  maxTokens: [1, 4096],
  reasoningMaxTokens: [1, 8192],
  reasoningEvery: [0, 48],
} as const

type CountField = keyof typeof COUNT_RANGES

function countProblem(field: CountField, value: string): string | undefined {
  const [min, max] = COUNT_RANGES[field]
  const trimmed = value.trim()
  if (!/^\d+$/.test(trimmed)) return "Enter a whole number."
  const count = Number(trimmed)
  return count < min || count > max ? `Between ${min} and ${max}.` : undefined
}

function SettingsCard({ settings }: { settings: HealthCheckSettings }) {
  const update = useUpdateHealthCheckSettings()
  const [draft, setDraft] = useState({
    interval: String(settings.interval_minutes),
    prompt: settings.prompt,
    maxTokens: String(settings.max_tokens),
    reasoningMaxTokens: String(settings.reasoning_max_tokens),
    reasoningEvery: String(settings.reasoning_every_runs),
  })
  // The draft the last successful save sent. "Saved" shows only while the form
  // still holds it, so an edit made since, even during the save, does not read
  // as saved, and the save in flight keeps Save disabled until it settles.
  const [saved, setSaved] = useState<typeof draft | null>(null)
  const set = (field: keyof typeof draft) => (value: string) =>
    setDraft((current) => ({ ...current, [field]: value }))
  const showSaved =
    saved !== null &&
    (Object.keys(draft) as (keyof typeof draft)[]).every(
      (field) => draft[field] === saved[field],
    )
  const problems = {
    interval: countProblem("interval", draft.interval),
    maxTokens: countProblem("maxTokens", draft.maxTokens),
    reasoningMaxTokens: countProblem(
      "reasoningMaxTokens",
      draft.reasoningMaxTokens,
    ),
    reasoningEvery: countProblem("reasoningEvery", draft.reasoningEvery),
    prompt: draft.prompt.trim() === "" ? "Enter a prompt." : undefined,
  }
  const invalid = Object.values(problems).some(
    (problem) => problem !== undefined,
  )

  return (
    <SettingsGroup
      title="Schedule"
      description="Checks use this deployment's own provider keys. They write no usage and are billed to no organization, but your providers charge for every request."
    >
      <div className="flex flex-col gap-4 py-4">
        <ErrorBanner error={update.error} />
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-body">Run the checks on a schedule</p>
            <p className="mt-1 max-w-3xl text-caption">
              Every chat model each provider serves is sent the prompt below. A
              check passes on any answer, whatever it says.
            </p>
          </div>
          <Toggle
            isSelected={settings.schedule_enabled}
            onChange={(enabled) => update.mutate({ schedule_enabled: enabled })}
            label="Run the checks on a schedule"
            isDisabled={update.isPending}
          />
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            label="Minutes between runs"
            value={draft.interval}
            onChange={set("interval")}
            description="15 to 1440."
            isInvalid={problems.interval !== undefined}
            errorMessage={problems.interval}
          />
          <Field
            label="Include reasoning models every Nth run"
            value={draft.reasoningEvery}
            onChange={set("reasoningEvery")}
            description="They cost the most. 0 leaves them out."
            isInvalid={problems.reasoningEvery !== undefined}
            errorMessage={problems.reasoningEvery}
          />
          <Field
            label="Max output tokens"
            value={draft.maxTokens}
            onChange={set("maxTokens")}
            isInvalid={problems.maxTokens !== undefined}
            errorMessage={problems.maxTokens}
          />
          <Field
            label="Max output tokens for reasoning models"
            value={draft.reasoningMaxTokens}
            onChange={set("reasoningMaxTokens")}
            description="They spend tokens thinking before they answer."
            isInvalid={problems.reasoningMaxTokens !== undefined}
            errorMessage={problems.reasoningMaxTokens}
          />
        </div>
        <Field
          label="Prompt"
          value={draft.prompt}
          onChange={set("prompt")}
          isInvalid={problems.prompt !== undefined}
          errorMessage={problems.prompt}
        />
        <div>
          <Button
            size="sm"
            isDisabled={invalid || update.isPending}
            onPress={() => {
              const sent = draft
              update.mutate(
                {
                  interval_minutes: Number(sent.interval.trim()),
                  prompt: sent.prompt,
                  max_tokens: Number(sent.maxTokens.trim()),
                  reasoning_max_tokens: Number(sent.reasoningMaxTokens.trim()),
                  reasoning_every_runs: Number(sent.reasoningEvery.trim()),
                },
                { onSuccess: () => setSaved(sent) },
              )
            }}
          >
            {update.isPending ? "Saving…" : "Save"}
          </Button>
          {showSaved ? (
            <span role="status" className="ml-3 text-sm text-success">
              Saved
            </span>
          ) : null}
        </div>
      </div>
    </SettingsGroup>
  )
}

function addressProblem(addresses: string[]): string | undefined {
  const invalid = addresses.filter((address) => !looksLikeAddress(address))
  if (invalid.length > 0) {
    return `Not an email address: ${invalid.join(", ")}`
  }
  if (addresses.length > MAX_ALERT_EMAILS) {
    return `Up to ${MAX_ALERT_EMAILS} addresses.`
  }
  return undefined
}

/** Who hears about a failure. Saved apart from the schedule, which has its own Save. */
function AlertsCard({ saved }: { saved: string[] }) {
  const { mail_ready } = useDeployment()
  const update = useUpdateHealthCheckSettings()
  const [draft, setDraft] = useState(saved.join("\n"))
  const addresses = parseAddresses(draft).map((address) =>
    address.toLowerCase(),
  )
  const problem = addressProblem(addresses)
  const isDirty = addresses.join("\n") !== saved.join("\n")

  return (
    <SettingsGroup
      title="Email alerts"
      description="One email when a model stops answering, and one when it answers again. A run that changes nothing sends nothing, so a provider that stays down all night sends two emails, not one an hour."
    >
      <div className="flex flex-col gap-4 py-4">
        {mail_ready ? null : (
          <InfoBanner tone="warning">
            This deployment can't send email yet, so nobody below hears about a
            failure. Set up outgoing mail in{" "}
            <Link to="/settings" className="text-link hover:underline">
              Settings
            </Link>
            .
          </InfoBanner>
        )}
        <ErrorBanner error={update.error} />
        <TextArea
          label="Send alerts to"
          value={draft}
          onChange={setDraft}
          rows={3}
          placeholder="oncall@example.com"
          isInvalid={problem !== undefined}
          errorMessage={problem}
          description="Separated by commas or new lines. Leave it empty to send no email."
        />
        <div className="flex flex-wrap items-center gap-3">
          <Button
            size="sm"
            isDisabled={!isDirty || problem !== undefined || update.isPending}
            onPress={() => update.mutate({ alert_emails: addresses })}
          >
            {update.isPending ? "Saving…" : "Save addresses"}
          </Button>
          {update.isSuccess && !isDirty ? (
            <span role="status" className="text-sm text-success">
              Saved
            </span>
          ) : null}
        </div>
      </div>
    </SettingsGroup>
  )
}

/**
 * The operator's view of whether every model this deployment serves answers.
 *
 * Read from the runs a worker stores, not from customer traffic, so a provider
 * nobody has called today still has a status. One table per provider first,
 * because the question this page answers is which provider is failing; the
 * models follow, failing ones only unless asked.
 */
export function HealthChecksPage() {
  const status = useHealthCheckStatus()
  const run = useRunHealthChecks()
  const disable = useDisableFailingModels()
  const [showAll, setShowAll] = useState(false)
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(PAGE_SIZE_OPTIONS[0])
  const data = status.data

  if (status.isPending && !data) {
    return <PageLoading label="Loading health checks…" />
  }

  const busy = Boolean(data?.running || data?.settings.run_requested)
  const results = data
    ? data.results.filter((result) => showAll || result.status === "fail")
    : []
  const failing = data
    ? data.results.filter((result) => result.status === "fail").length
    : 0
  // A run or a disable can shrink the list under the page the operator is on.
  const lastPage = Math.max(0, Math.ceil(results.length / pageSize) - 1)
  const shownPage = Math.min(page, lastPage)
  const pageRows = results.slice(
    shownPage * pageSize,
    (shownPage + 1) * pageSize,
  )

  return (
    <div className="flex flex-col gap-6">
      <PageIntro
        title="Health"
        action={
          data ? (
            <Button
              size="sm"
              variant="primary"
              isDisabled={busy || run.isPending}
              onPress={() => run.mutate(undefined)}
            >
              {busy ? "Running…" : "Run now"}
            </Button>
          ) : null
        }
      >
        Whether every model your providers serve answers, checked with one small
        request each.
      </PageIntro>
      <ErrorBanner error={status.error ?? run.error ?? disable.error} />
      {data ? (
        <>
          <RunSummary status={data} />
          {data.last_run && data.last_run.total === 0 ? (
            <InfoBanner tone="warning">
              The last run found no chat model to check. Switch a model on under{" "}
              <Link
                to="/admin/hosted-providers"
                className="text-link hover:underline"
              >
                Hosted providers
              </Link>
              .
            </InfoBanner>
          ) : null}
          <DataTable
            ariaLabel="Providers"
            columns={PROVIDER_COLUMNS}
            rows={data.providers}
            getRowKey={(row) => row.provider}
            mobileLayout="cards"
            emptyContent="No results yet. Run the checks to see each provider."
          />
          {data.results.length > 0 ? (
            <SettingsGroup
              title={showAll ? "Models" : "Failing models"}
              action={
                <span className="flex flex-wrap items-center gap-2">
                  {failing > 0 ? (
                    <ConfirmButton
                      confirmLabel={`Disable ${failing} ${failing === 1 ? "model" : "models"}`}
                      isPending={disable.isPending}
                      onConfirm={() => disable.mutate()}
                    >
                      Disable all
                    </ConfirmButton>
                  ) : null}
                  <Button
                    size="sm"
                    onPress={() => {
                      setShowAll(!showAll)
                      setPage(0)
                    }}
                  >
                    {showAll ? "Failing only" : "Show all"}
                  </Button>
                </span>
              }
            >
              {disable.data ? (
                <p role="status" className="py-2 text-sm text-muted">
                  {disable.data.disabled === 1
                    ? "1 model switched off."
                    : `${disable.data.disabled} models switched off.`}{" "}
                  Switch them back on under{" "}
                  <Link
                    to="/admin/hosted-providers"
                    className="text-link hover:underline"
                  >
                    Hosted providers
                  </Link>
                  .
                </p>
              ) : null}
              <DataTable
                ariaLabel="Models"
                columns={RESULT_COLUMNS}
                rows={pageRows}
                getRowKey={(row) => row.selector}
                mobileLayout="cards"
                emptyContent="Every model answered."
              />
              {results.length > PAGE_SIZE_OPTIONS[0] ? (
                <TablePagination
                  page={shownPage}
                  pageSize={pageSize}
                  total={results.length}
                  rowsOnPage={pageRows.length}
                  onPageChange={setPage}
                  onPageSizeChange={(size) => {
                    setPageSize(size)
                    setPage(0)
                  }}
                />
              ) : null}
            </SettingsGroup>
          ) : null}
          <AlertsCard saved={data.settings.alert_emails} />
          <SettingsCard settings={data.settings} />
        </>
      ) : null}
    </div>
  )
}
