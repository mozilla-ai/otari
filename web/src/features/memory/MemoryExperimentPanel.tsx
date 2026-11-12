import { useState } from "react"

import type {
  MemoryExperiment,
  MemoryExperimentDifference,
  UpdateWorkspaceMemoryRequest,
  WorkspaceMemory,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Select } from "@/design-system/forms/Select"
import { SettingRow } from "@/design-system/layout/SettingRow"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { KpiCell } from "@/design-system/metrics/KpiCell"
import { KpiStrip } from "@/design-system/metrics/KpiStrip"
import { useMemoryExperiment } from "@/shared/api/memory"

const HOLDOUT_LABEL = "Sessions served without memory"
const HOLDOUT_OPTIONS = [10, 20, 30, 50].map((percent) => ({
  value: String(percent),
  label: `${percent}% of sessions`,
}))

const tokens = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 })
const percent = new Intl.NumberFormat("en-US", {
  style: "percent",
  maximumFractionDigits: 1,
  signDisplay: "exceptZero",
})
const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumSignificantDigits: 3,
})
const when = new Intl.DateTimeFormat("en-US", {
  dateStyle: "medium",
  timeStyle: "short",
})

function interval(
  difference: MemoryExperimentDifference,
  format: (value: number) => string,
): string {
  return `95% interval ${format(difference.low)} to ${format(difference.high)}`
}

function verdict(difference: MemoryExperimentDifference): string {
  if (!difference.conclusive) return "Not conclusive yet"
  return difference.absolute < 0 ? "Memory saves" : "Memory costs more"
}

function Results({ report }: { report: MemoryExperiment }) {
  const { control, memory, tokens: tokenDifference, cost } = report
  const needed = report.min_sessions
  const hitRate =
    report.recalls > 0
      ? new Intl.NumberFormat("en-US", { style: "percent" }).format(
          report.recalls_with_results / report.recalls,
        )
      : "—"
  return (
    <div className="flex flex-col gap-3">
      <KpiStrip isEmpty={control.sessions + memory.sessions === 0} columns={4}>
        <KpiCell
          label="Sessions"
          value={`${control.sessions} / ${memory.sessions}`}
          subline="without memory / with memory"
        />
        <KpiCell
          label="Tokens per session"
          value={tokens.format(memory.tokens_per_session)}
          subline={`${tokens.format(control.tokens_per_session)} without memory`}
        />
        <KpiCell
          label="Change with memory"
          value={
            tokenDifference?.relative != null
              ? percent.format(tokenDifference.relative)
              : "—"
          }
          subline={
            tokenDifference
              ? `${verdict(tokenDifference)}; ${interval(tokenDifference, (value) => tokens.format(value))}`
              : `Needs ${needed} sessions in each arm`
          }
        />
        <KpiCell
          label="Recall found something"
          value={hitRate}
          subline={`${report.recalls_with_results} of ${report.recalls} turns; ${report.searches} searches by agents`}
        />
      </KpiStrip>
      <p className="text-caption">
        Tokens count input and output, cached input once, net of memory's own
        model calls ({tokens.format(report.memory_call_tokens_per_session)} per
        session with memory).
        {cost
          ? ` Cost per session changes by ${usd.format(cost.absolute)} (${interval(cost, (value) => usd.format(value))}).`
          : " Cost is compared once every session's model has a price."}
      </p>
    </div>
  )
}

/**
 * The savings experiment: a share of the workspace's sessions served without
 * memory, compared with the rest by what each session used.
 *
 * Starting deals the sessions afresh; stopping keeps the results on show until
 * another experiment starts, so the share is fixed while one runs.
 */
export function MemoryExperimentPanel({
  workspaceId,
  memory,
  canManage,
  isSaving,
  onChange,
}: {
  workspaceId: string
  memory: WorkspaceMemory
  canManage: boolean
  isSaving: boolean
  onChange: (patch: Partial<UpdateWorkspaceMemoryRequest>) => void
}) {
  const [holdout, setHoldout] = useState("20")
  const report = useMemoryExperiment(
    workspaceId,
    memory.experiment_started_at != null,
  )
  const running = memory.experiment_holdout_percent > 0
  const locked = !canManage || isSaving

  return (
    <SettingsGroup
      title="Measure what memory saves"
      description="Serve a share of sessions without memory and compare what each session uses with the rest. Only requests that name their session take part: Claude Code and Codex do on their own, other clients send Otari-Conversation-Id."
    >
      {running ? (
        <SettingRow
          label="Experiment running"
          help={`Since ${when.format(new Date(memory.experiment_started_at ?? ""))}, serving ${memory.experiment_holdout_percent}% of sessions without memory.`}
          control={
            <Button
              variant="ghost"
              className="min-h-11 md:min-h-9"
              onPress={() => onChange({ experiment_holdout_percent: 0 })}
              isDisabled={locked}
            >
              Stop
            </Button>
          }
        />
      ) : (
        <div className="flex flex-col gap-2 md:flex-row md:items-end">
          <Select
            label={HOLDOUT_LABEL}
            value={holdout}
            onChange={setHoldout}
            options={HOLDOUT_OPTIONS}
            description="These sessions get neither memory's tools nor automatic recall."
            isDisabled={locked || !memory.enabled}
          />
          <Button
            variant="primary"
            className="min-h-11 md:min-h-9 shrink-0"
            onPress={() =>
              onChange({ experiment_holdout_percent: Number(holdout) })
            }
            isDisabled={locked || !memory.enabled}
          >
            {memory.experiment_started_at
              ? "Start a new experiment"
              : "Start experiment"}
          </Button>
        </div>
      )}
      {memory.experiment_ended_at && !running ? (
        <p className="text-caption">
          Results of the experiment that stopped{" "}
          {when.format(new Date(memory.experiment_ended_at))}.
        </p>
      ) : null}
      {report.data?.started_at ? <Results report={report.data} /> : null}
      <ErrorBanner error={report.error} />
    </SettingsGroup>
  )
}
