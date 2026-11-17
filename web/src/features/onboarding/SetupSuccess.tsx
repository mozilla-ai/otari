import { FiCheck } from "react-icons/fi"

import type { ActivationAttempt } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { DialogSection } from "@/design-system/feedback/Dialog"
import { formatCost, formatRelative } from "@/shared/helpers/format"

/**
 * The two figures a first request is judged by, large, over the band where the
 * sheet asked for the key.
 *
 * A cell whose value the gateway did not report is dropped rather than shown
 * empty, and the band goes when both are.
 */
function Figures({ attempt }: { attempt: ActivationAttempt }) {
  const cells = [
    {
      label: "Latency",
      value:
        attempt.latency_ms != null
          ? `${Math.round(attempt.latency_ms)} ms`
          : null,
    },
    {
      label: "Cost",
      value: attempt.cost_usd != null ? formatCost(attempt.cost_usd) : null,
    },
  ].filter((cell) => cell.value)

  if (cells.length === 0) return null

  return (
    <div className="border-border flex border-t">
      {cells.map((cell, index) => (
        <div
          key={cell.label}
          className={`flex min-w-0 flex-1 flex-col gap-1.5 px-6 pt-4 pb-5 ${
            index > 0 ? "border-border border-l" : ""
          }`}
        >
          <span className="text-mono-overline">{cell.label}</span>
          <span className="text-mono-figure text-foreground truncate">
            {cell.value}
          </span>
        </div>
      ))}
    </div>
  )
}

/**
 * What the request was, over the band where the sheet showed how to send it.
 * Rows the gateway did not report are dropped.
 */
function RequestFacts({ attempt }: { attempt: ActivationAttempt }) {
  const facts = [
    ["Model", attempt.model],
    ["Provider", attempt.provider],
    ["Request ID", attempt.request_id],
    ["Received", formatRelative(attempt.occurred_at)],
  ].filter((fact): fact is [string, string] => Boolean(fact[1]))

  return (
    // Takes the rest of the body, so the bands below stay where the sheet's
    // were and the frame does not resize.
    <DialogSection className="flex-1">
      <span className="text-emphasis">The request</span>
      <dl className="flex flex-col">
        {facts.map(([term, detail]) => (
          <div
            key={term}
            className="border-border flex items-baseline gap-4 border-b py-2.5 last:border-b-0"
          >
            <dt className="text-caption text-subtle w-28 shrink-0">{term}</dt>
            <dd className="text-mono-caption text-foreground min-w-0 truncate">
              {detail}
            </dd>
          </div>
        ))}
      </dl>
    </DialogSection>
  )
}

/** The finished counterpart of the listening panel, in the same band. */
function SetupComplete({ workspaceName }: { workspaceName?: string }) {
  return (
    <div className="flex min-w-0 flex-1 items-center gap-3.5">
      {/* The orb's own size, so the copy beside it does not move. */}
      <span className="bg-success-subtle flex size-10 shrink-0 items-center justify-center rounded-full">
        <FiCheck aria-hidden className="text-success size-5" />
      </span>
      <div className="flex min-w-0 flex-col gap-0.5" role="status">
        <span className="text-emphasis">Setup complete</span>
        <span className="text-caption text-subtle">
          Usage and the activity log now record every request to{" "}
          {workspaceName ?? "this workspace"}.
        </span>
      </div>
    </div>
  )
}

/**
 * The payoff, as the second face of the first-run sheet: the workspace's first
 * request landed.
 *
 * The same frame and the same bands as the setup face, each one answering
 * what stood in its place: the figures where the key was, the request where
 * the example was, the finished status where the listening panel was. The
 * frame keeps the sheet's size, and the wave that brings this face in leaves
 * every band where its counterpart stood.
 *
 * Still the quietest screen in the flow. One mark beside the heading, the
 * numbers as proof that the call was observed rather than something to
 * analyze, and one action to the page that does analyze them.
 */
export function setupSuccessFace({
  attempt,
  workspaceName,
  onOpenActivity,
}: {
  /** The request that succeeded. Absent if the gateway reported none. */
  attempt?: ActivationAttempt
  workspaceName?: string
  onOpenActivity: () => void
}) {
  return {
    title: "Your first call went through",
    description: (
      <>
        It landed in{" "}
        <span className="text-foreground font-medium">
          {workspaceName ?? "this workspace"}
        </span>
        . Otari observed it and finished setup for you.
      </>
    ),
    mark: (
      <span className="mt-0.5 flex shrink-0">
        <FiCheck aria-hidden className="text-success size-6" />
      </span>
    ),
    status: <SetupComplete workspaceName={workspaceName} />,
    footerStart: (
      <p className="text-caption">
        Your request is the first row in the activity log.
      </p>
    ),
    actions: (
      // Scoped to gateway traffic: imported usage is somebody else's requests,
      // and the one that just landed is the newest row of what is left.
      <Button variant="primary" onPress={onOpenActivity}>
        Continue to the activity log
      </Button>
    ),
    children: attempt ? (
      <>
        <Figures attempt={attempt} />
        <RequestFacts attempt={attempt} />
      </>
    ) : null,
  }
}
