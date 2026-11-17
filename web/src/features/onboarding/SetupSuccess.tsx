import { FiCheck } from "react-icons/fi"

import type { ActivationAttempt } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { DialogSection } from "@/design-system/feedback/Dialog"
import { formatCost, formatRelative } from "@/shared/helpers/format"

/** Latency and cost. A figure the gateway did not report is dropped. */
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

/** What the request was. A row the gateway did not report is dropped. */
function RequestFacts({ attempt }: { attempt: ActivationAttempt }) {
  const facts = [
    ["Model", attempt.model],
    ["Provider", attempt.provider],
    ["Request ID", attempt.request_id],
    ["Received", formatRelative(attempt.occurred_at)],
  ].filter((fact): fact is [string, string] => Boolean(fact[1]))

  return (
    // Takes the rest of the body, so the bands below stay where the sheet's were.
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

/** The listening panel's counterpart. */
function SetupComplete({ workspaceName }: { workspaceName?: string }) {
  return (
    <div className="flex min-w-0 flex-1 items-center gap-3.5">
      {/* The orb's size, so the copy beside it stays put. */}
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
 * The first-run sheet's second face, once the workspace's first request lands.
 * Its bands answer the setup face's in the same places: the figures where the
 * key was, the request where the example was, the finished status where the
 * listening panel was. One action, to the page that analyzes the request.
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
