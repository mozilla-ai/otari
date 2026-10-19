import type { TraceToolCost } from "@/client"
import { Tooltip } from "@/design-system/overlays/Tooltip"
import { formatCost, formatNumber } from "@/shared/helpers/format"

const DOTTED =
  "underline decoration-dotted decoration-control-border underline-offset-4"

function plural(count: number, word: string): string {
  return `${formatNumber(count)} ${word}${count === 1 ? "" : "s"}`
}

function Part({
  term,
  how,
  cost,
  detail,
}: {
  term: string
  how: string
  cost: number
  detail?: string
}) {
  return (
    <div className="contents">
      <dt className="text-caption text-subtle">
        <Tooltip content={how} delay={0}>
          {(props) => (
            <button type="button" {...props} className={DOTTED}>
              {term}
            </button>
          )}
        </Tooltip>
      </dt>
      <dd className="text-right text-mono-caption">{formatCost(cost)}</dd>
      <dd className="text-caption text-subtle">{detail}</dd>
    </div>
  )
}

// What a tool call cost, which a provider never bills on its own: read off the
// model requests either side of it, so every figure but a subagent's is an estimate.
export function CostCard({ cost }: { cost: TraceToolCost }) {
  const shared = (count: number) =>
    count > 1 ? `shared with ${plural(count - 1, "other call")}` : undefined
  return (
    <section className="flex flex-col gap-2">
      <h3 className="text-overline">Cost</h3>
      <p className="text-body">
        <span className="text-title">≈ {formatCost(cost.total)}</span>{" "}
        <span className="text-caption text-subtle">estimated</span>
      </p>
      <p className="text-caption text-subtle">
        Providers bill model requests, not tool calls, so this is read off the
        requests around the call: what it took to ask for it, to add its result
        to the context, and to read that result again on every later request.
      </p>
      <dl className="grid grid-cols-[auto_auto_1fr] items-baseline gap-x-4 gap-y-1.5">
        <Part
          term="Calling it"
          how="Its share of the output of the request that called it, which also paid for any text or thinking"
          cost={cost.calling}
          detail={shared(cost.calls_sharing_request)}
        />
        <Part
          term="Adding its result"
          how="How much the next request's context grew, at what that request paid for new input, usually a cache write"
          cost={cost.writing}
          detail={[
            `≈ ${plural(cost.result_tokens, "token")}`,
            shared(cost.calls_sharing_result),
          ]
            .filter(Boolean)
            .join(", ")}
        />
        <Part
          term="Re-reading it"
          how="Every later request in the same context reads the result again, at its cache-read rate, or its cache-write rate once the cache expired, until the context is compacted"
          cost={cost.rereading}
          detail={`in ${plural(cost.later_requests, "later request")}`}
        />
        {cost.subagent != null ? (
          <Part
            term="Its subagent"
            how="What the requests of the subagent this call ran cost, exactly"
            cost={cost.subagent}
            detail="exact"
          />
        ) : null}
      </dl>
    </section>
  )
}
