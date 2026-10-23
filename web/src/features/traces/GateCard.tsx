import { Chip, type ChipTone } from "@/design-system/indicators/Chip"
import { formatCost, formatNumber } from "@/shared/helpers/format"
import { GATE_KIND, GATE_MOMENT, says } from "./gateWords"
import type { GuardrailVerdict } from "./gateway"
import type { GateCheck, JudgeCalls } from "./types"

export const VERDICT: Record<
  GuardrailVerdict,
  { word: string; tone: ChipTone }
> = {
  passed: { word: "Passed", tone: "success" },
  blocked: { word: "Blocked", tone: "warning" },
  flagged: { word: "Flagged", tone: "warning" },
  refused: { word: "Could not decide, blocked", tone: "danger" },
  unchecked: { word: "Could not decide", tone: "danger" },
}

const VERDICT_SAYS: Record<GuardrailVerdict, string> = {
  passed: "Nothing this check read broke the gate.",
  blocked: "The gate stopped the agent here.",
  flagged: "The gate failed, but it is advisory, so the agent went ahead.",
  refused:
    "The gate could not decide, and stopped the agent rather than let it through unchecked.",
  unchecked:
    "The gate could not decide, so what it checked went ahead unchecked.",
}

function Fact({ term, children }: { term: string; children: string }) {
  return (
    <div className="flex flex-col gap-0.5 sm:flex-row sm:gap-3">
      <dt className="w-32 shrink-0 text-caption text-subtle">{term}</dt>
      <dd className="min-w-0 break-words text-body">{children}</dd>
    </div>
  )
}

function Read({ title, text }: { title: string; text: string }) {
  return (
    <section className="flex flex-col gap-1.5">
      <h3 className="text-overline">{title}</h3>
      <pre className="whitespace-pre-wrap break-words bg-surface-subtle px-3 py-2 text-mono-caption">
        {text}
      </pre>
    </section>
  )
}

function judgeTokens(judge: JudgeCalls): string {
  const cached = []
  if (judge.cacheReadTokens > 0) {
    cached.push(`${formatNumber(judge.cacheReadTokens)} read from the cache`)
  }
  if (judge.cacheWriteTokens > 0) {
    cached.push(`${formatNumber(judge.cacheWriteTokens)} written to the cache`)
  }
  const input = `${formatNumber(judge.inputTokens)} input`
  return `${cached.length > 0 ? `${input} (${cached.join(", ")})` : input}, ${formatNumber(judge.outputTokens)} output`
}

function JudgeFacts({ judge }: { judge: JudgeCalls }) {
  return (
    <section className="flex flex-col gap-1.5">
      <h3 className="text-overline">Its model calls</h3>
      <dl className="flex flex-col gap-2">
        <Fact term="Model">{judge.model}</Fact>
        <Fact term="Tokens">{judgeTokens(judge)}</Fact>
        <Fact term="Cost">
          {judge.cost != null
            ? `${formatCost(judge.cost)}, in the session's cost`
            : "Not priced"}
        </Fact>
      </dl>
    </section>
  )
}

// What a guardrail gate is and what this one check of it decided, in place of
// the generic input and output a span shows.
export function GateCard({ gate }: { gate: GateCheck }) {
  const verdict = VERDICT[gate.verdict]
  const others = gate.runs.filter((moment) => moment !== gate.moment)
  return (
    <div className="flex flex-col gap-4">
      <section className="flex flex-col gap-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <Chip tone={verdict.tone}>{verdict.word}</Chip>
          <span className="break-words text-mono-caption">{gate.gateId}</span>
        </div>
        <p className="text-body">{VERDICT_SAYS[gate.verdict]}</p>
      </section>

      <section className="flex flex-col gap-1.5">
        <h3 className="text-overline">What this gate is for</h3>
        <p className="text-body">
          {gate.message ??
            "The guardrail gives this gate no message, or its definition did not reach this server."}
        </p>
      </section>

      <dl className="flex flex-col gap-2">
        {gate.gateType ? (
          <Fact term="Kind">{says(GATE_KIND, gate.gateType)}</Fact>
        ) : null}
        {gate.enforcement ? (
          <Fact term="Enforcement">
            {gate.enforcement === "required"
              ? "Required: stops the agent when it fails"
              : "Advisory: warns, never stops the agent"}
          </Fact>
        ) : null}
        {gate.moment ? (
          <Fact term="This check">{says(GATE_MOMENT, gate.moment)}</Fact>
        ) : null}
        {others.length > 0 ? (
          <Fact term="Also runs">
            {others.map((moment) => says(GATE_MOMENT, moment)).join("; ")}
          </Fact>
        ) : null}
        {gate.origin ? (
          <Fact term="Defined in">
            {gate.origin === "user"
              ? "Your own guardrail, ~/.otari/"
              : "The repository's guardrail, .otari/"}
          </Fact>
        ) : null}
        {gate.digest ? (
          <Fact term="Guardrail version">{gate.digest}</Fact>
        ) : null}
      </dl>

      {gate.judge ? <JudgeFacts judge={gate.judge} /> : null}

      {gate.evidence ? (
        <Read title="What it checked" text={gate.evidence} />
      ) : null}
      {gate.detail ? <Read title="What it found" text={gate.detail} /> : null}
      {!gate.evidence && !gate.detail ? (
        <p className="text-caption text-subtle">
          What this check read is not kept. <code>otari hook</code> sends it
          with <code>OTARI_HOOK_REPORT_CONTENT=1</code>, to a server that keeps
          content.
        </p>
      ) : null}
    </div>
  )
}
