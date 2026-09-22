import type { OrganizationGuardrailDefinition } from "@/client"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Dot } from "@/design-system/indicators/Dot"
import {
  type DefinitionHealth,
  definitionHealth,
  type MandateConsequence,
} from "@/features/guardrails/buildState"

// `Badge` has only a quiet tone and a warning one, and these say a check is
// not happening, so the danger states are a dot and danger ink of their own.

function Mark({
  tone,
  children,
}: {
  tone: "danger" | "muted"
  children: string
}) {
  return (
    <span
      className={`flex items-center gap-2 text-xs ${tone === "danger" ? "text-danger" : "text-subtle"}`}
    >
      <Dot className={tone === "danger" ? "bg-danger" : "bg-text-subtle"} />
      {children}
    </span>
  )
}

const HEALTH_WORDS: Record<
  DefinitionHealth,
  { tone: "danger" | "muted"; label: string }
> = {
  running: { tone: "muted", label: "Active" },
  starting: { tone: "muted", label: "Starting" },
  off: { tone: "muted", label: "Off" },
  not_running: { tone: "danger", label: "Not running" },
  credentials_unreadable: { tone: "danger", label: "Credentials unreadable" },
}

/** Whether a definition is built on the worker that answered. */
export function DefinitionStatus({
  definition,
}: {
  definition: OrganizationGuardrailDefinition
}) {
  const words = HEALTH_WORDS[definitionHealth(definition)]
  return <Mark tone={words.tone}>{words.label}</Mark>
}

/** What a mandate is doing to the requests it covers, when not what it says. */
export function MandateConsequenceMark({
  consequence,
}: {
  consequence: MandateConsequence
}) {
  if (consequence === "") return null
  return (
    <Mark tone="danger">
      {consequence === "refused"
        ? "Requests refused"
        : "Requests served unchecked"}
    </Mark>
  )
}

/**
 * The count of mandates serving requests unchecked, consequence first. Counted
 * in mandates, since one broken definition can open several of them.
 */
export function ServedUncheckedBanner({ count }: { count: number }) {
  if (count === 0) return null
  const subject =
    count === 1
      ? "1 mandated guardrail is not running, so the requests it covers are"
      : `${count} mandated guardrails are not running, so the requests they cover are`
  return (
    <InfoBanner tone="warning">{`${subject} being served unchecked.`}</InfoBanner>
  )
}
