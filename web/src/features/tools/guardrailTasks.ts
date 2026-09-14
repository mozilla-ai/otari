/**
 * Turning the built-in guardrail catalog into the two controls that choose one.
 *
 * An operator knows what they want checked long before they know which vendor
 * checks it, so the form asks for the task first and offers only the guardrails
 * that do it. Both halves are derived from `GET /tool-settings/guardrails/catalog`
 * rather than listed here: this gateway ships forty guardrails and a newer one
 * ships more, so a list written down would be wrong on the next release.
 *
 * Two things are written down, and only these two. The task copy, because the
 * wire names are the API's vocabulary and "general_judge" is not a sentence an
 * operator should have to decode. And the order those tasks are offered in,
 * because sorting them by name or by count would move the common ones around as
 * the catalog grows.
 */

import type { BuiltInGuardrailSpec, GuardrailBackend } from "@/client"
import type { ComboBoxOption } from "@/design-system/forms/ComboBoxField"
import type { SelectOption } from "@/design-system/forms/Select"

interface TaskCopy {
  label: string
  /** One line on what the task catches, shown under the picker. */
  help: string
}

// Ordered as offered. Not alphabetical and not by how many guardrails do it:
// both would shuffle the list a gateway upgrade at a time, and the first two
// are what most deployments come here for.
const TASKS: ReadonlyArray<readonly [string, TaskCopy]> = [
  [
    "prompt_injection",
    {
      label: "Prompt injection",
      help: "Catches an attempt to override your instructions.",
    },
  ],
  [
    "content_safety",
    {
      label: "Harmful content",
      help: "Catches violence, self-harm, sexual and similar content.",
    },
  ],
  [
    "toxicity",
    {
      label: "Toxic language",
      help: "Catches insults, harassment and hate speech.",
    },
  ],
  [
    "pii",
    {
      label: "Personal data (PII)",
      help: "Catches names, addresses, card numbers and the like.",
    },
  ],
  [
    "hallucination",
    {
      label: "Made-up answers",
      help: "Checks an answer against the text it was given.",
    },
  ],
  [
    "off_topic",
    {
      label: "Off-topic replies",
      help: "Checks the answer stayed on a subject you allow.",
    },
  ],
  ["bias", { label: "Bias", help: "Catches unfair treatment of a group." }],
  [
    "tool_use",
    {
      label: "Tool misuse",
      help: "Checks a tool call is one the model should make.",
    },
  ],
  [
    "general_judge",
    {
      label: "Your own rule, judged by a model",
      help: "You write the rule; a model decides whether the text breaks it.",
    },
  ],
]

const TASK_COPY = new Map(TASKS)
const TASK_ORDER = TASKS.map(([category]) => category)

/** A snake_case wire name as a sentence, for a category this bundle predates. */
function fallbackLabel(category: string): string {
  const words = category.split("_").filter(Boolean).join(" ")
  return words === "" ? category : words[0].toUpperCase() + words.slice(1)
}

/** What the task is called in the dashboard. */
export function taskLabel(category: string): string {
  return TASK_COPY.get(category)?.label ?? fallbackLabel(category)
}

/** Every task one guardrail names, whether as its headline or not. */
function tasksOf(spec: BuiltInGuardrailSpec): Set<string> {
  return new Set<string>([spec.primary_category, ...(spec.categories ?? [])])
}

/**
 * The tasks this build can actually do, in the order above.
 *
 * A task no installed guardrail names is left out rather than offered empty: an
 * operator who picks it would reach a picker with nothing in it and no reason
 * given. A task this bundle has no copy for is still offered, at the end, since
 * a gateway newer than the dashboard is the ordinary way that happens.
 */
export function taskOptions(
  guardrails: readonly BuiltInGuardrailSpec[],
): SelectOption[] {
  const present = new Set<string>()
  for (const spec of guardrails) {
    for (const task of tasksOf(spec)) present.add(task)
  }
  const known = TASK_ORDER.filter((task) => present.has(task))
  const rest = [...present].filter((task) => !TASK_COPY.has(task)).sort()
  return [...known, ...rest].map((task) => ({
    value: task,
    label: taskLabel(task),
  }))
}

/** What the chosen task catches, and how many guardrails offer it. */
export function taskHelp(
  guardrails: readonly BuiltInGuardrailSpec[],
  task: string,
): string {
  if (task === "") return ""
  const count = guardrailsForTask(guardrails, task).length
  const noun = count === 1 ? "guardrail can" : "guardrails can"
  // The count is derived rather than written down, so it cannot disagree with a
  // gateway that ships one more guardrail than this bundle was written against.
  return `${TASK_COPY.get(task)?.help ?? ""} ${count} ${noun} do this.`.trim()
}

/**
 * The guardrails that do one task, what can run here first.
 *
 * Runnable first is the whole ordering decision. On a default install thirteen
 * of the eighteen prompt-injection guardrails want a Python extra that is not
 * present, so an alphabetical list opens on five rows nobody can use.
 */
export function guardrailsForTask(
  guardrails: readonly BuiltInGuardrailSpec[],
  task: string,
): BuiltInGuardrailSpec[] {
  if (task === "") return []
  return guardrails
    .filter((spec) => tasksOf(spec).has(task))
    .sort((a, b) => {
      if (a.runnable !== b.runnable) return a.runnable ? -1 : 1
      return a.display_name.localeCompare(b.display_name)
    })
}

// What the backend means for the operator's machine, rather than for the person
// who wrote the guardrail. The wire words name a model architecture; these name
// what it costs to run.
const BACKEND_WORDS: Record<GuardrailBackend, string | null> = {
  hosted_api: "hosted API",
  local_encoder: "local model",
  local_decoder: "local LLM",
  library_wrapped: "local library",
  // Nothing true to say, so the hint drops the segment rather than printing it.
  unknown: null,
}

/** The muted second line of a picker row: who runs it, where, and what it needs. */
function optionHint(spec: BuiltInGuardrailSpec): string {
  const parts = [spec.vendor, BACKEND_WORDS[spec.backend]]
  if (!spec.runnable) {
    parts.push(
      spec.missing_extra
        ? `install ${spec.missing_extra} to use`
        : "not available in this build",
    )
  } else if (spec.requires_api_key) {
    parts.push("needs an API key")
  }
  return parts.filter(Boolean).join(" · ")
}

/**
 * The picker's rows for one task.
 *
 * A guardrail that cannot run stays in the list, disabled, rather than being
 * filtered out of it. "Otari can do this once I install that extra" is worth
 * knowing, and a dimmed row says it where an absent row says nothing at all.
 */
export function guardrailOptions(
  guardrails: readonly BuiltInGuardrailSpec[],
  task: string,
): ComboBoxOption[] {
  return guardrailsForTask(guardrails, task).map((spec) => ({
    value: spec.guardrail_name,
    label: spec.display_name,
    hint: optionHint(spec),
    isDisabled: !spec.runnable,
  }))
}

/**
 * Whether a row survives what is being typed.
 *
 * `ComboBoxField` filters nothing itself, so the caller decides what a match is.
 * The hint is matched as well as the label, which is what lets "hosted" or a
 * vendor's name reach a guardrail whose own name carries neither.
 */
export function matchesQuery(
  spec: BuiltInGuardrailSpec,
  query: string,
): boolean {
  const needle = query.trim().toLowerCase()
  if (needle === "") return true
  return [spec.display_name, spec.vendor, optionHint(spec)]
    .join(" ")
    .toLowerCase()
    .includes(needle)
}

/**
 * A name for a new definition, taken from the task it does.
 *
 * The name is what a caller sends as its `profile`, and "prompt-injection" is
 * what it would have been typed as anyway, so suggesting it leaves the ordinary
 * case with one field to fill: the credential. Stops suggesting the moment the
 * operator types something of their own.
 */
export function suggestName(task: string, taken: readonly string[]): string {
  if (task === "") return ""
  const base = task.replaceAll("_", "-")
  const used = new Set(taken)
  if (!used.has(base)) return base
  let suffix = 2
  while (used.has(`${base}-${suffix}`)) suffix += 1
  return `${base}-${suffix}`
}
