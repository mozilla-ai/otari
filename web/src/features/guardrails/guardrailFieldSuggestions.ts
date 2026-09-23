/**
 * What a guardrail's JSON-typed arguments actually hold, so a form can ask for
 * them as controls rather than as a text box full of braces.
 *
 * Written down here, which is the exception in this feature and wants its
 * reason. Upstream's own `ParameterType.JSON` docstring calls it "the not
 * flat-form-able, use a JSON editor signal": `ParameterSpec` carries no item
 * type, no key list and no schema for these, so nothing published can say
 * whether a field is a list of sentences or a map of switches, let alone what
 * the keys are. The alternative to writing it down is a textarea, and a vendor's
 * detection vocabulary typed by hand into JSON is where a definition goes wrong.
 *
 * The same trade `features/providers/providerCredentialFields.ts` makes for
 * `client_args`, and on the same terms: the value is a passthrough nothing
 * between here and the vendor validates, so a name is only ever as good as the
 * documentation it was read from.
 *
 * **Every entry is a suggestion, never a whitelist.** A key this file has never
 * heard of still edits, still saves and still shows, because the JSON view sits
 * beside the controls and accepts anything. So a vendor adding a detection costs
 * an operator a checkbox and never a capability, which is what makes a list that
 * will go stale safe to keep.
 *
 * Sources, all of them any-guardrail's: each guardrail's constructor docstring
 * in `any_guardrail.guardrails.*`, and its page under
 * `docs.mozilla.ai/any-guardrail/api-reference`. Nothing here is invented, and a
 * key neither of those names is left out rather than guessed at.
 */

import type { GuardrailParameterSpec } from "@/client"

/** How a JSON argument is edited when it is not being edited as JSON. */
export type JsonFieldKind =
  /** A map of named switches, each on with its own optional settings. */
  | "flags"
  /** A map of keys the operator invents, each holding one plain value. */
  | "map"
  /** An ordered list of plain strings. */
  | "list"
  /** A list of objects built from a fixed set of named presets. */
  | "presets"

/** One switch a `flags` or `presets` field offers. */
export interface FieldOption {
  /** The wire key, or for `presets` the id this file gives the preset. */
  key: string
  label: string
  /** One line on what enabling it does. From the guardrail's own docstring. */
  help?: string
  /** What turning it on sends. Defaults to `true` for a flag. */
  value?: unknown
}

export interface JsonFieldSpec {
  kind: JsonFieldKind
  /**
   * One line, standing in for the catalog's own description.
   *
   * Only where the controls make that description wrong rather than long: a
   * paragraph on the dict shape to pass is what an operator needed before these
   * were checkboxes, and reading it beside them is being told to do the thing
   * they are no longer doing. The JSON view still shows the original.
   */
  help?: string
  options?: FieldOption[]
  /**
   * Which option to switch on for the operation the guardrail was picked under,
   * keyed by the catalog's own category. Absent for a category whose key the
   * vendor does not document; nothing is guessed, so the operator picks it.
   */
  byCategory?: Record<string, string>
  /** The label for one entry, where "item" would read poorly. */
  itemNoun?: string
}

// Keyed `guardrail_name.parameter`, which is how the catalog names both halves.
const FIELDS: Record<string, JsonFieldSpec> = {
  // `{"security": True}`, or nested per-policy settings such as
  // `{"safety": {"toxicity": 0.8}}` (alinia.py:75-76). `compliance` and
  // `hallucination` are named as policies in the class docstring (alinia.py:18).
  "alinia.detection_config": {
    kind: "flags",
    itemNoun: "detection",
    options: [
      {
        key: "security",
        label: "Security",
        help: "Prompt injection and data exfiltration.",
      },
      { key: "safety", label: "Safety", help: "Harmful and toxic content." },
      { key: "compliance", label: "Compliance", help: "Policy breaches." },
      {
        key: "hallucination",
        label: "Hallucination",
        help: "Answers not grounded in the documents given.",
      },
    ],
    byCategory: {
      prompt_injection: "security",
      content_safety: "safety",
      toxicity: "safety",
      hallucination: "hallucination",
    },
  },
  "alinia.metadata": {
    kind: "map",
    help: "Sent with every request, for Alinia-side monitoring.",
  },
  "alinia.blocked_response": {
    kind: "map",
    help: "What Alinia returns when it blocks something.",
  },
  "alinia.context_documents": {
    kind: "list",
    help: "Text the output-side checks are graded against.",
    itemNoun: "document",
  },

  // `{"granite_guardian": {}}` by default; `hap` and `pii` are the other two
  // (watsonx_guardian.py:118-120). Each takes its own settings dict, so an
  // enabled detector sends `{}` rather than `true`.
  "watsonx_guardian.detectors": {
    kind: "flags",
    itemNoun: "detector",
    options: [
      {
        key: "granite_guardian",
        label: "Granite Guardian",
        help: "Harm, bias, violence, jailbreak and groundedness.",
        value: {},
      },
      {
        key: "hap",
        label: "Hate, abuse and profanity",
        help: "Slurs, harassment and swearing.",
        value: {},
      },
      {
        key: "pii",
        label: "Personal data",
        help: "Names, addresses, card numbers.",
        value: {},
      },
    ],
    byCategory: {
      pii: "pii",
      toxicity: "hap",
      content_safety: "granite_guardian",
      bias: "granite_guardian",
      prompt_injection: "granite_guardian",
      hallucination: "granite_guardian",
    },
  },

  // A list of dicts, each at least `{"evaluator": ...}` plus optional
  // `criteria`, e.g. `[{"evaluator": "judge", "criteria":
  // "patronus:prompt-injection"}]` (patronus.py:69-72). `lynx` and
  // `answer-relevance` are the other two named (patronus.py:22-23).
  "patronus.evaluators": {
    kind: "presets",
    itemNoun: "evaluator",
    options: [
      {
        key: "prompt-injection",
        label: "Prompt injection",
        help: "Attempts to override your instructions.",
        value: { evaluator: "judge", criteria: "patronus:prompt-injection" },
      },
      {
        key: "hallucination",
        label: "Hallucination",
        help: "Answers not grounded in the context given.",
        value: { evaluator: "judge", criteria: "patronus:hallucination" },
      },
      {
        key: "lynx",
        label: "Lynx",
        help: "Patronus's own hallucination model.",
        value: { evaluator: "lynx" },
      },
      {
        key: "answer-relevance",
        label: "Answer relevance",
        help: "Whether the answer addresses the question.",
        value: { evaluator: "answer-relevance" },
      },
      {
        key: "judge",
        label: "Model as judge",
        help: "Your own rule, decided by a model.",
        value: { evaluator: "judge" },
      },
    ],
    byCategory: {
      prompt_injection: "prompt-injection",
      hallucination: "hallucination",
      off_topic: "answer-relevance",
      general_judge: "judge",
    },
  },
  "patronus.tags": {
    kind: "map",
    help: "Sent with every request, for your own reporting.",
  },
  "patronus.retrieved_context": {
    kind: "list",
    help: "Text the grounding checks are graded against.",
    itemNoun: "document",
  },

  "lakera_guard.metadata": {
    kind: "map",
    help: "Sent with every request, for your own reporting.",
  },
  "azure_content_safety.blocklist_names": {
    kind: "list",
    help: "Blocklists you created in Azure, by name.",
    itemNoun: "blocklist",
  },
  "azure_prompt_shields.documents": {
    kind: "list",
    help: "Extra text to scan for indirect injection.",
    itemNoun: "document",
  },
}

/**
 * How to edit one JSON argument, or undefined when there is nothing to say.
 *
 * A field this file does not name falls back to whatever its current value
 * looks like, and to the JSON view when it has none, which is the behavior a
 * guardrail added upstream gets for free.
 */
export function jsonFieldSpec(
  guardrailName: string,
  spec: GuardrailParameterSpec,
): JsonFieldSpec | undefined {
  return FIELDS[`${guardrailName}.${spec.name}`]
}

/** The option to switch on for the operation the guardrail was picked under. */
export function suggestedOption(
  field: JsonFieldSpec,
  category: string,
): FieldOption | undefined {
  const key = field.byCategory?.[category]
  return key === undefined
    ? undefined
    : field.options?.find((option) => option.key === key)
}

/**
 * What a new definition starts with, given the operation it was picked under.
 *
 * The operator chose "prompt injection" one control ago, so Alinia opening with
 * nothing ticked asks them to answer it again in the vendor's own words. Only
 * where the vendor documents which key means that operation: a category with no
 * entry seeds nothing rather than guessing one.
 *
 * Seeds a new definition only. An existing row's stored arguments are what it
 * opens on, whatever this would have suggested.
 */
export function suggestedCreateKwargs(
  specs: readonly GuardrailParameterSpec[],
  guardrailName: string,
  category: string,
): Record<string, unknown> {
  const seeded: Record<string, unknown> = {}
  if (category === "") return seeded
  for (const spec of specs) {
    const field = jsonFieldSpec(guardrailName, spec)
    if (!field) continue
    const option = suggestedOption(field, category)
    if (!option) continue
    const value = option.value ?? true
    if (field.kind === "presets") seeded[spec.name] = [value]
    if (field.kind === "flags") seeded[spec.name] = { [option.key]: value }
  }
  return seeded
}
