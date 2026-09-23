import { describe, expect, it } from "vitest"

import type { GuardrailParameterSpec } from "@/client"
import {
  jsonFieldSpec,
  suggestedCreateKwargs,
  suggestedOption,
} from "@/features/guardrails/guardrailFieldSuggestions"

function spec(name: string): GuardrailParameterSpec {
  return { name, type: "json", required: false, secret: false, storable: true }
}

const DETECTION_CONFIG = spec("detection_config")
const EVALUATORS = spec("evaluators")
const DETECTORS = spec("detectors")

describe("jsonFieldSpec", () => {
  it("describes a field the registry names", () => {
    expect(jsonFieldSpec("alinia", DETECTION_CONFIG)?.kind).toBe("flags")
    expect(jsonFieldSpec("patronus", EVALUATORS)?.kind).toBe("presets")
    expect(jsonFieldSpec("alinia", spec("metadata"))?.kind).toBe("map")
    expect(jsonFieldSpec("alinia", spec("context_documents"))?.kind).toBe(
      "list",
    )
  })

  it("says nothing about a field it has never heard of", () => {
    // Which is what leaves a guardrail added upstream with a working JSON
    // editor rather than with a control built from a guess.
    expect(jsonFieldSpec("alinia", spec("invented_later"))).toBeUndefined()
    expect(jsonFieldSpec("future_guardrail", DETECTION_CONFIG)).toBeUndefined()
  })

  it("keys on the guardrail as well as the parameter", () => {
    // Two guardrails can both take `metadata` and mean different things by it.
    expect(jsonFieldSpec("lakera_guard", spec("metadata"))?.kind).toBe("map")
    expect(
      jsonFieldSpec("bedrock_guardrails", spec("metadata")),
    ).toBeUndefined()
  })
})

describe("suggestedOption", () => {
  it("names the detection the chosen operation means", () => {
    const field = jsonFieldSpec("alinia", DETECTION_CONFIG)
    expect(suggestedOption(field!, "prompt_injection")?.key).toBe("security")
    expect(suggestedOption(field!, "hallucination")?.key).toBe("hallucination")
  })

  it("suggests nothing for an operation the vendor does not document a key for", () => {
    const field = jsonFieldSpec("alinia", DETECTION_CONFIG)
    expect(suggestedOption(field!, "pii")).toBeUndefined()
    expect(suggestedOption(field!, "tool_use")).toBeUndefined()
  })
})

describe("suggestedCreateKwargs", () => {
  it("switches on the detection the operator just asked for", () => {
    expect(
      suggestedCreateKwargs([DETECTION_CONFIG], "alinia", "prompt_injection"),
    ).toEqual({ detection_config: { security: true } })
  })

  it("sends a detector's own settings shape rather than a bare true", () => {
    // watsonx takes `{"granite_guardian": {}}`, not `{"granite_guardian": true}`.
    expect(
      suggestedCreateKwargs([DETECTORS], "watsonx_guardian", "pii"),
    ).toEqual({ detectors: { pii: {} } })
  })

  it("seeds a preset list as a list of one", () => {
    expect(
      suggestedCreateKwargs([EVALUATORS], "patronus", "prompt_injection"),
    ).toEqual({
      evaluators: [
        { evaluator: "judge", criteria: "patronus:prompt-injection" },
      ],
    })
  })

  it("seeds nothing before an operation is chosen", () => {
    expect(suggestedCreateKwargs([DETECTION_CONFIG], "alinia", "")).toEqual({})
  })

  it("seeds nothing for a field or an operation it cannot speak for", () => {
    expect(
      suggestedCreateKwargs([spec("metadata")], "alinia", "prompt_injection"),
    ).toEqual({})
    expect(suggestedCreateKwargs([DETECTION_CONFIG], "alinia", "pii")).toEqual(
      {},
    )
  })
})
