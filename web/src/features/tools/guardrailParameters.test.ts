import { describe, expect, it } from "vitest"

import type { GuardrailParameterSpec } from "@/client"
import {
  buildValidateKwargs,
  parameterErrors,
  parameterLabel,
  parseExtraJson,
  seedParameters,
} from "@/features/tools/guardrailParameters"

function spec(
  overrides: Partial<GuardrailParameterSpec> &
    Pick<GuardrailParameterSpec, "name" | "type">,
): GuardrailParameterSpec {
  return { required: false, secret: false, ...overrides }
}

describe("seedParameters", () => {
  it("puts a stored value in its typed field and everything else in the raw editor", () => {
    const seeded = seedParameters([spec({ name: "policy", type: "string" })], {
      policy: "No personal data.",
      unmapped: { deep: [1] },
    })

    expect(seeded.values.policy).toBe("No personal data.")
    expect(JSON.parse(seeded.extraJson)).toEqual({ unmapped: { deep: [1] } })
  })

  it("leaves a field blank rather than prefilling the profile's own default", () => {
    const seeded = seedParameters(
      [spec({ name: "threshold", type: "number", default: 0.5 })],
      null,
    )

    // Prefilling would store an explicit value where the guardrails service's
    // own default was meant to apply, and the two stop tracking each other.
    expect(seeded.values.threshold).toBe("")
    expect(seeded.extraJson).toBe("")
  })

  it("renders a stored object parameter as JSON the operator can edit", () => {
    const seeded = seedParameters(
      [spec({ name: "entity_types", type: "json" })],
      { entity_types: ["email", "phone"] },
    )

    expect(JSON.parse(String(seeded.values.entity_types))).toEqual([
      "email",
      "phone",
    ])
  })
})

describe("parameterErrors", () => {
  const specs = [
    spec({ name: "policy", type: "string", required: true }),
    spec({ name: "threshold", type: "number" }),
    spec({ name: "count", type: "integer" }),
    spec({ name: "version", type: "enum", choices: ["v1", "v2"] }),
    spec({ name: "context", type: "json" }),
  ]

  it("says nothing about a form that is filled in", () => {
    expect(
      parameterErrors(specs, {
        policy: "No personal data.",
        threshold: "0.8",
        count: "3",
        version: "v2",
        context: "[]",
      }),
    ).toEqual({})
  })

  it("names each parameter that cannot be sent as typed", () => {
    const errors = parameterErrors(specs, {
      policy: "  ",
      threshold: "high",
      count: "1.5",
      version: "v3",
      context: "{oops",
    })

    expect(Object.keys(errors).sort()).toEqual([
      "context",
      "count",
      "policy",
      "threshold",
      "version",
    ])
  })

  it("leaves an optional parameter blank without complaint", () => {
    expect(parameterErrors(specs, { policy: "x" })).toEqual({})
  })
})

describe("buildValidateKwargs", () => {
  it("coerces each field back to its JSON-native type", () => {
    const kwargs = buildValidateKwargs(
      [
        spec({ name: "threshold", type: "number" }),
        spec({ name: "count", type: "integer" }),
        spec({ name: "context", type: "json" }),
        spec({ name: "policy", type: "string" }),
      ],
      { threshold: "0.8", count: "3", context: '["a"]', policy: "keep it" },
      "",
    )

    expect(kwargs).toEqual({
      threshold: 0.8,
      count: 3,
      context: ["a"],
      policy: "keep it",
    })
  })

  it("omits an optional box nobody ticked, so the profile's default stands", () => {
    expect(
      buildValidateKwargs(
        [spec({ name: "strict", type: "boolean", default: true })],
        { strict: false },
        "",
      ),
    ).toBeNull()
  })

  it("sends a required boolean either way, because it has no unset state", () => {
    expect(
      buildValidateKwargs(
        [spec({ name: "strict", type: "boolean", required: true })],
        { strict: false },
        "",
      ),
    ).toEqual({ strict: false })
  })

  it("lets a typed field win the key it shares with the raw editor", () => {
    expect(
      buildValidateKwargs(
        [spec({ name: "policy", type: "string" })],
        { policy: "from the field" },
        '{"policy": "from the editor", "other": 1}',
      ),
    ).toEqual({ policy: "from the field", other: 1 })
  })

  it("reports an entry that configures nothing as null, not an empty object", () => {
    expect(buildValidateKwargs([], {}, "")).toBeNull()
  })
})

describe("parseExtraJson", () => {
  it("reads blank as no parameters at all", () => {
    expect(parseExtraJson("   ")).toEqual({ value: {} })
  })

  it("refuses anything that is not a JSON object", () => {
    expect(parseExtraJson("[1, 2]").error).toBeDefined()
    expect(parseExtraJson("not json").error).toBeDefined()
    expect(parseExtraJson("null").error).toBeDefined()
  })
})

describe("parameterLabel", () => {
  it("reads a snake_case parameter name as a sentence", () => {
    expect(parameterLabel("comparison_text")).toBe("Comparison text")
    expect(parameterLabel("policy")).toBe("Policy")
  })
})
