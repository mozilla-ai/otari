import { describe, expect, it } from "vitest"

import type { BuiltInGuardrailSpec, GuardrailParameterSpec } from "@/client"
import { splitParameters } from "@/features/tools/guardrailFieldSplit"

function param(
  overrides: Partial<GuardrailParameterSpec> &
    Pick<GuardrailParameterSpec, "name" | "type">,
): GuardrailParameterSpec {
  return { required: false, secret: false, storable: true, ...overrides }
}

function guardrail(
  overrides: Partial<BuiltInGuardrailSpec> &
    Pick<BuiltInGuardrailSpec, "guardrail_name">,
): BuiltInGuardrailSpec {
  return {
    display_name: "Test",
    description: "",
    vendor: "Test",
    backend: "hosted_api",
    primary_category: "content_safety",
    categories: ["content_safety"],
    stages: ["input"],
    output_shapes: ["binary"],
    default_license: "proprietary",
    requires_api_key: true,
    multilingual: false,
    multimodal: false,
    supports_batch: false,
    ...overrides,
  }
}

const names = (entries: GuardrailParameterSpec[]) =>
  entries.map((entry) => entry.name)

describe("splitParameters", () => {
  it("shows an argument with no default", () => {
    const spec = guardrail({ guardrail_name: "lakera_guard" })
    const { decisions, extras } = splitParameters(spec, [
      param({ name: "api_key", type: "string", required: true, secret: true }),
      param({ name: "endpoint", type: "string", default: "https://x" }),
    ])

    expect(names(decisions)).toEqual(["api_key"])
    expect(names(extras)).toEqual(["endpoint"])
  })

  it("shows a credential even when nothing marks it required", () => {
    // Bedrock's AWS keys. Reading `required` alone leaves the first thing
    // anyone fills in behind an accordion.
    const spec = guardrail({ guardrail_name: "bedrock_guardrails" })
    const { decisions } = splitParameters(spec, [
      param({ name: "aws_access_key_id", type: "string", secret: true }),
      param({ name: "guardrail_version", type: "string", default: "DRAFT" }),
    ])

    expect(names(decisions)).toEqual(["aws_access_key_id"])
  })

  it("shows every member of a one-of requirement group", () => {
    // The whole of watsonx Guardian: nothing on it is `required`, because its
    // credentials sit in groups instead.
    const spec = guardrail({
      guardrail_name: "watsonx_guardian",
      requirement_groups: [
        {
          description: "one of",
          parameters: ["api_key", "api_client"],
          env_vars: ["WATSONX_APIKEY"],
        },
        {
          description: "one of",
          parameters: ["project_id", "space_id", "api_client"],
          env_vars: [],
        },
      ],
    })
    const { decisions, extras } = splitParameters(spec, [
      param({ name: "api_key", type: "string", secret: true }),
      param({ name: "url", type: "string" }),
      param({ name: "project_id", type: "string" }),
      param({ name: "space_id", type: "string" }),
    ])

    expect(names(decisions)).toEqual(["api_key", "project_id", "space_id"])
    expect(names(extras)).toEqual(["url"])
  })

  it("shows the field the guardrail is actually about", () => {
    // watsonx's detectors are `required: false` with no default, and hiding
    // them would hide the point of configuring watsonx at all.
    const spec = guardrail({ guardrail_name: "watsonx_guardian" })
    const { decisions } = splitParameters(spec, [
      param({ name: "detectors", type: "json" }),
      param({ name: "unrelated", type: "json" }),
    ])

    expect(names(decisions)).toEqual(["detectors"])
  })

  it("folds away an argument that cannot be stored", () => {
    // There is no value to give it, so it is a thing to find rather than face.
    const spec = guardrail({ guardrail_name: "bedrock_guardrails" })
    const { decisions, extras } = splitParameters(spec, [
      param({
        name: "boto3_session",
        type: "json",
        secret: true,
        storable: false,
      }),
    ])

    expect(decisions).toEqual([])
    expect(names(extras)).toEqual(["boto3_session"])
  })

  it("folds away observability and tuning", () => {
    const spec = guardrail({ guardrail_name: "patronus" })
    const { extras } = splitParameters(spec, [
      param({ name: "tags", type: "json" }),
      param({ name: "success_strategy", type: "enum", default: "all_pass" }),
    ])

    expect(names(extras)).toEqual(["tags", "success_strategy"])
  })

  it("shows everything for a guardrail the catalog does not describe", () => {
    // A row whose class this build no longer ships: nothing can be called
    // advanced when nothing is known, so nothing is hidden.
    const { decisions, extras } = splitParameters(undefined, [
      param({ name: "policy", type: "string", required: true }),
    ])

    expect(names(decisions)).toEqual(["policy"])
    expect(extras).toEqual([])
  })
})
