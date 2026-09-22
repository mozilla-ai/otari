import { describe, expect, it } from "vitest"

import type {
  BuiltInGuardrailCatalog,
  BuiltInGuardrailSpec,
  GuardrailParameterSpec,
} from "@/client"
import {
  checkLabel,
  createFieldLayout,
  definableGuardrails,
  guardrailChecks,
  guardrailsForCheck,
  suggestDefinitionName,
} from "@/features/guardrails/builtInCatalog"

function parameter(
  overrides: Partial<GuardrailParameterSpec> &
    Pick<GuardrailParameterSpec, "name">,
): GuardrailParameterSpec {
  return {
    type: "string",
    required: false,
    secret: false,
    storable: true,
    ...overrides,
  }
}

function guardrail(
  overrides: Partial<BuiltInGuardrailSpec> &
    Pick<BuiltInGuardrailSpec, "guardrail_name">,
): BuiltInGuardrailSpec {
  return {
    backend: "hosted_api",
    categories: [],
    default_license: "",
    description: "",
    display_name: overrides.guardrail_name,
    multilingual: false,
    multimodal: false,
    output_shapes: [],
    primary_category: "content_safety",
    requires_api_key: true,
    stages: [],
    supports_batch: false,
    ...overrides,
  } as BuiltInGuardrailSpec
}

// The shape the installed library reports, trimmed to what these rules read.
const catalog: BuiltInGuardrailCatalog = {
  guardrails: [
    guardrail({
      guardrail_name: "lakera_guard",
      display_name: "Lakera Guard",
      primary_category: "prompt_injection",
      categories: ["content_safety", "pii", "prompt_injection"],
    }),
    guardrail({
      guardrail_name: "alinia",
      display_name: "Alinia",
      primary_category: "content_safety",
      categories: ["content_safety", "prompt_injection", "toxicity"],
    }),
    guardrail({
      guardrail_name: "patronus",
      display_name: "Patronus",
      primary_category: "general_judge",
      categories: ["general_judge", "hallucination", "prompt_injection"],
    }),
    guardrail({
      guardrail_name: "any_llm",
      display_name: "Any LLM",
      primary_category: "general_judge",
      categories: ["general_judge", "bias"],
    }),
  ],
}

describe("definableGuardrails", () => {
  it("leaves out any_llm, which the store refuses", () => {
    // The gateway lists it, correctly, and refuses to store it for an
    // organization because it would spend the operator's own key.
    expect(definableGuardrails(catalog).map((g) => g.guardrail_name)).toEqual([
      "lakera_guard",
      "alinia",
      "patronus",
    ])
  })

  it("answers nothing for a catalog that has not loaded", () => {
    expect(definableGuardrails(undefined)).toEqual([])
  })
})

describe("guardrailChecks", () => {
  it("is the union of every definable guardrail's categories", () => {
    // Not `primary_category`: that would hide Alinia and Patronus from
    // prompt injection, which both check.
    expect(guardrailChecks(catalog)).toEqual([
      "content_safety",
      "hallucination",
      "pii",
      "prompt_injection",
      "toxicity",
    ])
  })

  it("drops general_judge even though Patronus still lists it", () => {
    expect(guardrailChecks(catalog)).not.toContain("general_judge")
  })

  it("takes nothing from a guardrail the store refuses", () => {
    // `bias` is only any_llm's here.
    expect(guardrailChecks(catalog)).not.toContain("bias")
  })
})

describe("checkLabel", () => {
  it("spells pii out in full", () => {
    expect(checkLabel("pii")).toBe("Personally identifiable information")
  })

  it("reads a check it does not know as a sentence", () => {
    expect(checkLabel("tool_use")).toBe("Tool use")
    expect(checkLabel("something_new")).toBe("Something new")
  })
})

describe("guardrailsForCheck", () => {
  it("lists every definable guardrail that checks it, by name", () => {
    expect(
      guardrailsForCheck(catalog, "prompt_injection").map(
        (g) => g.display_name,
      ),
    ).toEqual(["Alinia", "Lakera Guard", "Patronus"])
  })

  it("lists nothing for no check", () => {
    expect(guardrailsForCheck(catalog, "")).toEqual([])
  })
})

describe("createFieldLayout", () => {
  it("puts secrets first, then either/or members, then the other required ones", () => {
    const layout = createFieldLayout(
      guardrail({
        guardrail_name: "g",
        create_parameters: [
          parameter({ name: "model", required: true }),
          parameter({ name: "timeout" }),
          parameter({ name: "project_id" }),
          parameter({ name: "api_key", secret: true }),
        ],
        requirement_groups: [
          { description: "", env_vars: [], parameters: ["project_id"] },
        ],
      }),
    )

    expect(layout.fields.map((p) => p.name)).toEqual([
      "api_key",
      "project_id",
      "model",
    ])
    expect(layout.advanced.map((p) => p.name)).toEqual(["timeout"])
  })

  it("keeps watsonx Guardian's key out of Advanced although nothing is required", () => {
    // Its credentials live in either/or groups, so a required-first rule would
    // hide the API key behind the accordion.
    const layout = createFieldLayout(
      guardrail({
        guardrail_name: "watsonx_guardian",
        create_parameters: [
          parameter({ name: "url" }),
          parameter({ name: "project_id" }),
          parameter({ name: "space_id" }),
          parameter({ name: "api_key", secret: true }),
          parameter({ name: "detectors", type: "json" }),
          parameter({
            name: "api_client",
            type: "json",
            secret: true,
            storable: false,
          }),
        ],
        // The library's own groups: each credential or its client object.
        requirement_groups: [
          {
            description: "",
            env_vars: [],
            parameters: ["api_key", "api_client"],
          },
          { description: "", env_vars: [], parameters: ["url", "api_client"] },
          {
            description: "",
            env_vars: [],
            parameters: ["project_id", "space_id", "api_client"],
          },
        ],
      }),
    )

    expect(layout.fields.map((p) => p.name)).toEqual([
      "api_key",
      "url",
      "project_id",
      "space_id",
    ])
    expect(layout.advanced.map((p) => p.name)).toEqual(["detectors"])
    expect(layout.unstorable).toEqual(["api_client"])
  })

  it("draws no control for an unstorable argument, and names it", () => {
    const layout = createFieldLayout(
      guardrail({
        guardrail_name: "bedrock_guardrails",
        create_parameters: [
          parameter({ name: "boto3_session", secret: true, storable: false }),
          parameter({ name: "aws_access_key_id", secret: true }),
        ],
      }),
    )

    expect(layout.fields.map((p) => p.name)).toEqual(["aws_access_key_id"])
    expect(layout.advanced).toEqual([])
    expect(layout.unstorable).toEqual(["boto3_session"])
  })
})

describe("suggestDefinitionName", () => {
  it("suggests the guardrail's name in kebab case", () => {
    expect(suggestDefinitionName("lakera_guard", [])).toBe("lakera-guard")
  })

  it("steps past a name the organization already uses", () => {
    expect(
      suggestDefinitionName("lakera_guard", ["lakera-guard", "lakera-guard-2"]),
    ).toBe("lakera-guard-3")
  })
})
