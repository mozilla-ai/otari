import { describe, expect, it } from "vitest"

import type { BuiltInGuardrailSpec } from "@/client"
import {
  findGuardrail,
  guardrailDocsHref,
  guardrailOptions,
  guardrailsForOperation,
  operationLabel,
  operationOptions,
  suggestName,
} from "@/features/tools/guardrailOperations"

function guardrail(
  overrides: Partial<BuiltInGuardrailSpec> &
    Pick<BuiltInGuardrailSpec, "guardrail_name" | "display_name">,
): BuiltInGuardrailSpec {
  return {
    backend: "hosted_api",
    categories: ["prompt_injection"],
    default_license: "proprietary",
    description: "",
    multilingual: false,
    multimodal: false,
    output_shapes: ["binary"],
    primary_category: "prompt_injection",
    requires_api_key: true,
    stages: ["input"],
    supports_batch: false,
    vendor: "Acme",
    ...overrides,
  }
}

// Shaped as the real catalog is: Lakera's headline is prompt injection but it
// also detects PII, and Bedrock's headline is content safety. That difference is
// the whole reason this module reads `categories`.
const LAKERA = guardrail({
  guardrail_name: "lakera_guard",
  display_name: "Lakera Guard",
  vendor: "Lakera",
  description: "Checks text against Lakera Guard.",
  primary_category: "prompt_injection",
  categories: ["content_safety", "pii", "prompt_injection"],
})
const BEDROCK = guardrail({
  guardrail_name: "bedrock_guardrails",
  display_name: "Bedrock Guardrails",
  vendor: "Amazon",
  primary_category: "content_safety",
  categories: ["content_safety", "off_topic", "pii"],
})
const SHIELDS = guardrail({
  guardrail_name: "azure_prompt_shields",
  display_name: "Azure Prompt Shields",
  vendor: "Microsoft",
  primary_category: "prompt_injection",
  categories: ["prompt_injection"],
})
// Declares PII and has a detection vocabulary that names no key for it, which
// is the real Alinia.
const ALINIA = guardrail({
  guardrail_name: "alinia",
  display_name: "Alinia",
  vendor: "Alinia AI",
  primary_category: "content_safety",
  categories: ["content_safety", "pii", "prompt_injection"],
  create_parameters: [
    {
      name: "detection_config",
      type: "json",
      required: true,
      secret: false,
      storable: true,
    },
  ],
})

const CATALOG = [LAKERA, BEDROCK, SHIELDS]
const WITH_ALINIA = [...CATALOG, ALINIA]

describe("operationOptions", () => {
  it("offers every category the catalog declares, not only the headline ones", () => {
    // `off_topic` and `pii` are nobody's `primary_category` here, so reading
    // that field instead would drop both.
    expect(operationOptions(CATALOG).map((option) => option.value)).toEqual([
      "content_safety",
      "off_topic",
      "pii",
      "prompt_injection",
    ])
  })

  it("leaves out an operation no installed guardrail can do", () => {
    const values = operationOptions(CATALOG).map((option) => option.value)
    expect(values).not.toContain("tool_use")
  })

  it("offers a category this bundle predates, since the gateway may be newer", () => {
    const future = guardrail({
      guardrail_name: "future",
      display_name: "Future",
      categories: ["something_new"],
      primary_category: "prompt_injection",
    })

    const options = operationOptions([future])
    expect(options).toContainEqual({
      value: "something_new",
      label: "Something new",
    })
  })

  it("sorts, so an added guardrail does not reshuffle the list", () => {
    const values = operationOptions([...CATALOG].reverse()).map(
      (option) => option.value,
    )
    expect(values).toEqual([...values].sort())
  })

  it("has nothing to offer for an empty catalog", () => {
    expect(operationOptions([])).toEqual([])
  })
})

describe("guardrailsForOperation", () => {
  it("includes a guardrail whose headline category is a different one", () => {
    const names = guardrailsForOperation(CATALOG, "pii").map(
      (spec) => spec.guardrail_name,
    )
    expect(names).toEqual(["bedrock_guardrails", "lakera_guard"])
  })

  it("orders by the name the picker shows", () => {
    const names = guardrailsForOperation(CATALOG, "prompt_injection").map(
      (spec) => spec.display_name,
    )
    expect(names).toEqual(["Azure Prompt Shields", "Lakera Guard"])
  })

  it("offers nothing until an operation is chosen", () => {
    expect(guardrailsForOperation(CATALOG, "")).toEqual([])
  })
})

describe("guardrailOptions", () => {
  it("names the vendor beside the guardrail, since two can do one job", () => {
    expect(guardrailOptions(CATALOG, "prompt_injection")).toEqual([
      {
        value: "azure_prompt_shields",
        label: "Azure Prompt Shields · Microsoft",
      },
      { value: "lakera_guard", label: "Lakera Guard · Lakera" },
    ])
  })
})

describe("operationLabel", () => {
  it("reads a wire name as a sentence", () => {
    expect(operationLabel("prompt_injection")).toBe("Prompt injection")
  })

  it("spells out an acronym the mechanical rule cannot expand", () => {
    expect(operationLabel("pii")).toBe("Personally Identifiable Information")
  })

  it("still renders a category this bundle predates", () => {
    // The map is formatting, not a list of what exists: a name it has never
    // seen falls through rather than disappearing.
    expect(operationLabel("something_new")).toBe("Something new")
  })
})

describe("findGuardrail", () => {
  it("finds one by the name a stored row selects", () => {
    expect(findGuardrail(CATALOG, "lakera_guard")).toBe(LAKERA)
  })

  it("is undefined for a class this build no longer ships", () => {
    expect(findGuardrail(CATALOG, "retired_guardrail")).toBeUndefined()
  })
})

describe("suggestName", () => {
  it("names a definition after what it checks", () => {
    expect(suggestName("prompt_injection", [])).toBe("prompt-injection")
  })

  it("steps past a name already in use", () => {
    expect(suggestName("pii", ["pii", "pii-2"])).toBe("pii-3")
  })

  it("suggests nothing before an operation is chosen", () => {
    expect(suggestName("", [])).toBe("")
  })
})

describe("guardrailDocsHref", () => {
  const REFERENCE = "https://docs.mozilla.ai/any-guardrail/api-reference/index"

  // Each of these was read off any-guardrail's own published index, so the
  // derivation is checked against the site rather than against itself.
  it.each([
    [LAKERA, `${REFERENCE}/prompt-injection/lakera-guard`],
    [BEDROCK, `${REFERENCE}/content-safety/bedrock-guardrails`],
    [SHIELDS, `${REFERENCE}/prompt-injection/azure-prompt-shields`],
    [
      guardrail({
        guardrail_name: "any_llm",
        display_name: "AnyLlm",
        primary_category: "general_judge",
        categories: ["general_judge"],
      }),
      `${REFERENCE}/general-judge/any-llm`,
    ],
    [
      guardrail({
        guardrail_name: "watsonx_guardian",
        display_name: "watsonx Guardian",
        primary_category: "content_safety",
        categories: ["content_safety"],
      }),
      `${REFERENCE}/content-safety/watsonx-guardian`,
    ],
  ])("points $display_name at its reference page", (spec, href) => {
    expect(guardrailDocsHref(spec)).toBe(href)
  })

  it("files a guardrail under its headline category, not every one it detects", () => {
    // The one place this module reads `primary_category`: a picker asks who can
    // do a job, but a documentation site files each page once.
    expect(guardrailDocsHref(LAKERA)).toContain("/prompt-injection/")
    expect(guardrailDocsHref(LAKERA)).not.toContain("/pii/")
  })
})

describe("a guardrail that declares a job it cannot be asked for", () => {
  it("is not offered for that job", () => {
    // Alinia detects personal data, and neither any-guardrail nor the vendor
    // walkthrough names the detection that does it, so choosing it here would
    // reach a form with no way to ask.
    const names = guardrailsForOperation(WITH_ALINIA, "pii").map(
      (spec) => spec.guardrail_name,
    )

    expect(names).not.toContain("alinia")
    // Lakera declares PII too and is told what to check by its API key alone,
    // so it has no vocabulary to be missing from and stays.
    expect(names).toContain("lakera_guard")
  })

  it("is still offered for the jobs it can be asked for", () => {
    const names = guardrailsForOperation(WITH_ALINIA, "prompt_injection").map(
      (spec) => spec.guardrail_name,
    )

    expect(names).toContain("alinia")
  })

  it("does not take the operation away while something else can do it", () => {
    const values = operationOptions(WITH_ALINIA).map((option) => option.value)

    expect(values).toContain("pii")
  })
})

describe("a guardrail whose documented key outruns its categories", () => {
  // watsonx Guardian documents a `pii` detector and its metadata lists no `pii`
  // category. A key that is documented is better evidence than a category that
  // is missing.
  const WATSONX = guardrail({
    guardrail_name: "watsonx_guardian",
    display_name: "watsonx Guardian",
    vendor: "IBM",
    primary_category: "content_safety",
    categories: ["bias", "content_safety", "toxicity"],
    create_parameters: [
      {
        name: "detectors",
        type: "json",
        required: false,
        secret: false,
        storable: true,
      },
    ],
  })

  it("is offered for the job its key names", () => {
    const names = guardrailsForOperation([WATSONX, LAKERA], "pii").map(
      (spec) => spec.guardrail_name,
    )

    expect(names).toContain("watsonx_guardian")
  })

  it("puts that job on the first control", () => {
    expect(operationOptions([WATSONX]).map((option) => option.value)).toContain(
      "pii",
    )
  })
})

describe("the jobs the first control does not offer", () => {
  it("leaves out general judge, which is not a detection", () => {
    const judge = guardrail({
      guardrail_name: "any_llm",
      display_name: "AnyLlm",
      primary_category: "general_judge",
      categories: ["general_judge"],
    })

    expect(operationOptions([judge, LAKERA]).map((o) => o.value)).not.toContain(
      "general_judge",
    )
    expect(guardrailsForOperation([judge], "general_judge")).toEqual([])
  })
})
