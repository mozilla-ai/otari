import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import type { GuardrailParameterSpec } from "@/client"
import { GuardrailParameterFields } from "@/features/guardrails/GuardrailParameterFields"

function spec(
  overrides: Partial<GuardrailParameterSpec> &
    Pick<GuardrailParameterSpec, "name">,
): GuardrailParameterSpec {
  return {
    type: "string",
    required: true,
    secret: false,
    storable: true,
    ...overrides,
  }
}

// Alinia's order as the layout hands it over: the key, its JSON, the endpoint.
const ALINIA = [
  spec({ name: "api_key", secret: true }),
  spec({ name: "detection_config", type: "json" }),
  spec({ name: "endpoint" }),
]

function labelsInOrder() {
  // The plain JSON box labels itself "... (JSON)", which is not what is tested.
  return [
    ...screen.getAllByText(/^(Api key|Endpoint|Detection config)( \(JSON\))?$/),
  ].map((node) => node.textContent?.replace(" (JSON)", ""))
}

describe("GuardrailParameterFields", () => {
  it("pairs the plain fields and puts a full-width JSON field after them", () => {
    // The JSON field spans both columns, so drawn between the key and the
    // endpoint it leaves each of them alone on its own row.
    render(
      <GuardrailParameterFields
        specs={ALINIA}
        scopeName="alinia"
        values={{}}
        errors={{}}
        disabled={false}
        guardrailName="alinia"
        onChange={() => {}}
      />,
    )

    expect(labelsInOrder()).toEqual(["Api key", "Endpoint", "Detection config"])
  })

  it("keeps the given order where no JSON field is drawn full width", () => {
    render(
      <GuardrailParameterFields
        specs={ALINIA}
        scopeName="alinia"
        values={{}}
        errors={{}}
        disabled={false}
        onChange={() => {}}
      />,
    )

    expect(labelsInOrder()).toEqual(["Api key", "Detection config", "Endpoint"])
  })
})
