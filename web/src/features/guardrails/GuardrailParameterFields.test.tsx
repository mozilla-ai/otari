import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

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

  describe("an enum parameter", () => {
    const MODE = [
      spec({
        name: "mode",
        type: "enum",
        required: false,
        choices: ["strict", "lenient"],
      }),
    ]

    function renderMode(value: string, onChange = vi.fn()) {
      render(
        <GuardrailParameterFields
          specs={MODE}
          scopeName="alinia"
          values={{ mode: value }}
          errors={{}}
          disabled={false}
          onChange={onChange}
        />,
      )
      return onChange
    }

    it("names an unset value by its label, not as a blank", () => {
      renderMode("")

      expect(screen.getByRole("button", { name: /Mode/ })).toHaveTextContent(
        "Default",
      )
    })

    it("reports a choice as itself and the blank entry as an empty value", async () => {
      const user = userEvent.setup()
      const onChange = renderMode("strict")

      await user.click(screen.getByRole("button", { name: /Mode/ }))
      await user.click(screen.getByRole("option", { name: "lenient" }))
      expect(onChange).toHaveBeenLastCalledWith("mode", "lenient")

      await user.click(screen.getByRole("button", { name: /Mode/ }))
      await user.click(screen.getByRole("option", { name: "Default" }))
      expect(onChange).toHaveBeenLastCalledWith("mode", "")
    })

    it("never offers a required enum's prompt as a choice", async () => {
      const user = userEvent.setup()
      render(
        <GuardrailParameterFields
          specs={[
            spec({
              name: "mode",
              type: "enum",
              choices: ["strict", "lenient"],
            }),
          ]}
          scopeName="alinia"
          values={{}}
          errors={{}}
          disabled={false}
          onChange={() => {}}
        />,
      )

      const trigger = screen.getByRole("button", { name: /Mode/ })
      expect(trigger).toHaveTextContent("Choose a value")
      await user.click(trigger)
      expect(
        screen.getAllByRole("option").map((option) => option.textContent),
      ).toEqual(["strict", "lenient"])
    })

    it("shows an error in place of the description, not beside it", () => {
      render(
        <GuardrailParameterFields
          specs={[
            spec({
              name: "mode",
              type: "enum",
              required: true,
              choices: ["strict"],
              description: "How hard it checks.",
            }),
          ]}
          scopeName="alinia"
          values={{}}
          errors={{ mode: "Choose a value." }}
          disabled={false}
          onChange={() => {}}
        />,
      )

      expect(screen.getByText("Choose a value.")).toBeVisible()
      expect(screen.queryByText("How hard it checks.")).toBeNull()
    })
  })
})
