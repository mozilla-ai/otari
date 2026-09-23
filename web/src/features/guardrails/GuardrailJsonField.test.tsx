import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it } from "vitest"

import type { GuardrailParameterSpec } from "@/client"
import { GuardrailJsonField } from "@/features/guardrails/GuardrailJsonField"

function spec(name: string): GuardrailParameterSpec {
  return { name, type: "json", required: false, secret: false, storable: true }
}

/** The field with the value it reports back, as the real form holds it. */
function Harness({
  parameter,
  guardrailName,
  initial = "",
}: {
  parameter: string
  guardrailName: string
  initial?: string
}) {
  const [value, setValue] = useState(initial)
  return (
    <>
      <GuardrailJsonField
        spec={spec(parameter)}
        guardrailName={guardrailName}
        value={value}
        error={undefined}
        disabled={false}
        description="What it does."
        onChange={setValue}
      />
      <output data-testid="value">{value}</output>
    </>
  )
}

const held = () => screen.getByTestId("value").textContent ?? ""

describe("a field of named switches", () => {
  it("offers the detections the guardrail documents", async () => {
    render(<Harness parameter="detection_config" guardrailName="alinia" />)

    expect(
      screen.getByRole("checkbox", { name: "Security" }),
    ).toBeInTheDocument()
    expect(screen.getByRole("checkbox", { name: "Safety" })).toBeInTheDocument()
    expect(
      screen.getByText("Prompt injection and data exfiltration."),
    ).toBeInTheDocument()
  })

  it("draws no line under the whole group, not even the catalog's paragraph", () => {
    // Each box keeps its own explanation; a sentence saying what the group is
    // repeats the label above it.
    render(
      <GuardrailJsonField
        spec={{
          name: "evaluators",
          type: "json",
          required: true,
          secret: false,
          storable: true,
          description: "The catalog's long paragraph on the list to pass.",
        }}
        guardrailName="patronus"
        value=""
        error={undefined}
        disabled={false}
        description="The catalog's long paragraph on the list to pass."
        onChange={() => {}}
      />,
    )

    expect(
      screen.getByText("Patronus's own hallucination model."),
    ).toBeVisible()
    expect(screen.queryByText(/Which checks Patronus runs/)).toBeNull()
    expect(screen.queryByText(/long paragraph/)).toBeNull()
  })

  it("writes the wire key when one is switched on", async () => {
    const user = userEvent.setup()
    render(<Harness parameter="detection_config" guardrailName="alinia" />)

    await user.click(screen.getByRole("checkbox", { name: "Security" }))

    expect(JSON.parse(held())).toEqual({ security: true })
  })

  it("sends a detector's own settings shape rather than a bare true", async () => {
    const user = userEvent.setup()
    render(<Harness parameter="detectors" guardrailName="watsonx_guardian" />)

    await user.click(
      screen.getByRole("checkbox", { name: "Hate, abuse and profanity" }),
    )

    expect(JSON.parse(held())).toEqual({ hap: {} })
  })

  it("shows a key it has never heard of, and lets it be corrected", async () => {
    // The whole reason a list that will go stale is safe to ship: a detection
    // the vendor added is visible, editable and removable.
    render(
      <Harness
        parameter="detection_config"
        guardrailName="alinia"
        initial='{"invented_later": true}'
      />,
    )

    expect(screen.getByLabelText("Other detection 1")).toHaveValue(
      "invented_later",
    )
    expect(
      screen.getByRole("button", { name: "Remove invented_later" }),
    ).toBeInTheDocument()
  })

  it("lets an operator name a detection the registry does not have", async () => {
    // Alinia is offered for personal data and no source names the detection
    // that covers it, so a closed set of switches is a dead end.
    const user = userEvent.setup()
    render(<Harness parameter="detection_config" guardrailName="alinia" />)

    await user.click(
      screen.getByRole("button", { name: /Add another detection/ }),
    )
    await user.type(screen.getByLabelText("Other detection 1"), "pii")

    expect(JSON.parse(held())).toEqual({ pii: true })
  })

  it("removes a switch's key rather than setting it false", async () => {
    const user = userEvent.setup()
    render(
      <Harness
        parameter="detection_config"
        guardrailName="alinia"
        initial='{"security": true, "safety": true}'
      />,
    )

    await user.click(screen.getByRole("checkbox", { name: "Safety" }))

    expect(JSON.parse(held())).toEqual({ security: true })
  })
})

describe("a field of free key-value pairs", () => {
  it("adds and fills a row without anyone typing a brace", async () => {
    const user = userEvent.setup()
    render(<Harness parameter="metadata" guardrailName="lakera_guard" />)

    await user.click(screen.getByRole("button", { name: "Add entry" }))
    await user.type(screen.getByLabelText("Key 1"), "user_id")
    await user.type(screen.getByLabelText(/^Value for/), "u-42")

    expect(JSON.parse(held())).toEqual({ user_id: "u-42" })
  })

  it("reads a number as a number and text as text", async () => {
    const user = userEvent.setup()
    render(
      <Harness
        parameter="metadata"
        guardrailName="lakera_guard"
        initial='{"retries": "", "app": ""}'
      />,
    )

    await user.type(screen.getByLabelText("Value for retries"), "3")
    await user.type(screen.getByLabelText("Value for app"), "checkout")

    expect(JSON.parse(held())).toEqual({ retries: 3, app: "checkout" })
  })
})

describe("a field of plain text items", () => {
  it("adds one row per document", async () => {
    const user = userEvent.setup()
    render(
      <Harness
        parameter="blocklist_names"
        guardrailName="azure_content_safety"
      />,
    )

    await user.click(screen.getByRole("button", { name: "Add blocklist" }))
    await user.type(screen.getByLabelText("Blocklist 1"), "house-terms")

    expect(JSON.parse(held())).toEqual(["house-terms"])
  })
})

describe("the JSON view", () => {
  it("shows what the controls built", async () => {
    const user = userEvent.setup()
    render(<Harness parameter="detection_config" guardrailName="alinia" />)

    await user.click(screen.getByRole("checkbox", { name: "Security" }))
    await user.click(screen.getByRole("radio", { name: "JSON" }))

    expect(screen.getByRole("textbox")).toHaveValue('{\n  "security": true\n}')
  })

  it("leaves a value nobody touched exactly as it was stored", async () => {
    // Formatting it on arrival would make every open of an existing definition
    // look like an edit, and `useDirtySnapshot` would guard on the way out.
    const user = userEvent.setup()
    render(
      <Harness
        parameter="detection_config"
        guardrailName="alinia"
        initial='{"security": true}'
      />,
    )

    await user.click(screen.getByRole("radio", { name: "JSON" }))

    expect(screen.getByRole("textbox")).toHaveValue('{"security": true}')
  })

  it("is the only view for a value the controls cannot hold", () => {
    // A detection config given as a registered configuration id is a string,
    // which no switch represents. It is left alone rather than rewritten.
    render(
      <Harness
        parameter="detection_config"
        guardrailName="alinia"
        initial='"cfg-123"'
      />,
    )

    expect(screen.getByText(/cannot be shown as fields/)).toBeInTheDocument()
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument()
  })

  it("says so when the text is not JSON at all, and keeps it", async () => {
    const user = userEvent.setup()
    render(
      <Harness
        parameter="metadata"
        guardrailName="lakera_guard"
        initial="{not json"
      />,
    )

    expect(screen.getByText("Not valid JSON.")).toBeInTheDocument()
    await user.click(screen.getByRole("radio", { name: "JSON" }))
    expect(screen.getByRole("textbox")).toHaveValue("{not json")
  })

  it("is all a field the registry does not name gets", () => {
    render(<Harness parameter="invented_later" guardrailName="alinia" />)

    expect(screen.getByRole("textbox")).toBeInTheDocument()
    expect(
      screen.queryByRole("radio", { name: "Fields" }),
    ).not.toBeInTheDocument()
  })
})

describe("an operation the registry cannot map", () => {
  it("says so rather than leaving the operator at a dead end", () => {
    render(
      <GuardrailJsonField
        spec={spec("detection_config")}
        guardrailName="alinia"
        operation="pii"
        value=""
        error={undefined}
        disabled={false}
        description="What it does."
        onChange={() => undefined}
      />,
    )

    expect(screen.getByText(/cannot say which detection/)).toBeInTheDocument()
  })

  it("stays quiet for an operation it can map", () => {
    render(
      <GuardrailJsonField
        spec={spec("detection_config")}
        guardrailName="alinia"
        operation="prompt_injection"
        value=""
        error={undefined}
        disabled={false}
        description="What it does."
        onChange={() => undefined}
      />,
    )

    expect(screen.queryByText(/cannot say which/)).not.toBeInTheDocument()
  })

  it("stops saying it once something is set", () => {
    render(
      <GuardrailJsonField
        spec={spec("detection_config")}
        guardrailName="alinia"
        operation="pii"
        value='{"pii": true}'
        error={undefined}
        disabled={false}
        description="What it does."
        onChange={() => undefined}
      />,
    )

    expect(screen.queryByText(/cannot say which/)).not.toBeInTheDocument()
  })
})
