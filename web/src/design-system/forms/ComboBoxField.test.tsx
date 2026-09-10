import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it } from "vitest"

import {
  ComboBoxField,
  type ComboBoxOption,
} from "@/design-system/forms/ComboBoxField"

const OPTIONS: ComboBoxOption[] = [
  { value: "openai:gpt-4o", label: "openai:gpt-4o" },
  {
    value: "anthropic:claude-sonnet-4-5",
    label: "anthropic:claude-sonnet-4-5",
  },
]

/** The field is controlled, so a test that types into it has to hold the text. */
function Harness({
  options = OPTIONS,
  onValue,
  ...rest
}: {
  options?: ComboBoxOption[]
  onValue?: (value: string) => void
} & Partial<Parameters<typeof ComboBoxField>[0]>) {
  const [value, setValue] = useState("")
  return (
    <ComboBoxField
      label="Serves"
      {...rest}
      value={value}
      options={options}
      onChange={(next) => {
        setValue(next)
        onValue?.(next)
      }}
    />
  )
}

describe("ComboBoxField", () => {
  it("reports a picked option once, as its value rather than its label", async () => {
    const seen: string[] = []
    render(
      <Harness
        options={[{ value: "018f-0001", label: "Ada Lovelace" }]}
        onValue={(value) => seen.push(value)}
      />,
    )

    await userEvent.click(screen.getByRole("combobox", { name: "Serves" }))
    await userEvent.click(
      await screen.findByRole("option", { name: "Ada Lovelace" }),
    )

    // One report, not two. react-aria writes the row's display text into the
    // input after reporting the selection; forwarding that echo would hand the
    // caller a label after a key, and a label does not identify a row (two rows
    // may share one, and one row's label may be another row's value).
    expect(seen).toEqual(["018f-0001"])
  })

  it("still reports text the operator edits after picking a row", async () => {
    const seen: string[] = []
    render(
      <Harness
        allowsCustomValue
        options={[{ value: "018f-0001", label: "Ada Lovelace" }]}
        onValue={(value) => seen.push(value)}
      />,
    )

    const field = screen.getByRole("combobox", { name: "Serves" })
    await userEvent.click(field)
    await userEvent.click(
      await screen.findByRole("option", { name: "Ada Lovelace" }),
    )
    await userEvent.type(field, "!")

    // Only the one echo is swallowed. Anything typed afterwards is the
    // operator's, so a field that went quiet after a pick would be a worse bug
    // than the one the swallowing fixes.
    expect(seen.at(-1)).toBe("018f-0001!")
  })

  it("reports free text when the caller allows it", async () => {
    const seen: string[] = []
    render(<Harness allowsCustomValue onValue={(value) => seen.push(value)} />)

    await userEvent.type(
      screen.getByRole("combobox", { name: "Serves" }),
      "groq:llama",
    )

    // Discovery is a shortcut, not a whitelist, so what was typed stands.
    expect(seen.at(-1)).toBe("groq:llama")
  })

  // The defect both issues reported: the popover opened (the chevron pointed up)
  // with nothing in it and nothing saying why.
  it("says why the menu is empty when the source has nothing to offer", async () => {
    render(
      <Harness
        options={[]}
        isSourceEmpty
        emptyMessage="No models discovered yet. Add a provider credential."
        noMatchesMessage="No model matches."
      />,
    )

    await userEvent.click(screen.getByRole("combobox", { name: "Serves" }))

    expect(
      await screen.findByText(/No models discovered yet/),
    ).toBeInTheDocument()
    expect(screen.queryByText("No model matches.")).toBeNull()
  })

  it("says the other sentence when the source has options and the query matched none", async () => {
    render(
      <Harness
        options={[]}
        isSourceEmpty={false}
        emptyMessage="No models discovered yet."
        noMatchesMessage="No model matches. Type a selector to use it anyway."
      />,
    )

    await userEvent.click(screen.getByRole("combobox", { name: "Serves" }))

    // Two facts, two sentences: the source is fine here, the query is not.
    expect(await screen.findByText(/No model matches/)).toBeInTheDocument()
    expect(screen.queryByText("No models discovered yet.")).toBeNull()
  })

  it("renders an option's hint as a second line and keeps it in the row's name", async () => {
    render(
      <Harness
        label="Owner"
        options={[
          { value: "018f-0001", label: "Ada Lovelace", hint: "018f-0001" },
        ]}
      />,
    )

    await userEvent.click(screen.getByRole("combobox", { name: "Owner" }))

    const option = await screen.findByRole("option", {
      name: "Ada Lovelace (018f-0001)",
    })
    expect(option).toHaveTextContent("Ada Lovelace")
    expect(option).toHaveTextContent("018f-0001")
  })

  it("announces the description and the error on the input", async () => {
    const { rerender } = render(
      <ComboBoxField
        label="Serves"
        value=""
        onChange={() => {}}
        options={OPTIONS}
        description="Callers never see it."
      />,
    )

    const input = screen.getByRole("combobox", { name: "Serves" })
    const describedBy = input.getAttribute("aria-describedby")
    expect(describedBy).not.toBeNull()
    expect(document.getElementById(describedBy!)).toHaveTextContent(
      "Callers never see it.",
    )

    rerender(
      <ComboBoxField
        label="Serves"
        value=""
        onChange={() => {}}
        options={OPTIONS}
        description="Callers never see it."
        isInvalid
        errorMessage="Name a model before saving."
      />,
    )

    // Through HeroUI's error slot rather than a loose span, so the message is
    // announced with the field rather than sitting elsewhere in the form.
    expect(screen.getByRole("combobox", { name: "Serves" })).toHaveAttribute(
      "aria-invalid",
      "true",
    )
    expect(
      await screen.findByText("Name a model before saving."),
    ).toBeInTheDocument()
  })
})
