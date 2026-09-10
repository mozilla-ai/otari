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

/** The field is controlled, so a test that picks or types has to hold the value. */
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
  it("reports a picked row once, as its id, and displays its label", async () => {
    const seen: string[] = []
    render(
      <Harness
        options={[{ value: "018f-0001", label: "Ada Lovelace" }]}
        onValue={(value) => seen.push(value)}
      />,
    )

    const field = screen.getByRole("combobox", { name: "Serves" })
    await userEvent.click(field)
    await userEvent.click(
      await screen.findByRole("option", { name: "Ada Lovelace" }),
    )

    // One report, and it is the id. The label is what the box shows, which is
    // the whole division of labor here.
    expect(seen).toEqual(["018f-0001"])
    expect(field).toHaveValue("Ada Lovelace")
  })

  it("displays a label the caller resolves after mount", () => {
    const { rerender } = render(
      <ComboBoxField
        label="Owner"
        value="018f-0001"
        onChange={() => {}}
        options={[]}
      />,
    )

    // A roster or a catalog read lands after the field paints, so the id it
    // arrives with has to give way to the name it resolves to.
    const field = screen.getByRole("combobox", { name: "Owner" })
    expect(field).toHaveValue("018f-0001")

    rerender(
      <ComboBoxField
        label="Owner"
        value="018f-0001"
        onChange={() => {}}
        options={[{ value: "018f-0001", label: "Ada Lovelace" }]}
      />,
    )

    expect(field).toHaveValue("Ada Lovelace")
  })

  it("repaints when the caller moves the value out from under it", async () => {
    const props = {
      label: "Serves" as const,
      onChange: () => {},
      options: OPTIONS,
      allowsCustomValue: true,
    }
    const { rerender } = render(
      <ComboBoxField {...props} value="openai:gpt-4o" />,
    )

    const field = screen.getByRole("combobox", { name: "Serves" })
    await userEvent.type(field, "!")

    rerender(<ComboBoxField {...props} value="anthropic:claude-sonnet-4-5" />)

    // A row of these fields whose neighbor is removed hands a mounted field
    // somebody else's value, so a value the field did not report wins over the
    // text left in the box.
    expect(field).toHaveValue("anthropic:claude-sonnet-4-5")
  })

  it("clears the value when the box is emptied", async () => {
    const seen: string[] = []
    render(
      <Harness
        options={[{ value: "018f-0001", label: "Ada Lovelace" }]}
        onValue={(value) => seen.push(value)}
      />,
    )

    const field = screen.getByRole("combobox", { name: "Serves" })
    await userEvent.click(field)
    await userEvent.click(
      await screen.findByRole("option", { name: "Ada Lovelace" }),
    )
    await userEvent.clear(field)

    // Emptying the box is the one edit a whitelist reports, since it is the
    // only way to take a selection back.
    expect(seen).toEqual(["018f-0001", ""])
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

    // Editing the box is not holding a selection: what is in it is text, and a
    // field that went quiet after a pick would be the worse bug.
    expect(seen.at(-1)).toBe("Ada Lovelace!")
  })

  it("keeps text typed over a picked row when the field is left", async () => {
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
    await userEvent.clear(field)
    await userEvent.type(field, "ci-bot")
    await userEvent.tab()

    // Leaving the field is what commits it, and react-aria clears its selection
    // there when the text no longer matches the row. The typed value has to
    // survive that: it is what the form submits.
    expect(seen.at(-1)).toBe("ci-bot")
    expect(field).toHaveValue("ci-bot")
  })

  it("returns to the picked row when custom values are not allowed", async () => {
    render(
      <Harness options={[{ value: "018f-0001", label: "Ada Lovelace" }]} />,
    )

    const field = screen.getByRole("combobox", { name: "Serves" })
    await userEvent.click(field)
    await userEvent.click(
      await screen.findByRole("option", { name: "Ada Lovelace" }),
    )
    await userEvent.type(field, "zzz")
    await userEvent.tab()

    // A whitelist keeps no text nobody offered, so the box goes back to
    // reading as the value rather than sitting on a search that lost.
    expect(field).toHaveValue("Ada Lovelace")
  })

  it("publishes the input's text for a caller that filters", async () => {
    const queries: string[] = []
    render(
      <Harness
        allowsCustomValue
        options={[{ value: "018f-0001", label: "Ada Lovelace" }]}
        onQueryChange={(query) => queries.push(query)}
      />,
    )

    await userEvent.type(
      screen.getByRole("combobox", { name: "Serves" }),
      "ada",
    )
    expect(queries.at(-1)).toBe("ada")

    await userEvent.click(
      await screen.findByRole("option", { name: "Ada Lovelace" }),
    )

    // Empty once a row is picked: the field shows a choice rather than a
    // search, so the caller offers its whole list again, not the one row.
    expect(queries.at(-1)).toBe("")
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
