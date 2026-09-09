import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"
import { FilterMultiComboBox } from "@/shared/components/navigation/FilterMultiComboBox"

describe("FilterMultiComboBox", () => {
  function Harness({
    onChange,
    maxValues,
    allowsCustom,
  }: {
    onChange?: (values: string[]) => void
    maxValues?: number
    allowsCustom?: boolean
  }) {
    const [values, setValues] = useState<string[]>([])
    return (
      <FilterMultiComboBox
        label="Model"
        values={values}
        onChange={(next) => {
          setValues(next)
          onChange?.(next)
        }}
        options={[
          { value: "gpt-5.6", label: "gpt-5.6" },
          { value: "claude-sonnet-5", label: "claude-sonnet-5" },
        ]}
        placeholder="All models"
        maxValues={maxValues}
        allowsCustom={allowsCustom}
      />
    )
  }

  it("accumulates picks and drops the picked options from the list", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} />)

    const input = screen.getByRole("combobox", { name: "Model" })
    await user.click(input)
    await user.click(await screen.findByRole("option", { name: "gpt-5.6" }))
    expect(onChange).toHaveBeenLastCalledWith(["gpt-5.6"])

    // The list stays open on what is left, so a comparison is one gesture; a
    // picked value is not offered again (picking it twice would be a no-op).
    expect(
      screen.queryByRole("option", { name: "gpt-5.6" }),
    ).not.toBeInTheDocument()
    await user.click(
      await screen.findByRole("option", { name: "claude-sonnet-5" }),
    )
    expect(onChange).toHaveBeenLastCalledWith(["gpt-5.6", "claude-sonnet-5"])

    // The input reports the size of the selection: the values themselves are the
    // page's chips, not text crammed into the box.
    expect(input).toHaveAttribute("placeholder", "2 selected")
  })

  it("narrows the list as you type without committing the text", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} />)

    const input = screen.getByRole("combobox", { name: "Model" })
    await user.click(input)
    await user.type(input, "claude")
    expect(
      screen.queryByRole("option", { name: "gpt-5.6" }),
    ).not.toBeInTheDocument()
    expect(
      await screen.findByRole("option", { name: "claude-sonnet-5" }),
    ).toBeInTheDocument()
    // Typing alone filters nothing: only a picked option becomes a filter value.
    expect(onChange).not.toHaveBeenCalled()
  })

  it("commits typed text on Enter when custom values are allowed", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} allowsCustom />)

    const input = screen.getByRole("combobox", { name: "Model" })
    await user.click(input)
    await user.type(input, "unlisted-model")
    await user.keyboard("{Enter}")

    expect(onChange).toHaveBeenLastCalledWith(["unlisted-model"])
  })

  it("does not also commit the query when Enter picks a highlighted option", async () => {
    // One press must add one value. Arrowing onto an option and pressing Enter
    // selects it; committing the partial text beside it would add "claude" as well.
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} allowsCustom />)

    const input = screen.getByRole("combobox", { name: "Model" })
    await user.click(input)
    await user.type(input, "claude")
    await user.keyboard("{ArrowDown}")
    await user.keyboard("{Enter}")

    expect(onChange).toHaveBeenCalledTimes(1)
    expect(onChange).toHaveBeenLastCalledWith(["claude-sonnet-5"])
  })

  it("ignores Enter on typed text when custom values are not allowed", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} />)

    const input = screen.getByRole("combobox", { name: "Model" })
    await user.click(input)
    await user.type(input, "not-an-option")
    await user.keyboard("{Enter}")

    expect(onChange).not.toHaveBeenCalled()
  })

  it("stops at the value ceiling the endpoints accept", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} maxValues={1} />)

    const input = screen.getByRole("combobox", { name: "Model" })
    await user.click(input)
    await user.click(await screen.findByRole("option", { name: "gpt-5.6" }))
    expect(onChange).toHaveBeenLastCalledWith(["gpt-5.6"])

    // Past the ceiling the remaining options are inert rather than a pick that
    // 422s every query on the page, and the input says the set is full.
    expect(input).toHaveAttribute("placeholder", "1 selected (max)")
    const remaining = await screen.findByRole("option", {
      name: "claude-sonnet-5",
    })
    expect(remaining).toHaveAttribute("aria-disabled", "true")
    await user.click(remaining)
    expect(onChange).toHaveBeenCalledTimes(1)
  })
})
