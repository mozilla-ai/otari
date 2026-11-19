import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { SpendPeriodPicker } from "@/features/spend/SpendPeriodPicker"
import { AppProviders as Provider } from "@/tests/providers"

function setPhone(isPhone: boolean) {
  vi.stubGlobal(
    "matchMedia",
    vi.fn().mockImplementation((query: string) => ({
      matches: isPhone,
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false,
    })),
  )
}

describe("SpendPeriodPicker", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("lays the six windows out as one row on a wide screen", async () => {
    setPhone(false)
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(
      <Provider>
        <SpendPeriodPicker value="30d" onChange={onChange} />
      </Provider>,
    )
    const group = screen.getByRole("radiogroup", { name: "Window" })
    const radios = within(group).getAllByRole("radio")
    expect(radios.map((radio) => radio.closest("label")?.textContent)).toEqual([
      "Today",
      "Yesterday",
      "7D",
      "30D",
      "3M",
      "6M",
    ])
    expect(within(group).getByRole("radio", { name: "30D" })).toBeChecked()
    await user.click(within(group).getByRole("radio", { name: "Yesterday" }))
    expect(onChange).toHaveBeenCalledWith("yesterday")
  })

  it("is a select on a phone, where six segments do not fit", async () => {
    setPhone(true)
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(
      <Provider>
        <SpendPeriodPicker value="7d" onChange={onChange} />
      </Provider>,
    )
    expect(screen.queryByRole("radiogroup")).toBeNull()
    const trigger = screen.getByRole("button", { name: /7D/ })
    await user.click(trigger)
    const options = await screen.findAllByRole("option")
    expect(options.map((option) => option.textContent)).toEqual([
      "Today",
      "Yesterday",
      "7D",
      "30D",
      "3M",
      "6M",
    ])
    await user.click(screen.getByRole("option", { name: "6M" }))
    expect(onChange).toHaveBeenCalledWith("6m")
  })
})
