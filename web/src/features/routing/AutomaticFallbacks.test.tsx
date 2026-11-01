import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it } from "vitest"

import type { AutomaticFallback, FallbackOffering } from "@/client"
import { AutomaticFallbacks } from "@/features/routing/AutomaticFallbacks"
import { renderWithRouter } from "@/tests/router"

const offering = (
  selector: string,
  input_price: number | null = 1,
): FallbackOffering => ({
  selector,
  provider: selector.split(":")[0],
  provider_type: selector.split(":")[0],
  input_price,
  own_tools: [],
  otari_tools: [],
})

const FLASH: AutomaticFallback = {
  id: "deepseek/deepseek-v4.1-flash-1017",
  name: "DeepSeek V4.1 Flash 1017",
  offerings: [
    offering("deepinfra:deepseek-ai/DeepSeek-V4.1-Flash-1017", 0.098),
    offering("nebius:deepseek-ai/DeepSeek-V4.1-Flash-1017", 0.3),
    offering("together:deepseek-ai/DeepSeek-V4.1-Flash-1017", 0.3),
  ],
  requests: 1000,
  fell_back: 0,
  pinned: 0,
  pinned_provider: null,
  pinned_keys: [],
}

const PINNED: AutomaticFallback = {
  ...FLASH,
  pinned: 880,
  pinned_provider: "together",
  pinned_keys: [
    {
      api_key_id: "key-zed",
      key_name: "Zed",
      provider: "together",
      requests: 800,
    },
    { api_key_id: null, key_name: null, provider: "together", requests: 80 },
  ],
}

const rowOf = async (id: string) =>
  (await screen.findByText(id)).closest("tr") as HTMLElement

describe("AutomaticFallbacks", () => {
  it("lists a model by catalog ID with its requests, share and priced order", async () => {
    await renderWithRouter(<AutomaticFallbacks fallbacks={[FLASH]} />)

    const row = await rowOf(FLASH.id)
    expect(within(row).queryByText(FLASH.name)).not.toBeInTheDocument()
    expect(
      within(row).getByRole("button", { name: "Copy catalog ID" }),
    ).toBeInTheDocument()
    expect(within(row).getByText("1,000")).toBeInTheDocument()
    expect(within(row).getByText("100%")).toBeInTheDocument()
    expect(within(row).getByText("$0.098")).toBeInTheDocument()
    expect(within(row).getAllByText("$0.30")).toHaveLength(2)
    expect(within(row).getAllByLabelText("then")).toHaveLength(2)
    const deepinfra = within(row).getByText(/DeepInfra/)
    const together = within(row).getByText(/Together/)
    expect(
      deepinfra.compareDocumentPosition(together) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    expect(
      within(row).queryByRole("button", {
        name: "Keys that named a provider",
      }),
    ).not.toBeInTheDocument()
  })

  it("keeps the order when most requests named a provider, and flags the share", async () => {
    await renderWithRouter(<AutomaticFallbacks fallbacks={[PINNED]} />)

    const row = await rowOf(PINNED.id)
    const share = within(row).getByText("12%")
    expect(share).toHaveClass("text-warning")
    expect(within(row).getAllByLabelText("then")).toHaveLength(2)
    expect(within(row).getByText(/DeepInfra/)).toBeInTheDocument()
  })

  it("rounds the share down, so 100% means every request sent the catalog ID", async () => {
    await renderWithRouter(
      <AutomaticFallbacks
        fallbacks={[
          {
            ...PINNED,
            pinned: 1,
            pinned_keys: [PINNED.pinned_keys[0]],
          },
        ]}
      />,
    )
    const row = await rowOf(PINNED.id)
    const share = within(row).getByText("99%")
    expect(share).not.toHaveClass("text-warning")
  })

  it("opens a row on the keys that named a provider", async () => {
    const user = userEvent.setup()
    await renderWithRouter(<AutomaticFallbacks fallbacks={[PINNED]} />)

    await user.click(
      await screen.findByRole("button", { name: "Keys that named a provider" }),
    )
    expect(await screen.findByText("Zed")).toBeInTheDocument()
    expect(screen.getByText("800")).toBeInTheDocument()
    expect(screen.getByText("No key")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Hide keys" }))
    expect(screen.queryByText("Zed")).not.toBeInTheDocument()
  })

  it("shows a dash for a model one provider serves, and the provider alone", async () => {
    const single: AutomaticFallback = {
      ...FLASH,
      offerings: [
        offering("together:deepseek-ai/DeepSeek-V4.1-Flash-1017", 0.3),
      ],
    }
    await renderWithRouter(<AutomaticFallbacks fallbacks={[single]} />)

    const row = await rowOf(single.id)
    expect(within(row).getByText("—")).toBeInTheDocument()
    expect(within(row).queryByText(/%$/)).not.toBeInTheDocument()
    expect(within(row).queryByLabelText("then")).not.toBeInTheDocument()
  })

  it("names each provider once, Bedrock without Amazon, and leaves an unpriced one unpriced", async () => {
    const kimi: AutomaticFallback = {
      ...FLASH,
      id: "moonshotai/kimi-k3",
      offerings: [
        offering("bedrock:global.moonshotai.kimi-k3", 0.6),
        offering("bedrock:us.moonshotai.kimi-k3", 0.6),
        offering("fireworks:accounts/fireworks/models/kimi-k3", null),
      ],
    }
    await renderWithRouter(<AutomaticFallbacks fallbacks={[kimi]} />)

    const row = await rowOf(kimi.id)
    expect(within(row).getAllByText(/Bedrock/)).toHaveLength(1)
    expect(within(row).getAllByLabelText("then")).toHaveLength(1)
    expect(within(row).getAllByText(/\$/)).toHaveLength(1)
  })

  it("says when nothing was sent in the window", async () => {
    await renderWithRouter(<AutomaticFallbacks fallbacks={[]} />)
    expect(
      await screen.findByText("No traffic in the last 30 days."),
    ).toBeInTheDocument()
  })

  it("reports a failed read instead of an empty table", async () => {
    await renderWithRouter(
      <AutomaticFallbacks fallbacks={[]} error={new Error("Gateway down")} />,
    )
    expect(await screen.findByRole("alert")).toHaveTextContent("Gateway down")
    expect(
      screen.queryByText("No traffic in the last 30 days."),
    ).not.toBeInTheDocument()
  })
})
