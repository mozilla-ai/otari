import { screen, within } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import type { AutomaticFallback, FallbackOffering } from "@/client"
import { AutomaticFallbacks } from "@/features/routing/AutomaticFallbacks"
import { renderWithRouter } from "@/tests/router"

const offering = (
  selector: string,
  own: string[] = [],
  otari: string[] = [],
): FallbackOffering => ({
  selector,
  provider: selector.split(":")[0],
  provider_type: selector.split(":")[0],
  own_tools: own,
  otari_tools: otari,
})

const SONNET: AutomaticFallback = {
  id: "anthropic/claude-sonnet-5.5",
  name: "Claude Sonnet 5.5",
  offerings: [
    offering("anthropic:claude-sonnet-5-5", ["web_search"]),
    offering("bedrock:anthropic.claude-sonnet-5-5", [], ["web_search"]),
  ],
  requests: 1200,
  fell_back: 7,
  pinned: 0,
  pinned_provider: null,
}

const rowOf = async (name: string) =>
  (await screen.findByText(name)).closest("tr") as HTMLElement

describe("AutomaticFallbacks", () => {
  it("lists a model by name, copies its catalog ID, and shows the order tried", async () => {
    await renderWithRouter(<AutomaticFallbacks fallbacks={[SONNET]} />)

    const row = await rowOf("Claude Sonnet 5.5")
    // The catalog ID is not repeated as text: it is the name's title and what the button copies.
    expect(within(row).queryByText(SONNET.id)).not.toBeInTheDocument()
    expect(within(row).getByTitle(SONNET.id)).toBeInTheDocument()
    expect(
      within(row).getByRole("button", { name: "Copy catalog ID" }),
    ).toBeInTheDocument()
    expect(within(row).getByText("1,200")).toBeInTheDocument()
    expect(within(row).getByText("7")).toBeInTheDocument()
    expect(within(row).getByText("Anthropic")).toBeInTheDocument()
    expect(within(row).getByLabelText("then")).toBeInTheDocument()
    expect(within(row).getByText("Own tools")).toBeInTheDocument()
    expect(within(row).getByText("Otari's tools")).toBeInTheDocument()
    expect(within(row).queryByText(/No backup/)).not.toBeInTheDocument()
  })

  it("labels a mixed provider's tools as plain text, not a control", async () => {
    const gpt: AutomaticFallback = {
      ...SONNET,
      id: "openai/gpt-x",
      name: "GPT X",
      offerings: [
        offering(
          "openai:gpt-x",
          ["web_search", "code_execution"],
          ["web_fetch"],
        ),
        offering("bedrock:openai.gpt-x", [], ["web_search"]),
      ],
    }
    await renderWithRouter(<AutomaticFallbacks fallbacks={[gpt]} />)

    const row = await rowOf("GPT X")
    expect(within(row).getByText("Own tools")).toBeInTheDocument()
    expect(
      within(row).queryByRole("button", { name: "Own tools" }),
    ).not.toBeInTheDocument()
  })

  it("names each provider once, and Bedrock without Amazon", async () => {
    const kimi: AutomaticFallback = {
      ...SONNET,
      id: "moonshotai/kimi-k3",
      name: "Kimi K3",
      offerings: [
        offering("bedrock:global.moonshotai.kimi-k3", [], ["web_search"]),
        offering("bedrock:us.moonshotai.kimi-k3", [], ["web_search"]),
        offering(
          "fireworks:accounts/fireworks/models/kimi-k3",
          [],
          ["web_search"],
        ),
      ],
    }
    await renderWithRouter(<AutomaticFallbacks fallbacks={[kimi]} />)

    const row = await rowOf("Kimi K3")
    expect(within(row).getAllByText("Bedrock")).toHaveLength(1)
    expect(within(row).getAllByLabelText("then")).toHaveLength(1)
  })

  it("labels a provider's tools as its own where the server says it runs them", async () => {
    const gemini: AutomaticFallback = {
      ...SONNET,
      id: "google/gemini-x",
      name: "Gemini X",
      offerings: [offering("gemini:models/gemini-x", ["web_search"])],
    }
    await renderWithRouter(<AutomaticFallbacks fallbacks={[gemini]} />)

    const row = await rowOf("Gemini X")
    expect(within(row).getByText("Own tools")).toBeInTheDocument()
  })

  it("warns that a model on one provider has no backup", async () => {
    const gemini: AutomaticFallback = {
      ...SONNET,
      id: "google/gemini-x",
      name: "Gemini X",
      offerings: [offering("gemini:gemini-x")],
    }
    await renderWithRouter(<AutomaticFallbacks fallbacks={[gemini]} />)

    const row = await rowOf("Gemini X")
    expect(within(row).getByText("No backup")).toBeInTheDocument()
    expect(within(row).queryByLabelText("then")).not.toBeInTheDocument()
  })

  it("shows the pinned provider when every request named one", async () => {
    const pinned: AutomaticFallback = {
      ...SONNET,
      pinned: SONNET.requests,
      pinned_provider: "bedrock",
    }
    await renderWithRouter(<AutomaticFallbacks fallbacks={[pinned]} />)

    const row = await rowOf("Claude Sonnet 5.5")
    expect(within(row).getByText("No backup · pinned")).toBeInTheDocument()
    expect(within(row).queryByText("Anthropic")).not.toBeInTheDocument()
    expect(within(row).queryByLabelText("then")).not.toBeInTheDocument()
  })

  it("marks a model pinned once most of its requests are", async () => {
    await renderWithRouter(
      <AutomaticFallbacks
        fallbacks={[
          {
            ...SONNET,
            pinned: SONNET.requests - 3,
            pinned_provider: "anthropic",
          },
        ]}
      />,
    )
    const row = await rowOf("Claude Sonnet 5.5")
    expect(within(row).getByText("No backup · pinned")).toBeInTheDocument()
  })

  it("keeps the chain when only some requests were pinned", async () => {
    await renderWithRouter(
      <AutomaticFallbacks
        fallbacks={[{ ...SONNET, pinned: 3, pinned_provider: "bedrock" }]}
      />,
    )
    const row = await rowOf("Claude Sonnet 5.5")
    expect(within(row).getByLabelText("then")).toBeInTheDocument()
    expect(within(row).queryByText(/No backup/)).not.toBeInTheDocument()
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
