import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ProviderInstanceComboBox } from "@/features/providers/ProviderInstanceComboBox"

function modelObject(id: string) {
  return {
    id,
    object: "model",
    created: 0,
    owned_by: id.split(":")[0],
    pricing_source: "none",
  }
}

// Mocked at the transport, per the standards: the query key and the hook are
// part of what a picker over a shared read has to get right.
function mockCatalog(ids: string[]) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(
    async () =>
      new Response(
        JSON.stringify({ object: "list", data: ids.map(modelObject) }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
  )
}

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ProviderInstanceComboBox", () => {
  it("offers each configured instance once, however many models it serves", async () => {
    mockCatalog([
      "openai-eu:gpt-4o",
      "openai-eu:gpt-4o-mini",
      "anthropic:claude-sonnet-4",
    ])
    const user = userEvent.setup()
    renderWithClient(
      <ProviderInstanceComboBox
        label="Provider instance"
        value=""
        onChange={() => {}}
      />,
    )
    const input = await screen.findByRole("combobox", {
      name: "Provider instance",
    })

    await user.click(input)

    expect(
      await screen.findByRole("option", { name: "anthropic" }),
    ).toBeInTheDocument()
    expect(screen.getAllByRole("option", { name: "openai-eu" })).toHaveLength(1)
  })

  it("leaves out a name that is not an instance", async () => {
    // An alias and a routing policy are listed in the catalog under a bare
    // display name. Neither is something a ceiling can be narrowed to, and
    // offering one would store a cap that binds nothing.
    mockCatalog(["openai:gpt-4o", "fast", "cheap-pool"])
    const user = userEvent.setup()
    renderWithClient(
      <ProviderInstanceComboBox
        label="Provider instance"
        value=""
        onChange={() => {}}
      />,
    )
    await user.click(
      await screen.findByRole("combobox", { name: "Provider instance" }),
    )

    await screen.findByRole("option", { name: "openai" })
    expect(screen.queryByRole("option", { name: "fast" })).toBeNull()
    expect(screen.queryByRole("option", { name: "cheap-pool" })).toBeNull()
  })

  it("reports what was typed, so an instance the catalog cannot see still stands", async () => {
    mockCatalog(["openai:gpt-4o"])
    const onChange = vi.fn()
    const user = userEvent.setup()
    renderWithClient(
      <ProviderInstanceComboBox
        label="Provider instance"
        value=""
        onChange={onChange}
      />,
    )

    await user.type(
      await screen.findByRole("combobox", { name: "Provider instance" }),
      "vllm-lab",
    )

    expect(onChange).toHaveBeenLastCalledWith("vllm-lab")
  })

  it("says why the popover is empty rather than leaving an empty box", async () => {
    mockCatalog([])
    const user = userEvent.setup()
    renderWithClient(
      <ProviderInstanceComboBox
        label="Provider instance"
        value=""
        onChange={() => {}}
      />,
    )
    await user.click(
      await screen.findByRole("combobox", { name: "Provider instance" }),
    )

    expect(
      await screen.findByText(/No provider serves a model here yet/),
    ).toBeInTheDocument()
  })

  it("says the list could not be read, rather than that nothing is configured", async () => {
    // The two leave the same empty popover and want opposite things said: one
    // is filled by configuring a provider, the other by nothing on this form.
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async () =>
        new Response(JSON.stringify({ detail: "nope" }), {
          status: 500,
          headers: { "Content-Type": "application/json" },
        }),
    )
    const user = userEvent.setup()
    renderWithClient(
      <ProviderInstanceComboBox
        label="Provider instance"
        value=""
        onChange={() => {}}
      />,
    )
    await user.click(
      await screen.findByRole("combobox", { name: "Provider instance" }),
    )

    expect(
      await screen.findByText(/model list could not be read/),
    ).toBeInTheDocument()
    expect(screen.queryByText(/No provider serves a model here yet/)).toBeNull()
  })
})
