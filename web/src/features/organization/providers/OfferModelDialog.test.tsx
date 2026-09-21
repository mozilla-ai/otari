import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { orgProviderKey } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

import { OfferModelDialog } from "./OfferModelDialog"

const KEY = orgProviderKey({ name: "Production", provider: "openai" })

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi(available: Record<string, unknown>) {
  const requests: { url: string; method: string; body?: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    requests.push({
      url,
      method: (init?.method ?? "GET").toUpperCase(),
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })
    if (url.includes("/available-models")) return jsonResponse(available)
    return jsonResponse({ id: "x", model: "typed" })
  })
  return requests
}

function renderDialog(onOffered = vi.fn()) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <QueryClientProvider client={client}>
      <OfferModelDialog
        providerKey={KEY}
        isOpen
        onOpenChange={vi.fn()}
        onOffered={onOffered}
      />
    </QueryClientProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("OfferModelDialog", () => {
  it("offers the model that was typed, trimmed", async () => {
    const requests = mockApi({ provider: "openai", models: ["gpt-4o"] })
    const user = userEvent.setup()
    await renderDialog()

    const dialog = await screen.findByRole("dialog")
    const field = within(dialog).getByRole("combobox")
    await user.type(field, "  o3-mini  ")
    // The box opens on focus, and an open react-aria popover `aria-hidden`s the
    // rest of the dialog, so the submit is out of reach until it is put away.
    // Escape only while it reports itself open, which is testing.md's rule: sent
    // to a closed popover the key travels on to the dialog and closes that
    // instead, and the failure reads as a missing submit button.
    if (field.getAttribute("aria-expanded") === "true") {
      await user.keyboard("{Escape}")
    }
    await user.click(
      within(dialog).getByRole("button", { name: "Offer model" }),
    )

    await waitFor(() => {
      const offered = requests.find((request) => request.method === "POST")
      expect(offered?.body).toEqual({ model: "o3-mini" })
    })
  })

  it("will not submit an empty model", async () => {
    mockApi({ provider: "openai", models: [] })
    await renderDialog()

    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).getByRole("button", { name: "Offer model" }),
    ).toBeDisabled()
  })

  it("says why the picker is empty when the provider refused", async () => {
    // A refusal arrives in the body rather than as a status, and the field still
    // has to work: the model is sent exactly as typed either way.
    mockApi({
      provider: "openai",
      models: [],
      error: "the upstream refused the credential",
    })
    const user = userEvent.setup()
    await renderDialog()

    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox"))

    expect(
      await screen.findByText(/the upstream refused the credential/),
    ).toBeInTheDocument()
  })

  it("says so when the provider publishes no model list at all", async () => {
    // The case this dialog exists for: a backend that cannot be asked is still a
    // backend whose models an admin can name.
    mockApi({
      provider: "openai",
      models: [],
      discovery_unsupported: true,
    })
    const user = userEvent.setup()
    await renderDialog()

    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox"))

    expect(
      await screen.findByText(/publishes no model list/),
    ).toBeInTheDocument()
  })
})
