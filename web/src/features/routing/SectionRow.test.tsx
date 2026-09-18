import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { SectionRow } from "./SectionRow"

// The row reads the model catalog, which is a query. Mocked at the transport so
// the real hook runs and picks its own branch: a provider that listed cleanly
// and matches the row's value has nothing to say, and one that could not be
// listed does. `failed` is derived from a provider's `ok`, not from a separate
// field, so an unhealthy provider is how the hint is reached.
function mockCatalog({ isHealthy }: { isHealthy: boolean }) {
  return vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(
      JSON.stringify({
        providers: [
          {
            provider: "openai",
            ok: isHealthy,
            models: isHealthy ? [{ key: "openai/gpt-4o" }] : [],
          },
        ],
      }),
      { status: 200 },
    ),
  )
}

function renderRow(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>{ui}</QueryClientProvider>
    </DeploymentProvider>,
  )
}

describe("SectionRow", () => {
  afterEach(() => vi.restoreAllMocks())

  it("renders the controls it is given", () => {
    mockCatalog({ isHealthy: true })
    renderRow(
      <SectionRow id="row-hint" modelValue="openai/gpt-4o">
        <button type="button">Pick a model</button>
      </SectionRow>,
    )
    expect(
      screen.getByRole("button", { name: "Pick a model" }),
    ).toBeInTheDocument()
  })

  it("says nothing under a row whose catalog listed cleanly", async () => {
    mockCatalog({ isHealthy: true })
    renderRow(
      <SectionRow id="row-hint" modelValue="openai/gpt-4o">
        <button type="button">Pick a model</button>
      </SectionRow>,
    )
    // The hint element carries the id the field points its description at, so
    // its absence is what says there is nothing to announce. Waiting for the
    // element to go is what waits for the query: the row carries a "Loading
    // models…" hint until the catalog lands, so asserting straight away reads
    // the loading state rather than the answer.
    await waitFor(() => expect(document.getElementById("row-hint")).toBeNull())
  })

  it("carries the catalog's hint under the whole row, announced with the field", async () => {
    mockCatalog({ isHealthy: false })
    renderRow(
      <SectionRow id="row-hint" modelValue="openai/gpt-4o">
        <button type="button">Pick a model</button>
      </SectionRow>,
    )
    // Under the row rather than inside the picker: a wrapped sentence beside an
    // input lifts that field's line clear of the siblings it shares a row with.
    const hint = await screen.findByText(/Could not list models for openai/)
    expect(hint).toHaveAttribute("id", "row-hint")
  })
})
