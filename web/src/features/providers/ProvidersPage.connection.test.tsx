import { QueryClient } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ProvidersPage } from "@/features/providers/ProvidersPage"
import { PROVIDER_HEALTH_REFRESH_MS } from "@/shared/api/providers"
import {
  healthRequestCount,
  healthResponse,
  mockApi,
  providerInfo,
  renderPage,
  storedProvider,
} from "@/tests/providersPage"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ProvidersPage connection tests", () => {
  it("scrolls a connection-test verdict into view, since the body scrolls", async () => {
    // The verdict is the body's last child, under fields that already fill an
    // `lg` dialog on a laptop, so without this the footer button goes back to
    // "Test connection" and nothing else visibly happens.
    const scrollIntoView = vi.spyOn(Element.prototype, "scrollIntoView")
    mockApi({
      meta: [],
      stored: [],
      testResult: {
        ok: true,
        model_count: 3,
        error: null,
        discovery_unsupported: false,
      },
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider" }),
    )
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.type(screen.getByLabelText(/^Name/), "my-local-llm")
    await user.type(
      screen.getByLabelText("API base"),
      "http://localhost:8000/v1",
    )
    scrollIntoView.mockClear()
    await user.click(screen.getByRole("button", { name: "Test connection" }))

    const verdict = await screen.findByText(/Connected\. 3 models available\./)
    // The live region, which is the element that carries the ref.
    const region = verdict.closest('[role="status"]')
    expect(scrollIntoView.mock.instances).toContain(region)
  })

  it("renders a connection-test outcome in the body, not in the footer", async () => {
    // The unverified case is four lines plus the provider's reply. In the footer
    // it grows the one row feedback.md says never changes height and shoves the
    // form up mid-typing; in the body it scrolls with the fields.
    mockApi({
      meta: [],
      stored: [],
      testResult: {
        ok: false,
        model_count: 0,
        error: "Error code: 404",
        discovery_unsupported: true,
      },
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider" }),
    )
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.type(screen.getByLabelText(/^Name/), "my-local-llm")
    await user.type(
      screen.getByLabelText("API base"),
      "http://localhost:8000/v1",
    )
    await user.click(screen.getByRole("button", { name: "Test connection" }))

    const outcome = await screen.findByText(/does not list models/)
    const footer = document.querySelector(".otari-form-dialog__footer")
    expect(footer).not.toBeNull()
    expect(footer?.contains(outcome)).toBe(false)
    // The button that ran it stays in the footer.
    expect(
      footer?.contains(screen.getByRole("button", { name: "Test connection" })),
    ).toBe(true)
  })

  it("reports a successful connection test", async () => {
    mockApi({
      stored: [storedProvider("openai", "1234")],
      testResult: {
        ok: true,
        model_count: 5,
        error: null,
        discovery_unsupported: false,
      },
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Test" }))

    expect(
      await screen.findByText(/Connected\. 5 models available\./),
    ).toBeInTheDocument()
  })

  // The gateway-wide require_pricing alarm moved to the app shell; its behavior
  // (show, enable default pricing, dismiss) is covered in PricingWarning.test.tsx.

  it("warns instead of condemning a provider whose backend has no /models endpoint", async () => {
    // otari#447: a provider that answers no model listing may still serve
    // requests, so it must not read as "Unreachable" like a bad key does.
    mockApi({
      meta: [providerInfo("openai"), providerInfo("otari")],
      health: [
        {
          instance: "openai",
          ok: true,
          model_count: 3,
          error: null,
          checked_at: null,
          discovery_unsupported: false,
        },
        {
          instance: "otari",
          ok: false,
          model_count: 0,
          error: "Error code: 404 - {'detail': 'Not Found'}",
          checked_at: null,
          discovery_unsupported: true,
        },
      ],
    })
    renderPage(<ProvidersPage />)

    const pill = await screen.findByText("No model discovery")
    expect(
      within(pill.closest("tr")!).getAllByText("otari").length,
    ).toBeGreaterThan(0)
    expect(screen.queryByText("Unreachable")).not.toBeInTheDocument()
    // The provider error stays available, alongside why it is not fatal.
    expect(pill).toHaveAttribute("title", expect.stringContaining("404"))
    expect(pill).toHaveAttribute(
      "title",
      expect.stringContaining("may still work"),
    )
    // The summary calls it out separately from the reachable count.
    expect(
      await screen.findByText("1 of 2 providers reachable"),
    ).toBeInTheDocument()
    expect(screen.getByText("1 without model discovery")).toBeInTheDocument()
  })

  it("reports a test against a provider with no model listing as unverified, not failed", async () => {
    mockApi({
      stored: [storedProvider("otari", "1234")],
      testResult: {
        ok: false,
        model_count: 0,
        error: "Error code: 404",
        discovery_unsupported: true,
      },
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Test" }))

    expect(await screen.findByText(/could not be verified/)).toBeInTheDocument()
    // The provider error stays on screen: a 404 is also what a wrong api_base
    // returns, so hiding it would mask a misconfiguration behind reassurance.
    expect(screen.getByText("Error code: 404")).toBeInTheDocument()
  })

  it("drops a connection-test verdict once the provider is edited", async () => {
    // otari#464: the verdict describes the credentials the test ran against, so
    // leaving it under the row after a save contradicts the status pill above it
    // and sends the operator hunting for a backend bug.
    mockApi({
      stored: [storedProvider("otari", "1234")],
      testResult: {
        ok: false,
        model_count: 0,
        error: "authentication failed: invalid key",
        discovery_unsupported: false,
      },
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Test" }))
    expect(
      await screen.findByText("authentication failed: invalid key"),
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.clear(screen.getByLabelText("API base"))
    await user.type(
      screen.getByLabelText("API base"),
      "https://api.otari.ai/v1",
    )
    await user.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(
        screen.queryByText("authentication failed: invalid key"),
      ).not.toBeInTheDocument(),
    )
  })

  it("does not let a test still in flight write its verdict back after a save", async () => {
    // A test against a wrong api_base settles only when it times out, which is
    // when the operator is most likely to go and fix the credentials. The save
    // retires the verdict; the late result must not restore it.
    let releaseTest: () => void = () => {}
    const testGate = new Promise<void>((resolve) => {
      releaseTest = resolve
    })
    mockApi({
      stored: [storedProvider("otari", "1234")],
      testResult: {
        ok: false,
        model_count: 0,
        error: "authentication failed: invalid key",
        discovery_unsupported: false,
      },
      testGate,
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Test" }))
    expect(await screen.findByText("Testing…")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.clear(screen.getByLabelText("API base"))
    await user.type(
      screen.getByLabelText("API base"),
      "https://api.otari.ai/v1",
    )
    await user.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(screen.queryByText("Testing…")).not.toBeInTheDocument(),
    )

    releaseTest()

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Test" })).toBeEnabled(),
    )
    expect(
      screen.queryByText("authentication failed: invalid key"),
    ).not.toBeInTheDocument()
  })

  it("does not carry a verdict over to a provider re-added under the same name", async () => {
    mockApi({
      stored: [storedProvider("otari", "1234")],
      testResult: {
        ok: false,
        model_count: 0,
        error: "authentication failed: invalid key",
        discovery_unsupported: false,
      },
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Test" }))
    expect(
      await screen.findByText("authentication failed: invalid key"),
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Delete" }))
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Delete provider",
      }),
    )
    await screen.findByText("Welcome to Otari")

    await user.click(
      screen.getByRole("button", { name: "Add your first provider" }),
    )
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.type(screen.getByLabelText("Name"), "otari")
    await user.type(
      screen.getByLabelText("API base"),
      "https://api.otari.ai/v1",
    )
    await user.click(screen.getByRole("button", { name: "Add provider" }))

    // The rebuilt row is a different provider: it must start with no verdict.
    await screen.findByRole("button", { name: "Test" })
    expect(
      screen.queryByText("authentication failed: invalid key"),
    ).not.toBeInTheDocument()
  })

  it("does not let a test still in flight write its verdict back after a delete", async () => {
    // The delete path retires the verdict through its own callback, separate from
    // the save path, and the request behind it is still hanging. A provider
    // re-added under the same name is a different provider, so the late result
    // must not surface on its row.
    let releaseTest: () => void = () => {}
    const testGate = new Promise<void>((resolve) => {
      releaseTest = resolve
    })
    mockApi({
      stored: [storedProvider("otari", "1234")],
      testResult: {
        ok: false,
        model_count: 0,
        error: "authentication failed: invalid key",
        discovery_unsupported: false,
      },
      testGate,
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Test" }))
    expect(await screen.findByText("Testing…")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Delete" }))
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Delete provider",
      }),
    )
    await screen.findByText("Welcome to Otari")

    await user.click(
      screen.getByRole("button", { name: "Add your first provider" }),
    )
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.type(screen.getByLabelText("Name"), "otari")
    await user.type(
      screen.getByLabelText("API base"),
      "https://api.otari.ai/v1",
    )
    await user.click(screen.getByRole("button", { name: "Add provider" }))
    await screen.findByRole("button", { name: "Test" })

    releaseTest()

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Test" })).toBeEnabled(),
    )
    expect(
      screen.queryByText("authentication failed: invalid key"),
    ).not.toBeInTheDocument()
  })

  it("lets the retest after a save win, even if the pre-save test answers last", async () => {
    // Saving retires the pending verdict, which re-enables Test in the same
    // instant, so the operator can retest the fixed credentials while the
    // pre-save request is still hanging. Only the run id distinguishes the two:
    // the late answer belongs to credentials that no longer exist.
    let releaseStale: () => void = () => {}
    const staleGate = new Promise<void>((resolve) => {
      releaseStale = resolve
    })
    mockApi({
      stored: [storedProvider("otari", "1234")],
      testCalls: [
        {
          gate: staleGate,
          result: {
            ok: false,
            model_count: 0,
            error: "authentication failed: invalid key",
            discovery_unsupported: false,
          },
        },
        {
          result: {
            ok: true,
            model_count: 7,
            error: null,
            discovery_unsupported: false,
          },
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Test" }))
    expect(await screen.findByText("Testing…")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.clear(screen.getByLabelText("API base"))
    await user.type(
      screen.getByLabelText("API base"),
      "https://api.otari.ai/v1",
    )
    await user.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(screen.queryByText("Testing…")).not.toBeInTheDocument(),
    )

    await user.click(screen.getByRole("button", { name: "Test" }))
    releaseStale()

    expect(
      await screen.findByText("Connected. 7 models available."),
    ).toBeInTheDocument()
    expect(
      screen.queryByText("authentication failed: invalid key"),
    ).not.toBeInTheDocument()
  })

  it("settles both rows when two connection tests run at once", async () => {
    // One useMutation observer serves every row, and TanStack Query detaches it
    // from the previous mutation as soon as the next mutate lands, dropping that
    // call's callbacks. Testing a second provider while the first was in flight
    // used to leave the first row spinning on "Testing…" with its Test button
    // disabled for the life of the page.
    let releaseTests: () => void = () => {}
    const testGate = new Promise<void>((resolve) => {
      releaseTests = resolve
    })
    mockApi({
      stored: [
        storedProvider("anthropic", "1111"),
        storedProvider("openai", "2222"),
      ],
      testResult: {
        ok: true,
        model_count: 4,
        error: null,
        discovery_unsupported: false,
      },
      testGate,
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1111")
    expect(screen.getAllByRole("button", { name: "Test" })).toHaveLength(2)
    await user.click(screen.getAllByRole("button", { name: "Test" })[0])
    await user.click(screen.getAllByRole("button", { name: "Test" })[1])
    expect(await screen.findAllByText("Testing…")).toHaveLength(2)

    releaseTests()

    // Both verdicts land, and neither row is left stuck pending.
    await waitFor(() =>
      expect(
        screen.getAllByText("Connected. 4 models available."),
      ).toHaveLength(2),
    )
    expect(screen.queryByText("Testing…")).not.toBeInTheDocument()
    for (const button of screen.getAllByRole("button", { name: "Test" })) {
      expect(button).toBeEnabled()
    }
  })

  it("does not automatically re-check all providers within an hour", async () => {
    const fetchMock = mockApi({ meta: [providerInfo("openai")] })
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    const first = renderPage(<ProvidersPage />, client)

    await screen.findByText("1 of 1 provider reachable")
    expect(healthRequestCount(fetchMock)).toBe(1)

    first.unmount()
    client.setQueryData(
      ["provider-health"],
      healthResponse([
        {
          instance: "openai",
          ok: true,
          model_count: 3,
          error: null,
          checked_at: null,
          discovery_unsupported: false,
        },
      ]),
      { updatedAt: Date.now() - (PROVIDER_HEALTH_REFRESH_MS - 5_000) },
    )
    renderPage(<ProvidersPage />, client)

    await screen.findByText("1 of 1 provider reachable")
    await waitFor(() => expect(healthRequestCount(fetchMock)).toBe(1))
  })

  it("forces a live re-check of every provider on Re-check all", async () => {
    const user = userEvent.setup()
    mockApi({
      meta: [providerInfo("openai")],
      health: [
        {
          instance: "openai",
          ok: true,
          model_count: 3,
          error: null,
          checked_at: null,
          discovery_unsupported: false,
        },
      ],
      healthRefresh: [
        {
          instance: "openai",
          ok: false,
          model_count: 0,
          error: "provider down",
          checked_at: null,
          discovery_unsupported: false,
        },
      ],
    })
    renderPage(<ProvidersPage />)

    const row = (await screen.findByText("Reachable")).closest("tr")!
    expect(within(row).getAllByText("openai").length).toBeGreaterThan(0)

    await user.click(screen.getByRole("button", { name: "Re-check all" }))

    expect(await within(row).findByText("Unreachable")).toBeInTheDocument()
    expect(
      await screen.findByText("0 of 1 provider reachable"),
    ).toBeInTheDocument()
  })
})
