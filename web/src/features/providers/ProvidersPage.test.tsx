import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ProvidersPage } from "@/features/providers/ProvidersPage"
import { organizationContext } from "@/tests/fixtures"
import {
  mockApi,
  providerInfo,
  renderPage,
  storedProvider,
} from "@/tests/providersPage"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ProvidersPage", () => {
  it("lists config and stored providers with provenance and redacted keys", async () => {
    mockApi({
      meta: [
        providerInfo("openai", "OPENAI_API_KEY"),
        providerInfo("anthropic"),
      ],
      stored: [storedProvider("anthropic", "4242")],
    })

    renderPage(<ProvidersPage />)

    // Key off cells unique to each row (the instance name appears in two columns).
    const storedRow = (await screen.findByText("••••4242")).closest("tr")!
    expect(within(storedRow).getByText("STORED")).toBeInTheDocument()

    const configRow = screen.getByText("OPENAI_API_KEY").closest("tr")!
    expect(within(configRow).getByText("CONFIG")).toBeInTheDocument()
    // The plaintext key is never shown, only the last 4.
    expect(document.body.textContent).not.toContain("sk-")
  })

  it("replaces the welcome onboarding with the add form", async () => {
    mockApi({ meta: [], stored: [] })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    expect(await screen.findByText("Welcome to Otari")).toBeInTheDocument()
    // The heading keeps its action beside the first-run panel's own copy of it.
    // It used to hide while that panel showed, on the reasoning that two
    // competed; the panel is a band and the form it opens is over the page.
    expect(
      screen.getByRole("button", { name: "Add provider" }),
    ).toBeInTheDocument()
    // Only the onboarding panel ("Welcome to Otari") shows: the table (and its own
    // "no rows" fallback, whose "No providers yet" text is unique to it) is
    // suppressed so the two empty states are not stacked.
    expect(screen.queryByText(/No providers yet/)).not.toBeInTheDocument()
    expect(
      screen.queryByRole("grid", { name: "Providers" }),
    ).not.toBeInTheDocument()
    const firstProvider = screen.getByRole("button", {
      name: "Add your first provider",
    })
    await user.click(firstProvider)

    expect(screen.getByPlaceholderText("Search providers…")).toBeInTheDocument()
    // The panel stays mounted under the dialog. It used to unmount, which took
    // away the node react-aria restores focus to on close, so closing dropped
    // focus to `<body>`.
    expect(screen.getByText("Welcome to Otari")).toBeInTheDocument()
    await user.keyboard("{Escape}")
    await waitFor(() => expect(firstProvider).toHaveFocus())
  })

  it("points the onboarding quickstart at the gateway-served tutorial in a new tab", async () => {
    mockApi({ meta: [], stored: [] })
    renderPage(<ProvidersPage />)

    await screen.findByText("Welcome to Otari")
    const quickstart = screen.getByRole("link", { name: "quickstart" })
    // /welcome is a gateway-rendered page, not a client route: a router Link
    // (href "#/welcome") would hit the catch-all and redirect to the overview.
    expect(quickstart).toHaveAttribute("href", "/welcome")
    // Following it leaves the SPA, so it must not replace the dashboard tab.
    expect(quickstart).toHaveAttribute("target", "_blank")
    expect(quickstart).toHaveAttribute("rel", "noreferrer")
  })

  it("disables adding providers when OTARI_SECRET_KEY is not set", async () => {
    mockApi({
      stored: [storedProvider("openai", "1234")],
      context: organizationContext({
        provider_key_encryption_available: false,
      }),
    })
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    // The button starts enabled and flips once the context resolves, so wait
    // for the settled disabled state rather than asserting on first paint.
    expect(await screen.findByText(/OTARI_SECRET_KEY/)).toBeInTheDocument()
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Add provider" }),
      ).toBeDisabled(),
    )
  })

  it("disables the first-run add button when OTARI_SECRET_KEY is not set", async () => {
    mockApi({
      meta: [],
      stored: [],
      context: organizationContext({
        provider_key_encryption_available: false,
      }),
    })
    renderPage(<ProvidersPage />)

    expect(await screen.findByText("Welcome to Otari")).toBeInTheDocument()
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Add your first provider" }),
      ).toBeDisabled(),
    )
  })

  it("keeps adding providers available when the operator-only settings read is refused", async () => {
    // #839: the gate used to be inferred from /api/v1/settings, which is
    // operator-only, so a refusal reported a missing key on a deployment that
    // has one.
    mockApi({
      stored: [storedProvider("openai", "1234")],
      settingsRefused: true,
    })
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Add provider" }),
      ).toBeEnabled(),
    )
    expect(screen.queryByText(/OTARI_SECRET_KEY/)).toBeNull()
    // The page reads settings only for the pricing hint, so its refusal must not
    // leave a permanent "Not authorized" alert on a page that otherwise works.
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("fails closed and disables adding providers when the context can't be loaded", async () => {
    mockApi({ stored: [storedProvider("openai", "1234")], contextError: true })
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    // A context error leaves the key state unknown; disable rather than let the
    // operator fill in the form and fail on submit.
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Add provider" }),
      ).toBeDisabled(),
    )
    // Report the read that actually failed. Claiming the key is unset would be a
    // guess, and the wrong one whenever the deployment has one.
    expect(await screen.findByRole("alert")).toHaveTextContent("boom")
    expect(screen.queryByText(/OTARI_SECRET_KEY/)).toBeNull()
  })

  it("retracts an open add form if the context then reports OTARI_SECRET_KEY is unset", async () => {
    let releaseContext = () => {}
    const contextGate = new Promise<void>((resolve) => {
      releaseContext = resolve
    })
    // The onboarding gate ignores the context loading, so the first-run card (and
    // its enabled add button) is reachable before /v1/organizations/me resolves.
    mockApi({
      meta: [],
      stored: [],
      context: organizationContext({
        provider_key_encryption_available: false,
      }),
      contextGate,
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add your first provider" }),
    )
    expect(screen.getByPlaceholderText("Search providers…")).toBeInTheDocument()

    // The context lands late and reports the key is unavailable: the form must
    // retract so its submit can never reach the create mutation.
    releaseContext()
    await waitFor(() =>
      expect(
        screen.queryByPlaceholderText("Search providers…"),
      ).not.toBeInTheDocument(),
    )
    expect(screen.getByText(/OTARI_SECRET_KEY/)).toBeInTheDocument()
  })

  it("hides the onboarding once a provider exists", async () => {
    mockApi({ stored: [storedProvider("openai", "1234")] })
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    expect(screen.queryByText("Welcome to Otari")).not.toBeInTheDocument()
  })

  it("flags a stored provider whose key can't be decrypted", async () => {
    mockApi({ stored: [storedProvider("home-lab", "0000", false)] })
    renderPage(<ProvidersPage />)

    expect(await screen.findByText(/key unreadable/)).toBeInTheDocument()
    // Test is disabled for an unreadable key; Edit/Delete remain to recover it.
    expect(screen.getByRole("button", { name: "Test" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Edit" })).toBeEnabled()
  })

  it("shows each provider's reachability, including config-only providers", async () => {
    mockApi({
      meta: [providerInfo("openai"), providerInfo("anthropic")],
      health: [
        {
          instance: "openai",
          ok: true,
          model_count: 12,
          error: null,
          checked_at: "2026-07-21T00:00:00+00:00",
          discovery_unsupported: false,
        },
        {
          instance: "anthropic",
          ok: false,
          model_count: 0,
          error: "authentication failed: invalid key",
          checked_at: "2026-07-21T00:00:00+00:00",
          discovery_unsupported: false,
        },
      ],
    })
    renderPage(<ProvidersPage />)

    // Scope by the status pill's row (the provider name repeats in the Type cell).
    const reachableRow = (await screen.findByText("Reachable")).closest("tr")!
    expect(within(reachableRow).getAllByText("openai").length).toBeGreaterThan(
      0,
    )

    const unreachablePill = screen.getByText("Unreachable")
    const unreachableRow = unreachablePill.closest("tr")!
    expect(
      within(unreachableRow).getAllByText("anthropic").length,
    ).toBeGreaterThan(0)
    // The provider error rides along as the pill's tooltip.
    expect(unreachablePill).toHaveAttribute(
      "title",
      expect.stringContaining("authentication failed"),
    )
  })

  it("summarizes how many providers are reachable", async () => {
    mockApi({
      meta: [providerInfo("openai"), providerInfo("anthropic")],
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
          instance: "anthropic",
          ok: false,
          model_count: 0,
          error: "down",
          checked_at: null,
          discovery_unsupported: false,
        },
      ],
    })
    renderPage(<ProvidersPage />)

    expect(
      await screen.findByText("1 of 2 providers reachable"),
    ).toBeInTheDocument()
  })

  it("links a provider name to the filtered models page", async () => {
    mockApi({
      meta: [providerInfo("openai"), providerInfo("anthropic")],
    })

    renderPage(<ProvidersPage />)

    // Clicking a provider navigates to the Models page filtered to that provider.
    const openaiLink = await screen.findByRole("link", { name: "openai" })
    expect(openaiLink).toHaveAttribute("href", "/models?provider=openai")

    const anthropicLink = screen.getByRole("link", { name: "anthropic" })
    expect(anthropicLink).toHaveAttribute("href", "/models?provider=anthropic")
  })
})
