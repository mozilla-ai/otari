import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { DeploymentBootstrap, ProviderHealth } from "@/client"
import { HostedProviderPage } from "@/features/hostedProviders/HostedProviderPage"
import {
  hostedModel,
  hostedProvider,
  hostedRequests,
  mockApi,
  providerInfo,
  renderHosted,
} from "@/tests/hostedProviders"
import { getRowMenuItem, pressRowAction } from "@/tests/rowMenu"

function renderProviderPage(
  name: string,
  deployment: Partial<DeploymentBootstrap> = {},
) {
  return renderHosted(
    <HostedProviderPage name={name} />,
    `/admin/hosted-providers/${name}`,
    deployment,
  )
}

const platform = () => hostedProvider({ name: "otari-ai", provider: "otari" })

afterEach(() => {
  vi.restoreAllMocks()
})

describe("HostedProviderPage", () => {
  it("tests a stored provider from its row, on what it stores", async () => {
    const requests = mockApi({
      providers: [hostedProvider({ name: "openai-eu" })],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai-eu")

    await user.click(
      await screen.findByRole("button", { name: "Test openai-eu" }),
    )

    expect(
      await screen.findByText(/Connected\. 3 models available\./),
    ).toBeInTheDocument()
    expect(hostedRequests(requests, "POST")).toEqual([
      expect.objectContaining({
        url: expect.stringContaining("/hosted-providers/test"),
        body: { provider: "openai", name: "openai-eu" },
      }),
    ])
  })

  it("offers no test for a key no secret reads", async () => {
    mockApi({ providers: [hostedProvider({ decryptable: false })] })
    await renderProviderPage("openai")

    expect(
      await screen.findByRole("button", { name: "Test openai" }),
    ).toBeDisabled()
    expect(screen.getByText("Key unreadable")).toBeInTheDocument()
  })

  it("switches serving models not on its list from the edit dialog", async () => {
    const requests = mockApi({
      providers: [hostedProvider({ serves_unlisted_models: true })],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    const dialog = await screen.findByRole("dialog")
    const box = within(dialog).getByRole("checkbox", {
      name: "Serve models not on its list",
    })
    expect(box).toBeChecked()
    await user.click(box)
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "PATCH")[0].body).toEqual({
      api_base: "",
      serves_unlisted_models: false,
    })
  })

  it("marks a provider declared zero data retention and lets an edit lift it", async () => {
    const requests = mockApi({
      providers: [hostedProvider({ zero_data_retention: true })],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    expect(await screen.findByText("Zero data retention")).toBeInTheDocument()
    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    const dialog = await screen.findByRole("dialog")
    const box = within(dialog).getByRole("checkbox", {
      name: "Zero data retention",
    })
    expect(box).toBeChecked()
    await user.click(box)
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "PATCH")[0].body).toEqual({
      api_base: "",
      zero_data_retention: false,
    })
  })

  it("shows the routing locality and lets an edit change and clear it", async () => {
    const requests = mockApi({
      providers: [hostedProvider({ routing_locality: "cloud" })],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    expect(await screen.findByText("Routing locality")).toBeInTheDocument()
    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    let dialog = await screen.findByRole("dialog")
    await user.click(
      within(dialog).getByRole("button", { name: /Routing locality/ }),
    )
    await user.click(await screen.findByRole("option", { name: /^Private/ }))
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "PATCH")[0].body).toEqual({
      api_base: "",
      routing_locality: "private",
    })

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    dialog = await screen.findByRole("dialog")
    await user.click(
      within(dialog).getByRole("button", { name: /Routing locality/ }),
    )
    await user.click(
      await screen.findByRole("option", { name: /^Not declared/ }),
    )
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(2),
    )
    expect(hostedRequests(requests, "PATCH")[1].body).toEqual({
      api_base: "",
      routing_locality: null,
    })
  })

  it("changes and clears the countries from the edit dialog", async () => {
    const requests = mockApi({
      providers: [hostedProvider({ supported_countries: ["US"] })],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    let dialog = await screen.findByRole("dialog")
    let countries = within(dialog).getByRole("textbox", {
      name: "Available to requests from",
    })
    expect(countries).toHaveValue("US")
    await user.clear(countries)
    await user.type(countries, "gb")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "PATCH")[0].body).toEqual({
      api_base: "",
      supported_countries: ["GB"],
    })

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    dialog = await screen.findByRole("dialog")
    countries = within(dialog).getByRole("textbox", {
      name: "Available to requests from",
    })
    await user.clear(countries)
    await user.click(within(dialog).getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(2),
    )
    expect(hostedRequests(requests, "PATCH")[1].body).toEqual({
      api_base: "",
      supported_countries: null,
    })
  })

  it("edits a provider without resending its countries", async () => {
    const requests = mockApi({
      providers: [hostedProvider({ supported_countries: ["US", "DE"] })],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    const dialog = await screen.findByRole("dialog")
    await user.click(
      within(dialog).getByRole("checkbox", {
        name: "Serve models not on its list",
      }),
    )
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "PATCH")[0].body).toEqual({
      api_base: "",
      serves_unlisted_models: true,
    })
  })

  it("shows how the provider is reached and who it serves", async () => {
    mockApi({
      providers: [
        hostedProvider({
          api_key_last4: "wxyz",
          api_base: "https://eu.example.com/v1",
          restricted_to_organization_id: "33333333-3333-4333-8333-333333333333",
          supported_countries: ["US", "DE"],
        }),
      ],
    })
    await renderProviderPage("openai")

    const settings = await screen.findByLabelText("openai settings")
    expect(within(settings).getByText("••••wxyz")).toBeInTheDocument()
    expect(
      within(settings).getByText("https://eu.example.com/v1"),
    ).toBeInTheDocument()
    expect(within(settings).getByText("One organization")).toBeInTheDocument()
    expect(
      within(settings).getByText("33333333-3333-4333-8333-333333333333"),
    ).toBeInTheDocument()
    expect(within(settings).getByText("US, DE")).toBeInTheDocument()
  })

  it("serves every organization and country unless restricted", async () => {
    mockApi()
    await renderProviderPage("openai")

    const settings = await screen.findByLabelText("openai settings")
    expect(within(settings).getByText("Every organization")).toBeInTheDocument()
    expect(within(settings).getByText("Every country")).toBeInTheDocument()
  })

  it("edits a provider without resending its key or its restriction", async () => {
    const requests = mockApi({
      providers: [
        hostedProvider({
          api_base: "https://proxy.example.com/v1",
          restricted_to_organization_id: "33333333-3333-4333-8333-333333333333",
        }),
      ],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    const dialog = await screen.findByRole("dialog")
    const base = within(dialog).getByRole("textbox", { name: "API base URL" })
    await user.clear(base)
    await user.type(base, "https://proxy.example.com/v2")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    const patch = hostedRequests(requests, "PATCH")[0]
    expect(patch.url).toContain("/hosted-providers/openai")
    // The API showed no arguments, so none are sent: null would clear any it
    // could not read back.
    expect(patch.body).toEqual({ api_base: "https://proxy.example.com/v2" })
  })

  it("asks for the key again before pointing a provider at another host", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    await renderProviderPage("openai")

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    const dialog = await screen.findByRole("dialog")
    await user.type(
      within(dialog).getByRole("textbox", { name: "API base URL" }),
      "https://proxy.example.com/v1",
    )
    expect(
      within(dialog).getByText(
        "This endpoint is on another host, scheme or port, so enter the key again.",
      ),
    ).toBeInTheDocument()
    expect(within(dialog).getByRole("button", { name: "Save" })).toBeDisabled()

    await user.type(within(dialog).getByLabelText(/API key/), "sk-new")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "PATCH")[0].body).toMatchObject({
      api_key: "sk-new",
      api_base: "https://proxy.example.com/v1",
    })
  })

  it("edits a provider's links, sending a cleared one as null", async () => {
    const requests = mockApi({
      providers: [
        hostedProvider({
          terms_url: "https://openai.example/terms",
          privacy_url: "https://openai.example/privacy",
        }),
      ],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    const dialog = await screen.findByRole("dialog")
    const terms = within(dialog).getByRole("textbox", {
      name: "Terms of service URL",
    })
    expect(terms).toHaveValue("https://openai.example/terms")
    await user.clear(terms)
    await user.type(terms, "https://openai.example/terms-v2")
    await user.clear(
      within(dialog).getByRole("textbox", { name: "Privacy policy URL" }),
    )
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "PATCH")[0].body).toEqual({
      api_base: "",
      terms_url: "https://openai.example/terms-v2",
      privacy_url: null,
    })
  })

  it("names the server's refusal of a link in the edit dialog", async () => {
    mockApi({
      patchRefusalStatus: 422,
      patchRefusal: [
        {
          loc: ["body", "terms_url"],
          msg: "URL scheme should be 'http' or 'https'",
          type: "url_scheme",
        },
      ],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    const dialog = await screen.findByRole("dialog")
    await user.type(
      within(dialog).getByRole("textbox", { name: "Terms of service URL" }),
      "https://openai.example/terms",
    )
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    expect(
      await within(dialog).findByText(/URL scheme should be/),
    ).toBeInTheDocument()
    expect(screen.getByRole("dialog")).toBeInTheDocument()
  })

  it("keeps the edit open on the server's refusal, naming it", async () => {
    mockApi({
      patchRefusal:
        "Pointing a hosted provider at another origin requires entering its key again",
    })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    expect(
      await within(dialog).findByText(
        /Pointing a hosted provider at another origin requires entering its key again/,
      ),
    ).toBeInTheDocument()
    expect(screen.getByRole("dialog")).toBeInTheDocument()
  })

  it("removes a provider only once the confirm is pressed", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    await renderProviderPage("openai")

    await user.click(
      await screen.findByRole("button", { name: "Remove openai" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(hostedRequests(requests, "DELETE")).toEqual([])
    await user.click(
      within(dialog).getByRole("button", { name: "Remove provider" }),
    )

    await waitFor(() =>
      expect(hostedRequests(requests, "DELETE")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "DELETE")[0].url).toContain(
      "/hosted-providers/openai",
    )
  })

  it("shows one kind of model at a time, naming a price's unit", async () => {
    const requests = mockApi({
      models: [
        hostedModel(),
        hostedModel({
          id: "m2",
          model: "gpt-image-1",
          type: "image",
          unit: "image",
          input_price_per_million: 40_000,
          output_price_per_million: null,
        }),
      ],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    const panel = await screen.findByRole("region", {
      name: "Models on openai",
    })
    await within(panel).findByText("gpt-image-1")
    await user.click(within(panel).getByRole("button", { name: "Image 1" }))

    await waitFor(() => expect(within(panel).queryByText("gpt-4o")).toBeNull())
    expect(
      within(panel).getByText(/Image output: \$0\.04\/image/).parentElement,
    ).toHaveTextContent("$0.04")
    expect(
      requests.some(
        (request) =>
          request.url.includes("/hosted-providers/openai/models?") &&
          request.url.includes("type=image"),
      ),
    ).toBe(true)
  })

  it("searches a provider's models by name", async () => {
    const requests = mockApi({
      models: [hostedModel(), hostedModel({ id: "m2", model: "gpt-image-1" })],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    const panel = await screen.findByRole("region", {
      name: "Models on openai",
    })
    await within(panel).findByText("gpt-image-1")
    await user.type(
      within(panel).getByRole("searchbox", {
        name: "Search models on openai",
      }),
      "IMAGE",
    )

    await waitFor(() => expect(within(panel).queryByText("gpt-4o")).toBeNull())
    expect(within(panel).getByText("gpt-image-1")).toBeInTheDocument()
    expect(
      requests.some(
        (request) =>
          request.url.includes("/hosted-providers/openai/models?") &&
          request.url.includes("search=IMAGE"),
      ),
    ).toBe(true)

    await user.clear(
      within(panel).getByRole("searchbox", {
        name: "Search models on openai",
      }),
    )
    await user.type(
      within(panel).getByRole("searchbox", {
        name: "Search models on openai",
      }),
      "nothing",
    )
    expect(
      await within(panel).findByText("No models match this search."),
    ).toBeInTheDocument()
    expect(within(panel).queryByText("No models offered yet")).toBeNull()
  })

  it("shows only the models a provider does not serve", async () => {
    const requests = mockApi({
      models: [
        hostedModel(),
        hostedModel({ id: "m2", model: "gpt-image-1", enabled: false }),
      ],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    const panel = await screen.findByRole("region", {
      name: "Models on openai",
    })
    await within(panel).findByText("gpt-4o")
    await user.click(within(panel).getByRole("radio", { name: "Not serving" }))

    await waitFor(() => expect(within(panel).queryByText("gpt-4o")).toBeNull())
    expect(within(panel).getByText("gpt-image-1")).toBeInTheDocument()
    expect(
      requests.some(
        (request) =>
          request.url.includes("/hosted-providers/openai/models?") &&
          request.url.includes("enabled=false"),
      ),
    ).toBe(true)
  })

  it("opens a provider's models with their price source and a serving switch", async () => {
    const requests = mockApi({
      models: [
        hostedModel(),
        hostedModel({
          id: "m2",
          model: "o9-preview",
          input_price_per_million: null,
          output_price_per_million: null,
          unit: null,
          price_source: null,
          enabled: false,
        }),
      ],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    const panel = await screen.findByRole("region", {
      name: "Models on openai",
    })
    expect(await within(panel).findByText("gpt-4o")).toBeInTheDocument()
    expect(within(panel).getByText("Default")).toBeInTheDocument()
    expect(within(panel).getByText("Unpriced")).toBeInTheDocument()

    await user.click(
      within(panel).getByRole("switch", { name: "Serve gpt-4o" }),
    )
    await waitFor(() =>
      expect(
        hostedRequests(requests, "PATCH").filter((request) =>
          request.url.includes("/hosted-providers/openai/models/m1"),
        ),
      ).toHaveLength(1),
    )
  })

  it("marks a model its provider refused as unavailable, and lets it be switched back on", async () => {
    const requests = mockApi({
      models: [
        hostedModel({ enabled: false, disabled_reason: "unavailable" }),
        hostedModel({
          id: "m2",
          model: "o9-preview",
          enabled: false,
          disabled_reason: "operator",
        }),
      ],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    const panel = await screen.findByRole("region", {
      name: "Models on openai",
    })
    expect(await within(panel).findAllByText("Unavailable")).toHaveLength(1)

    await user.click(
      within(panel).getByRole("switch", { name: "Serve gpt-4o" }),
    )
    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    const [patch] = hostedRequests(requests, "PATCH")
    expect(patch.url).toContain("/hosted-providers/openai/models/m1")
    expect(patch.body).toEqual({ enabled: true })
  })

  it("lists a provider defined in config.yml read-only, with its models still managed here", async () => {
    const requests = mockApi({
      providers: [
        hostedProvider({
          id: null,
          provider: "anthropic",
          source: "config",
          created_at: null,
        }),
      ],
    })
    const user = userEvent.setup()
    await renderProviderPage("anthropic")

    expect(await screen.findByText("config.yml")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Test anthropic" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Edit anthropic" })).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Remove anthropic" }),
    ).toBeNull()
    expect(
      screen.getByRole("switch", { name: "Serve traffic on anthropic" }),
    ).toBeDisabled()

    expect(screen.getByText(/defined in config\.yml/)).toBeInTheDocument()
    const panel = await screen.findByRole("region", {
      name: "Models on anthropic",
    })
    await user.click(
      await within(panel).findByRole("switch", { name: "Serve gpt-4o" }),
    )
    await waitFor(() =>
      expect(
        hostedRequests(requests, "PATCH").filter((request) =>
          request.url.includes("/hosted-providers/anthropic/models/m1"),
        ),
      ).toHaveLength(1),
    )
  })

  it("marks a stored provider that overrides config.yml, and says what removing it restores", async () => {
    mockApi({ providers: [hostedProvider({ shadows_config: true })] })
    const user = userEvent.setup()
    await renderProviderPage("openai")

    expect(await screen.findByText("Overrides config.yml")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Remove openai" }))
    const dialog = await screen.findByRole("alertdialog")
    expect(
      within(dialog).getByText(/defined in config\.yml serves again/),
    ).toBeInTheDocument()
  })

  it("shows no status for a provider a configured instance of its name is dialed under", async () => {
    const reachable = (instance: string): ProviderHealth => ({
      instance,
      ok: true,
      model_count: 1,
      error: null,
      checked_at: "2026-10-19T00:00:00Z",
      discovery_unsupported: false,
    })
    mockApi({
      providers: [
        hostedProvider({ name: "openai" }),
        hostedProvider({ id: "m", name: "mistral", provider: "mistral" }),
      ],
      configured: [providerInfo("openai")],
      health: [reachable("openai"), reachable("mistral")],
    })
    const shadowed = await renderProviderPage("openai")
    await screen.findByLabelText("openai settings")
    expect(screen.queryByText("Reachable")).toBeNull()
    shadowed.unmount()

    await renderProviderPage("mistral")
    expect(await screen.findByText("Reachable")).toBeInTheDocument()
  })

  it("tests an edit on the stored key without asking for it", async () => {
    const requests = mockApi({
      providers: [
        hostedProvider({
          name: "openai-eu",
          api_base: "https://eu.example.com/v1",
        }),
      ],
    })
    const user = userEvent.setup()
    await renderProviderPage("openai-eu")

    await user.click(
      await screen.findByRole("button", { name: "Edit openai-eu" }),
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(
      within(dialog).getByRole("button", { name: "Test connection" }),
    )

    await within(dialog).findByText(/Connected\./)
    const [test] = hostedRequests(requests, "POST")
    expect(test.url).toContain("/hosted-providers/test")
    expect(test.body).toEqual({
      provider: "openai",
      name: "openai-eu",
      api_base: "https://eu.example.com/v1",
    })
  })

  it("points at otari.ai's keys page when the connected provider is removed", async () => {
    mockApi({ providers: [platform()] })
    const user = userEvent.setup()
    await renderProviderPage("otari-ai", { platform_connect: true })

    await user.click(
      await screen.findByRole("button", { name: "Remove otari-ai" }),
    )

    const dialog = await screen.findByRole("alertdialog")
    expect(
      within(dialog).getByRole("link", { name: "API keys page" }),
    ).toHaveAttribute("href", "https://otari.ai/dashboard/#/keys")
  })

  it("names a default rate on an otari provider an estimate, and a listed one listed", async () => {
    mockApi({
      providers: [platform()],
      models: [
        hostedModel({ id: "m1", model: "openai/gpt-4o" }),
        hostedModel({
          id: "m2",
          model: "openai/gpt-4o-mini",
          price_source: "provider",
        }),
      ],
    })
    await renderProviderPage("otari-ai")

    const panel = await screen.findByRole("region", {
      name: "Models on otari-ai",
    })
    expect(await within(panel).findByText("Estimate")).toBeInTheDocument()
    expect(within(panel).getByText("Listed")).toBeInTheDocument()
    expect(within(panel).queryByText("Default")).not.toBeInTheDocument()
  })

  it("asks again rather than reading not-found from a list cached before the provider was added", async () => {
    mockApi({ providers: [hostedProvider({ name: "added-elsewhere" })] })
    renderHosted(
      <HostedProviderPage name="added-elsewhere" />,
      "/admin/hosted-providers/added-elsewhere",
      {},
      [],
      (client) =>
        client.setQueryData(["hosted-providers", "list"], {
          data: [],
          count: 0,
        }),
    )

    expect(
      await screen.findByRole("heading", { name: "added-elsewhere" }),
    ).toBeInTheDocument()
    expect(
      screen.queryByText("No hosted provider named added-elsewhere"),
    ).not.toBeInTheDocument()
  })

  it("says so when no hosted provider has the name, with the way back", async () => {
    mockApi()
    await renderProviderPage("nope")

    expect(
      await screen.findByText("No hosted provider named nope"),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "← Hosted providers" }),
    ).toHaveAttribute("href", "/admin/hosted-providers")
  })

  it("refuses a caller who does not operate the deployment", async () => {
    const requests = mockApi({ granted: false })
    await renderProviderPage("openai")

    expect(
      await screen.findByText("Hosted providers is not available to you"),
    ).toBeInTheDocument()
    expect(hostedRequests(requests)).toEqual([])
  })

  describe("vendors", () => {
    const together = () =>
      hostedProvider({ name: "together", provider: "together" })
    const routed = () => [
      hostedModel({ id: "q1", model: "Qwen/Qwen3.5-9B", vendor: "Alibaba" }),
      hostedModel({
        id: "q2",
        model: "Qwen/Qwen-Image-2.0",
        vendor: "Alibaba",
        price_source: "deployment",
        input_price_per_million: 0.03,
        output_price_per_million: 0.03,
        default_rates: {
          source: "defaults",
          input_price_per_million: 0.035,
          output_price_per_million: 0.035,
          unit: "token",
        },
      }),
      hostedModel({
        id: "d1",
        model: "deepseek-ai/DeepSeek-V3.2",
        vendor: "DeepSeek",
        enabled: false,
        disabled_reason: "unpriced",
      }),
    ]

    it("groups the models under their vendors, closed until opened", async () => {
      mockApi({ providers: [together()], models: routed() })
      const user = userEvent.setup()
      await renderProviderPage("together")

      const vendors = await screen.findByRole("list", {
        name: "Vendors on together",
      })
      expect(within(vendors).getByText("Alibaba")).toBeInTheDocument()
      expect(within(vendors).getByText("2 of 2 serving")).toBeInTheDocument()
      expect(within(vendors).getByText("1 unpriced")).toBeInTheDocument()
      expect(
        within(vendors).getByText("1 custom ≠ default"),
      ).toBeInTheDocument()
      expect(screen.queryByText("Qwen/Qwen3.5-9B")).toBeNull()

      await user.click(
        within(vendors).getByRole("button", { name: "Show Alibaba models" }),
      )
      const table = await screen.findByRole("grid", {
        name: "Alibaba models on together",
      })
      expect(within(table).getByText("Qwen/Qwen3.5-9B")).toBeInTheDocument()
      expect(screen.queryByText("deepseek-ai/DeepSeek-V3.2")).toBeNull()
    })

    it("keeps the vendor row of a provider's only vendor while it is off, its one way back on", async () => {
      mockApi({
        providers: [together()],
        models: routed().filter((model) => model.vendor === "Alibaba"),
        vendorsOff: ["Alibaba"],
      })
      await renderProviderPage("together")

      expect(
        await screen.findByRole("switch", { name: "Serve Alibaba models" }),
      ).not.toBeChecked()
    })

    it("gives a provider's only vendor its row and menu, open from the start", async () => {
      mockApi({
        providers: [together()],
        models: routed().filter((model) => model.vendor === "Alibaba"),
      })
      await renderProviderPage("together")

      expect(
        await screen.findByRole("switch", { name: "Serve Alibaba models" }),
      ).toBeChecked()
      expect(
        screen.getByRole("button", { name: "More for Alibaba" }),
      ).toBeEnabled()
      expect(
        await screen.findByRole("grid", { name: "Alibaba models on together" }),
      ).toBeInTheDocument()
    })

    it("lists a vendor switched off that no model goes by, so it can be switched back on", async () => {
      mockApi({
        providers: [together()],
        models: routed(),
        vendorsOff: ["Old Name"],
      })
      await renderProviderPage("together")

      const vendors = await screen.findByRole("list", {
        name: "Vendors on together",
      })
      expect(within(vendors).getByText("No models")).toBeInTheDocument()
      expect(
        within(vendors).getByRole("switch", { name: "Serve Old Name models" }),
      ).not.toBeChecked()
      expect(
        within(vendors).getByRole("button", { name: "More for Old Name" }),
      ).toBeDisabled()
    })

    it("offers the first refresh on a provider with no models and no vendor switched off", async () => {
      mockApi({ providers: [together()], models: [] })
      await renderProviderPage("together")

      expect(
        await screen.findByText("No models offered yet"),
      ).toBeInTheDocument()
    })

    it("keeps a vendor switched off in view on a provider with no models left", async () => {
      mockApi({ providers: [together()], models: [], vendorsOff: ["Old Name"] })
      await renderProviderPage("together")

      expect(
        await screen.findByRole("switch", { name: "Serve Old Name models" }),
      ).not.toBeChecked()
      expect(screen.queryByText("No models offered yet")).toBeNull()
    })

    it("says so when a provider offers more models than the list reads", async () => {
      mockApi({ providers: [together()], models: routed(), modelCount: 1500 })
      await renderProviderPage("together")

      expect(
        await screen.findByText(/Showing the first \d+ of 1500 models/),
      ).toBeInTheDocument()
    })

    it("switches a vendor off with one request, holding its models' switches", async () => {
      const requests = mockApi({
        providers: [together()],
        models: routed(),
        vendorsOff: ["DeepSeek"],
      })
      const user = userEvent.setup()
      await renderProviderPage("together")

      await user.click(
        await screen.findByRole("switch", { name: "Serve Alibaba models" }),
      )
      await waitFor(() =>
        expect(hostedRequests(requests, "PATCH")).toEqual([
          expect.objectContaining({
            url: expect.stringContaining("/hosted-providers/together/vendors"),
            body: { vendor: "Alibaba", enabled: false },
          }),
        ]),
      )

      // A vendor already off shows why its models do not serve.
      await user.click(
        screen.getByRole("button", { name: "Show DeepSeek models" }),
      )
      expect(screen.getByText("Vendor off")).toBeInTheDocument()
      expect(
        screen.getByText(/DeepSeek is switched off on this provider/),
      ).toBeInTheDocument()
      expect(
        screen.getByRole("switch", { name: "Serve deepseek-ai/DeepSeek-V3.2" }),
      ).toBeDisabled()
      // A model's price stays editable while its vendor is off.
      expect(
        await getRowMenuItem(user, "Edit deepseek-ai/DeepSeek-V3.2 price"),
      ).not.toHaveAttribute("aria-disabled", "true")
    })

    it("serves every priced model of a vendor from its menu, saying what it skipped", async () => {
      const requests = mockApi({ providers: [together()], models: routed() })
      const user = userEvent.setup()
      await renderProviderPage("together")

      await user.click(
        await screen.findByRole("button", { name: "More for DeepSeek" }),
      )
      await user.click(
        await screen.findByRole("menuitem", { name: "Serve all models" }),
      )

      expect(
        await screen.findByText(
          "Served 2 more models by DeepSeek. Skipped 1 without a price.",
        ),
      ).toBeInTheDocument()
      expect(
        hostedRequests(requests, "POST").filter((request) =>
          request.url.includes("/models/switch"),
        ),
      ).toEqual([
        expect.objectContaining({
          body: { vendor: "DeepSeek", enabled: true, only_unpriced: false },
        }),
      ])
    })

    it("puts one custom price back on the default from its row", async () => {
      const requests = mockApi({ providers: [together()], models: routed() })
      const user = userEvent.setup()
      await renderProviderPage("together")

      await user.click(
        await screen.findByRole("button", { name: "Show Alibaba models" }),
      )
      expect(
        await screen.findByText(/models\.dev \$0\.035 in/),
      ).toBeInTheDocument()
      await pressRowAction(
        user,
        "Use the default price for Qwen/Qwen-Image-2.0",
      )

      await waitFor(() =>
        expect(
          hostedRequests(requests, "POST").filter((request) =>
            request.url.includes("/models/reset-prices"),
          ),
        ).toEqual([
          expect.objectContaining({
            body: { models: ["Qwen/Qwen-Image-2.0"], dry_run: false },
          }),
        ]),
      )
    })

    it("previews a price sync, then resets only the ticked models", async () => {
      const current = {
        source: "deployment" as const,
        input_price_per_million: 2,
        output_price_per_million: 2,
        unit: "token",
      }
      const requests = mockApi({
        providers: [together()],
        models: routed(),
        priceReset: {
          applied: false,
          skipped_no_default: ["custom/no-default"],
          changes: [
            {
              model: "Qwen/Qwen3-Coder",
              vendor: "Alibaba",
              current,
              default: {
                ...current,
                source: "defaults",
                input_price_per_million: 1.5,
                output_price_per_million: 1.5,
              },
              matches: false,
            },
            {
              model: "Qwen/Qwen-Image-2.0",
              vendor: "Alibaba",
              current,
              default: {
                ...current,
                source: "defaults",
                input_price_per_million: 0.035,
                output_price_per_million: 0.035,
              },
              matches: false,
            },
            {
              model: "moonshotai/Kimi-K2",
              vendor: "Moonshot AI",
              current,
              default: { ...current, source: "defaults" },
              matches: true,
            },
          ],
        },
      })
      const user = userEvent.setup()
      await renderProviderPage("together")

      await user.click(
        await screen.findByRole("button", { name: "Sync prices…" }),
      )
      const dialog = await screen.findByRole("dialog", { name: "Sync prices" })
      const differs = await within(dialog).findByRole("region", {
        name: "Differs from the default",
      })
      const matches = within(dialog).getByRole("region", {
        name: "Matches the default",
      })
      expect(within(differs).getAllByRole("checkbox")).toHaveLength(2)
      expect(within(matches).getByRole("checkbox")).not.toBeChecked()
      expect(
        within(dialog).getByText(/1 custom price has no default price/),
      ).toBeInTheDocument()

      await user.click(
        within(differs).getByRole("checkbox", { name: /Qwen\/Qwen3-Coder/ }),
      )
      await user.click(
        within(dialog).getByRole("button", { name: "Use default price for 1" }),
      )

      await waitFor(() =>
        expect(
          hostedRequests(requests, "POST")
            .filter((request) => request.url.includes("/models/reset-prices"))
            .map((request) => request.body),
        ).toEqual([
          { dry_run: true, vendor: null },
          { models: ["Qwen/Qwen-Image-2.0"], dry_run: false },
        ]),
      )
      expect(
        await screen.findByText("Reset 1 model to the default price."),
      ).toBeInTheDocument()
    })

    it("names a ticked model the server kept, its default gone since the preview", async () => {
      const current = {
        source: "deployment" as const,
        input_price_per_million: 2,
        output_price_per_million: 2,
        unit: "token",
      }
      const change = (model: string) => ({
        model,
        vendor: "Alibaba",
        current,
        default: {
          ...current,
          source: "defaults" as const,
          input_price_per_million: 1,
        },
        matches: false,
      })
      mockApi({
        providers: [together()],
        models: routed(),
        priceReset: {
          applied: false,
          skipped_no_default: [],
          changes: [change("Qwen/Qwen3-Coder"), change("Qwen/Qwen-Image-2.0")],
        },
        lostDefaults: ["Qwen/Qwen-Image-2.0"],
      })
      const user = userEvent.setup()
      await renderProviderPage("together")

      await user.click(
        await screen.findByRole("button", { name: "Sync prices…" }),
      )
      const dialog = await screen.findByRole("dialog", { name: "Sync prices" })
      await user.click(
        await within(dialog).findByRole("button", {
          name: "Use default price for 2",
        }),
      )

      expect(
        await screen.findByText(
          "Reset 1 model to the default price. Kept the price of Qwen/Qwen-Image-2.0: no default price now.",
        ),
      ).toBeInTheDocument()
    })
  })
})
