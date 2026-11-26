import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { DeploymentBootstrap } from "@/client"
import { HostedProvidersPage } from "@/features/hostedProviders/HostedProvidersPage"
import { WelcomeDialog } from "@/features/welcome/WelcomeDialog"
import {
  OWN_KEYS,
  WELCOME_ANSWERED_KEY,
} from "@/features/welcome/welcomeAnswer"
import { organizationContext } from "@/tests/fixtures"
import {
  countriesFileInput,
  EMPTY_SWEEP,
  hostedProvider,
  hostedRegion,
  hostedRequests,
  jsonResponse,
  mockApi,
  PRICES,
  providerInfo,
  type Request,
  renderHosted,
} from "@/tests/hostedProviders"

function renderPage(deployment: Partial<DeploymentBootstrap> = {}) {
  return renderHosted(
    <HostedProvidersPage />,
    "/admin/hosted-providers",
    deployment,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
  localStorage.clear()
})

describe("HostedProvidersPage", () => {
  it("lists each provider with how much of its catalog it serves", async () => {
    mockApi({
      providers: [
        hostedProvider({
          models: { total: 18, serving: 12, unpriced: 3, vendors_off: 1 },
        }),
        hostedProvider({
          id: "22222222-2222-4222-8222-222222222222",
          provider: "mistral",
          models: { total: 0, serving: 0, unpriced: 0, vendors_off: 0 },
        }),
      ],
    })
    await renderPage()

    const table = await screen.findByRole("grid", { name: "Hosted providers" })
    const row = (name: string) =>
      within(table).getByRole("rowheader", { name: new RegExp(`^${name}`) })
        .parentElement as HTMLElement
    await within(table).findByText("12 of 18 serving")
    expect(within(row("openai")).getByText("1 vendor off")).toBeInTheDocument()
    expect(within(row("openai")).getByText("3 unpriced")).toBeInTheDocument()
    expect(within(row("mistral")).getByText("None yet")).toBeInTheDocument()
    // The key, endpoint and scope are the provider page's.
    expect(within(table).queryByText("••••abcd")).toBeNull()
  })

  it("says where each provider keeps data, and how many regions it serves from", async () => {
    mockApi({
      providers: [
        hostedProvider({
          data_location: "CN",
          regions: [
            hostedRegion({ slug: "cn", position: 0, data_location: "CN" }),
            hostedRegion({ slug: "sg", position: 1, data_location: "SG" }),
            hostedRegion({ slug: "us", position: 2, data_location: "US" }),
          ],
        }),
        hostedProvider({
          id: "22222222-2222-4222-8222-222222222222",
          provider: "mistral",
          data_location: "EU",
        }),
        hostedProvider({
          id: "33333333-3333-4333-8333-333333333333",
          provider: "anthropic",
        }),
      ],
    })
    await renderPage()

    const table = await screen.findByRole("grid", { name: "Hosted providers" })
    const row = (name: string) =>
      within(table).getByRole("rowheader", { name: new RegExp(`^${name}`) })
    await within(table).findByText("3 regions")
    expect(within(row("openai")).getByText("3 regions")).toBeInTheDocument()
    expect(
      within(row("openai")).getByText("Data: China, Singapore, United States"),
    ).toBeInTheDocument()
    expect(
      within(row("mistral")).getByText("Data: European Union"),
    ).toBeInTheDocument()
    expect(within(row("mistral")).queryByText(/regions$/)).toBeNull()
    expect(
      within(row("anthropic")).getByText("Data: not declared"),
    ).toBeInTheDocument()
  })

  it("refuses a caller who does not operate the deployment without asking for the list", async () => {
    const requests = mockApi({ granted: false })
    await renderPage()

    expect(
      await screen.findByText("Hosted providers is not available to you"),
    ).toBeInTheDocument()
    expect(hostedRequests(requests)).toEqual([])
    expect(screen.queryByRole("button", { name: "Add provider" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Update catalog" })).toBeNull()
  })

  it("refuses an operator the list itself 404s, without offering a write", async () => {
    mockApi({ listRefused: true })
    await renderPage()

    expect(
      await screen.findByText("Hosted providers is not available to you"),
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Add provider" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Update catalog" })).toBeNull()
  })

  it("walks the first provider in when none is configured", async () => {
    mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    expect(await screen.findByText("Welcome to Otari")).toBeInTheDocument()
    // The panel owns the empty state, so no empty table sits under it.
    expect(screen.queryByRole("grid", { name: "Hosted providers" })).toBeNull()
    expect(screen.queryByText(/Enable default pricing/)).toBeNull()

    await user.click(
      screen.getByRole("button", { name: "Add your first provider" }),
    )
    expect(
      await screen.findByRole("dialog", { name: "Add hosted provider" }),
    ).toBeInTheDocument()
  })

  it("offers default pricing on first run while pricing is required and unset", async () => {
    const requests = mockApi({
      providers: [],
      pricing: { require_pricing: true, default_pricing: false },
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Enable default pricing" }),
    )
    await waitFor(() =>
      expect(
        requests.filter(
          (request) =>
            request.url.endsWith("/settings") && request.method === "PATCH",
        ),
      ).toEqual([expect.objectContaining({ body: { default_pricing: true } })]),
    )
  })

  it("disables adding a provider while OTARI_SECRET_KEY is not set", async () => {
    mockApi({ providers: [], keyEncryption: false })
    await renderPage()

    expect(
      await screen.findByText(/is not set, so provider keys/),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add provider" })).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Add your first provider" }),
    ).toBeDisabled()
  })

  it("says nothing about OTARI_SECRET_KEY while it is set", async () => {
    mockApi()
    await renderPage()

    await screen.findByRole("grid", { name: "Hosted providers" })
    expect(screen.queryByText(/OTARI_SECRET_KEY/)).toBeNull()
    expect(screen.getByRole("button", { name: "Add provider" })).toBeEnabled()
  })

  it("links each provider to its own page", async () => {
    mockApi({
      providers: [hostedProvider({ name: "openai-eu" })],
    })
    await renderPage()

    expect(
      await screen.findByRole("link", { name: "openai-eu" }),
    ).toHaveAttribute("href", "/admin/hosted-providers/openai-eu")
    // Its test, edit and remove are on that page, not on the row.
    expect(screen.queryByRole("button", { name: "Edit openai-eu" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Test openai-eu" })).toBeNull()
  })

  it("opens a provider it just added", async () => {
    mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderHosted(<HostedProvidersPage />, "/admin/hosted-providers", {}, [
      {
        path: "/admin/hosted-providers/anthropic",
        element: <p>The anthropic page</p>,
      },
    ])

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(within(dialog).getByLabelText("API key"), "sk-ant-secret")
    await user.click(
      within(dialog).getByRole("button", { name: "Add provider" }),
    )

    expect(await screen.findByText("The anthropic page")).toBeInTheDocument()
  })

  it("summarizes reachability and re-checks every provider on demand", async () => {
    const requests = mockApi({
      health: [
        {
          instance: "openai",
          ok: true,
          model_count: 3,
          error: null,
          checked_at: "2026-10-19T00:00:00Z",
          discovery_unsupported: false,
        },
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    expect(
      await screen.findByText("1 of 1 provider reachable"),
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Re-check all" }))
    await waitFor(() =>
      expect(
        requests.some((request) =>
          request.url.endsWith("/providers/health?refresh=true"),
        ),
      ).toBe(true),
    )
  })

  it("adds a provider that serves models not on its list", async () => {
    const requests = mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(within(dialog).getByLabelText("API key"), "sk-ant-secret")
    await user.click(
      within(dialog).getByRole("checkbox", {
        name: "Serve models not on its list",
      }),
    )
    await user.click(
      within(dialog).getByRole("button", { name: "Add provider" }),
    )

    await waitFor(() =>
      expect(hostedRequests(requests, "POST")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "POST")[0].body).toMatchObject({
      serves_unlisted_models: true,
    })
  })

  it("adds a provider declared zero data retention", async () => {
    const requests = mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(within(dialog).getByLabelText("API key"), "sk-ant-secret")
    await user.click(
      within(dialog).getByRole("checkbox", { name: "Zero data retention" }),
    )
    await user.click(
      within(dialog).getByRole("button", { name: "Add provider" }),
    )

    await waitFor(() =>
      expect(hostedRequests(requests, "POST")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "POST")[0].body).toMatchObject({
      zero_data_retention: true,
    })
  })

  it("adds a provider that serves only some countries", async () => {
    const requests = mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(within(dialog).getByLabelText("API key"), "sk-ant-secret")
    await user.type(
      within(dialog).getByRole("textbox", {
        name: "Available to requests from",
      }),
      "us, de",
    )
    await user.click(
      within(dialog).getByRole("button", { name: "Add provider" }),
    )

    await waitFor(() =>
      expect(hostedRequests(requests, "POST")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "POST")[0].body).toMatchObject({
      supported_countries: ["US", "DE"],
    })
  })

  it("loads the countries from a JSON file", async () => {
    const requests = mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(within(dialog).getByLabelText("API key"), "sk-ant-secret")
    await user.upload(
      countriesFileInput(),
      new File(['["US", "DE", "GB"]'], "countries.json", {
        type: "application/json",
      }),
    )

    await waitFor(() =>
      expect(
        within(dialog).getByRole("textbox", {
          name: "Available to requests from",
        }),
      ).toHaveValue("US, DE, GB"),
    )
    await user.click(
      within(dialog).getByRole("button", { name: "Add provider" }),
    )
    await waitFor(() =>
      expect(hostedRequests(requests, "POST")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "POST")[0].body).toMatchObject({
      supported_countries: ["US", "DE", "GB"],
    })
  })

  it("says so when the countries file is not a JSON list", async () => {
    mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.upload(
      countriesFileInput(),
      new File(["US, DE"], "countries.json", { type: "application/json" }),
    )

    expect(
      await within(dialog).findByText(
        'The file is not a JSON list of country codes, such as ["US", "DE"].',
      ),
    ).toBeInTheDocument()
    expect(
      within(dialog).getByRole("textbox", {
        name: "Available to requests from",
      }),
    ).toHaveValue("")
  })

  it("refuses a countries file whose entries are not all codes, keeping the draft", async () => {
    mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    const field = within(dialog).getByRole("textbox", {
      name: "Available to requests from",
    })
    await user.type(field, "GB")
    // An empty entry would vanish from the text and read as every country.
    await user.upload(
      countriesFileInput(),
      new File(['[""]'], "countries.json", { type: "application/json" }),
    )
    expect(
      await within(dialog).findByText(
        'The file holds entries that are not two-letter country codes: ""',
      ),
    ).toBeInTheDocument()
    expect(field).toHaveValue("GB")

    await user.upload(
      countriesFileInput(),
      new File(['["US","DE"]'], "countries.json", {
        type: "application/json",
      }),
    )
    await waitFor(() => expect(field).toHaveValue("US, DE"))
  })

  it("holds back a country that is not a two-letter code", async () => {
    mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(within(dialog).getByLabelText("API key"), "sk-ant-secret")
    await user.type(
      within(dialog).getByRole("textbox", {
        name: "Available to requests from",
      }),
      "US, USA",
    )

    expect(
      within(dialog).getByText("Not a two-letter country code: USA"),
    ).toBeInTheDocument()
    expect(
      within(dialog).getByRole("button", { name: "Add provider" }),
    ).toBeDisabled()
  })

  it("adds a provider with its key, leaving a blank endpoint out", async () => {
    const requests = mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(within(dialog).getByLabelText("API key"), "sk-ant-secret")
    await user.click(
      within(dialog).getByRole("button", { name: "Add provider" }),
    )

    await waitFor(() =>
      expect(hostedRequests(requests, "POST")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "POST")[0].body).toEqual({
      provider: "anthropic",
      api_key: "sk-ant-secret",
      serves_unlisted_models: false,
      zero_data_retention: false,
    })
  })

  it("adds a provider with its terms and privacy links", async () => {
    const requests = mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(within(dialog).getByLabelText("API key"), "sk-ant-secret")
    await user.type(
      within(dialog).getByRole("textbox", { name: "Terms of service URL" }),
      "https://anthropic.example/terms",
    )
    await user.type(
      within(dialog).getByRole("textbox", { name: "Privacy policy URL" }),
      "https://anthropic.example/privacy",
    )
    await user.click(
      within(dialog).getByRole("button", { name: "Add provider" }),
    )

    await waitFor(() =>
      expect(hostedRequests(requests, "POST")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "POST")[0].body).toMatchObject({
      terms_url: "https://anthropic.example/terms",
      privacy_url: "https://anthropic.example/privacy",
    })
  })

  it("holds back an add whose terms link is not an http(s) address", async () => {
    mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(within(dialog).getByLabelText("API key"), "sk-ant-secret")
    await user.type(
      within(dialog).getByRole("textbox", { name: "Terms of service URL" }),
      "anthropic.example/terms",
    )

    expect(
      within(dialog).getByText("Enter a full http:// or https:// address."),
    ).toBeInTheDocument()
    expect(
      within(dialog).getByRole("button", { name: "Add provider" }),
    ).toBeDisabled()
  })

  it("switches a provider off from its row", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("switch", { name: "Serve traffic on openai" }),
    )

    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "PATCH")[0].body).toEqual({
      enabled: false,
    })
  })

  it("warns that a provider config.yml already defines would be overridden", async () => {
    mockApi({
      providers: [
        hostedProvider({ id: null, provider: "anthropic", source: "config" }),
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).queryByText(/already defined in config\.yml/),
    ).toBeNull()
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))

    expect(
      within(dialog).getByText(/already defined in config\.yml/),
    ).toBeInTheDocument()
  })

  const updates = (requests: Request[]) =>
    hostedRequests(requests, "POST").filter((request) =>
      request.url.endsWith("/catalog-update"),
    )
  const previewsOf = (requests: Request[]) =>
    hostedRequests(requests, "POST").filter((request) =>
      request.url.endsWith("/catalog-update/preview"),
    )

  it("applies the previewed prices by their digest, then says what it did", async () => {
    const requests = mockApi({
      prices: PRICES,
      sweep: {
        ...EMPTY_SWEEP,
        providers: [
          {
            provider: "openai",
            added: ["gpt-5-1017"],
            repriced: [],
            error: null,
            discovery_unsupported: false,
            credential_unreadable: false,
            source: "database",
          },
        ],
      },
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Update catalog" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(
      await within(dialog).findByText(
        "models.dev has 2 new, 1 changed prices to apply.",
      ),
    ).toBeInTheDocument()
    expect(
      within(dialog).getByText(
        "1 new model to offer on openai, each served once something prices it.",
      ),
    ).toBeInTheDocument()
    expect(updates(requests)).toEqual([])

    await user.click(
      within(dialog).getByRole("button", { name: "Update catalog" }),
    )
    await waitFor(() => expect(updates(requests)).toHaveLength(1))
    expect(updates(requests)[0].body).toEqual({
      digest: PRICES.digest,
      cleanup: false,
    })
    expect(
      await screen.findByText(
        "Applied the latest models.dev prices. Offered 1 new model.",
      ),
    ).toBeInTheDocument()
  })

  it("names what the cleanup would remove only once it is ticked, before removing anything", async () => {
    const requests = mockApi({
      sweep: {
        ...EMPTY_SWEEP,
        removed: ["mistral:old-model"],
        removed_count: 1,
        removed_price_rows: 2,
        removed_override_rows: 0,
      },
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Update catalog" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(
      await within(dialog).findByText("models.dev prices are unchanged."),
    ).toBeInTheDocument()
    expect(within(dialog).queryByText("mistral:old-model")).toBeNull()

    await user.click(
      within(dialog).getByRole("checkbox", {
        name: "Also remove the prices of models no provider serves",
      }),
    )
    expect(
      await within(dialog).findByText(
        "1 model will be removed, taking 2 stored rates and 0 organization overrides with it.",
      ),
    ).toBeInTheDocument()
    expect(within(dialog).getByText("mistral:old-model")).toBeInTheDocument()
    expect(updates(requests)).toEqual([])

    await user.click(
      within(dialog).getByRole("button", { name: "Update and remove" }),
    )
    await waitFor(() => expect(updates(requests)).toHaveLength(1))
    expect(updates(requests)[0].body).toEqual({ digest: null, cleanup: true })
    expect(
      await screen.findByText("Removed 1 model nothing here serves."),
    ).toBeInTheDocument()
  })

  it("asks for a fresh preview on every opening and holds the confirm until it lands", async () => {
    let answerSecond: (response: Response) => void = () => {}
    const requests = mockApi({
      previews: [
        async () =>
          jsonResponse({
            prices: PRICES,
            price_error: null,
            catalog: EMPTY_SWEEP,
          }),
        () =>
          new Promise<Response>((resolve) => {
            answerSecond = resolve
          }),
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Update catalog" }),
    )
    let dialog = await screen.findByRole("alertdialog")
    expect(
      await within(dialog).findByText(
        "models.dev has 2 new, 1 changed prices to apply.",
      ),
    ).toBeInTheDocument()
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull())

    await user.click(screen.getByRole("button", { name: "Update catalog" }))
    dialog = await screen.findByRole("alertdialog")
    // The last opening's preview is gone, and nothing can be confirmed yet.
    expect(
      within(dialog).getByText(
        "Asking models.dev and every provider what changed…",
      ),
    ).toBeInTheDocument()
    expect(
      within(dialog).queryByText(
        "models.dev has 2 new, 1 changed prices to apply.",
      ),
    ).toBeNull()
    const confirm = within(dialog).getByRole("button", {
      name: "Update catalog",
    })
    expect(confirm).toBeDisabled()
    await user.click(confirm)

    answerSecond(
      jsonResponse({ prices: null, price_error: null, catalog: EMPTY_SWEEP }),
    )
    await waitFor(() => expect(confirm).toBeEnabled())
    expect(previewsOf(requests)).toHaveLength(2)
    expect(updates(requests)).toEqual([])
  })

  it("never confirms an update whose preview failed", async () => {
    const requests = mockApi({
      previews: [
        async () => jsonResponse({ detail: "Upstream unreachable" }, 502),
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Update catalog" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(
      await within(dialog).findByText("Upstream unreachable"),
    ).toBeInTheDocument()
    const confirm = within(dialog).getByRole("button", {
      name: "Update catalog",
    })
    expect(confirm).toBeDisabled()
    await user.click(confirm)
    expect(updates(requests)).toEqual([])
  })
})

describe("HostedProvidersPage, named providers and the move", () => {
  it("discards a stored provider that cannot move, once confirmed", async () => {
    const requests = mockApi({
      migration: [
        {
          instance: "groq",
          provider: "groq",
          api_base: null,
          outcome: "conflict",
          detail: "A hosted provider already has this name.",
          models: null,
          source_removed: false,
          notes: [],
        },
        {
          instance: "openai-eu",
          provider: "openai",
          api_base: null,
          outcome: "migrate",
          detail: null,
          models: null,
          source_removed: false,
          notes: [],
        },
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    // A row that can still move is moved, not discarded.
    await screen.findByRole("button", { name: "Discard groq" })
    expect(
      screen.queryByRole("button", { name: "Discard openai-eu" }),
    ).toBeNull()

    await user.click(screen.getByRole("button", { name: "Discard groq" }))
    const dialog = await screen.findByRole("alertdialog")
    expect(dialog).toHaveTextContent("Discard groq?")
    const previewReads = () =>
      requests.filter(
        (request) =>
          request.method === "GET" &&
          request.url.endsWith("/hosted-providers/migration"),
      ).length
    const readsBefore = previewReads()
    await user.click(within(dialog).getByRole("button", { name: "Discard" }))

    await waitFor(() =>
      expect(
        requests.some(
          (request) =>
            request.method === "DELETE" &&
            request.url.endsWith("/hosted-providers/migration/groq"),
        ),
      ).toBe(true),
    )
    await waitFor(() => expect(previewReads()).toBeGreaterThan(readsBefore))
  })

  it("keeps offering the move after one that left a provider behind", async () => {
    mockApi({
      migration: [
        {
          instance: "groq",
          provider: "groq",
          api_base: null,
          outcome: "failed",
          detail: "The endpoint was refused.",
          models: null,
          source_removed: false,
          notes: [],
        },
        {
          instance: "openai-eu",
          provider: "openai",
          api_base: null,
          outcome: "migrate",
          detail: null,
          models: null,
          source_removed: false,
          notes: [],
        },
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Review the move" }),
    )
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Move 1 provider",
      }),
    )

    expect(
      await screen.findByText(/Moved 1 of 2 stored providers/),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Review the move" }),
    ).toBeInTheDocument()
  })

  it("reports a row copied but still served from its original entry apart from a completed move", async () => {
    mockApi({
      migration: [
        {
          instance: "quiet",
          provider: "openai",
          api_base: null,
          outcome: "migrate",
          detail: null,
          models: null,
          source_removed: false,
          notes: [],
        },
      ],
    })
    const fetchMock = vi.mocked(globalThis.fetch)
    const answer = fetchMock.getMockImplementation()
    fetchMock.mockImplementation(async (input, init) => {
      if (
        String(input).includes("/hosted-providers/migration") &&
        init?.method === "POST"
      ) {
        return jsonResponse({
          rows: [
            {
              instance: "quiet",
              provider: "openai",
              api_base: null,
              outcome: "migrated",
              detail: null,
              models: 0,
              source_removed: false,
              notes: [],
            },
          ],
        })
      }
      return answer ? answer(input, init) : jsonResponse({})
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Review the move" }),
    )
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Move 1 provider",
      }),
    )

    expect(
      await screen.findByText(/Moved 0 of 1 stored providers\. 1 copied/),
    ).toBeInTheDocument()
    expect(screen.getByText(/^Copied, original entry kept/)).toBeInTheDocument()
  })

  it("shows inside the dialog why a failed re-read holds the move", async () => {
    mockApi({
      migration: [
        {
          instance: "openai-eu",
          provider: "openai",
          api_base: null,
          outcome: "migrate",
          detail: null,
          models: null,
          source_removed: false,
          notes: [],
        },
      ],
    })
    const user = userEvent.setup()
    await renderPage()
    await screen.findByRole("button", { name: "Review the move" })
    const fetchMock = vi.mocked(globalThis.fetch)
    const answer = fetchMock.getMockImplementation()
    fetchMock.mockImplementation(async (input, init) => {
      if (String(input).includes("/hosted-providers/migration")) {
        return jsonResponse({ detail: "Preview went away" }, 400)
      }
      return answer ? answer(input, init) : jsonResponse({})
    })

    await user.click(screen.getByRole("button", { name: "Review the move" }))

    const dialog = await screen.findByRole("alertdialog")
    expect(
      await within(dialog).findByText("Preview went away"),
    ).toBeInTheDocument()
    expect(
      within(dialog).getByRole("button", { name: "Move 1 provider" }),
    ).toBeDisabled()
  })

  it("says the move could not be previewed rather than hiding it", async () => {
    mockApi()
    const fetchMock = vi.mocked(globalThis.fetch)
    const answer = fetchMock.getMockImplementation()
    fetchMock.mockImplementation(async (input, init) => {
      if (String(input).includes("/hosted-providers/migration")) {
        return jsonResponse({ detail: "Preview unavailable" }, 500)
      }
      return answer ? answer(input, init) : jsonResponse({})
    })
    await renderPage()

    expect(await screen.findByText("Preview unavailable")).toBeInTheDocument()
  })

  it("names a provider of its own and tests it before saving", async () => {
    const requests = mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "OpenAI" }))
    await user.type(within(dialog).getByLabelText("Name"), "openai-eu")
    await user.type(
      within(dialog).getByLabelText("API base URL"),
      "https://eu.example.com/v1",
    )
    // An endpoint of its own may go without a key.
    expect(
      within(dialog).getByRole("button", { name: "Add provider" }),
    ).toBeEnabled()

    await user.click(
      within(dialog).getByRole("button", { name: "Test connection" }),
    )
    expect(
      await within(dialog).findByText(/Connected\. 3 models available\./),
    ).toBeInTheDocument()
    const [test] = hostedRequests(requests, "POST").filter((request) =>
      request.url.endsWith("/hosted-providers/test"),
    )
    expect(test.body).toEqual({
      provider: "openai",
      name: "openai-eu",
      api_base: "https://eu.example.com/v1",
    })
    expect(hostedRequests(requests, "POST")).toHaveLength(1)

    await user.click(
      within(dialog).getByRole("button", { name: "Add provider" }),
    )
    await waitFor(() =>
      expect(
        hostedRequests(requests, "POST").filter((request) =>
          request.url.endsWith("/hosted-providers"),
        ),
      ).toHaveLength(1),
    )
  })

  it("says the client options are encrypted at rest, as the hosted store keeps them", async () => {
    mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")

    expect(
      within(dialog).getByText(/Encrypted at rest, and an option/),
    ).toBeInTheDocument()
    expect(within(dialog).queryByText(/nothing here is encrypted/)).toBeNull()
  })

  it("holds the save until a key is typed for a provider that needs one", async () => {
    mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "OpenAI" }))

    await waitFor(() =>
      expect(
        within(dialog).getByRole("button", { name: "Add provider" }),
      ).toBeDisabled(),
    )
  })

  it("shows a provider's name beside its implementation, and a key no secret reads", async () => {
    mockApi({
      providers: [
        hostedProvider({
          name: "openai-eu",
          decryptable: false,
        }),
      ],
    })
    await renderPage()

    const table = await screen.findByRole("grid", { name: "Hosted providers" })
    expect(await within(table).findByText("openai-eu")).toBeInTheDocument()
    expect(within(table).getByText("openai")).toBeInTheDocument()
    expect(within(table).getByText("Key unreadable")).toBeInTheDocument()
  })

  it("lists the configured providers read-only in the same table", async () => {
    mockApi({
      providers: [hostedProvider({ name: "openai" })],
      configured: [providerInfo("anthropic")],
    })
    await renderPage()

    const table = await screen.findByRole("grid", { name: "Hosted providers" })
    const header = await within(table).findByRole("rowheader", {
      name: /^anthropic/,
    })
    const row = header.parentElement as HTMLElement
    expect(within(row).getByText("config.yml")).toBeInTheDocument()
    expect(
      within(row).getByText("Every model a request names"),
    ).toBeInTheDocument()
    expect(within(row).queryByRole("link")).toBeNull()
    expect(within(row).queryByRole("switch")).toBeNull()
    expect(
      within(table).getByRole("link", { name: "openai" }),
    ).toBeInTheDocument()
  })

  it("moves the stored providers once the preview is reviewed", async () => {
    const requests = mockApi({
      migration: [
        {
          instance: "openai-eu",
          provider: "openai",
          api_base: "https://eu.example.com/v1",
          outcome: "migrate",
          detail: null,
          models: null,
          source_removed: false,
          notes: [],
        },
        {
          instance: "groq",
          provider: "groq",
          api_base: null,
          outcome: "conflict",
          detail: "A hosted provider named 'groq' already exists.",
          models: null,
          source_removed: false,
          notes: [],
        },
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Review the move" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(within(dialog).getByText(/already exists/)).toBeInTheDocument()
    await user.click(
      within(dialog).getByRole("button", { name: "Move 1 provider" }),
    )

    expect(
      await screen.findByText(/Moved 1 of 2 stored providers/),
    ).toBeInTheDocument()
    const moves = hostedRequests(requests, "POST").filter((request) =>
      request.url.endsWith("/hosted-providers/migration"),
    )
    // Only what the preview offered to move, so nothing unreviewed moves.
    expect(moves.map((request) => request.body)).toEqual([
      { rows: [{ instance: "openai-eu", updated_at: null }] },
    ])
  })

  describe("connecting otari.ai", () => {
    const connectRequests = (requests: Request[]) =>
      requests.filter((request) => request.url.includes("/platform/connect"))

    it("offers it beside Add provider once providers exist, opening the card in place", async () => {
      mockApi()
      const user = userEvent.setup()
      await renderPage({ platform_connect: true })

      await user.click(
        await screen.findByRole("button", { name: "Connect otari.ai" }),
      )

      const card = await screen.findByRole("region", {
        name: "Connect otari.ai",
      })
      expect(
        within(card).getByText(/paid service, and requests sent to it leave/),
      ).toBeInTheDocument()
    })

    it("offers it on the first-provider panel", async () => {
      mockApi({ providers: [] })
      await renderPage({ platform_connect: true })

      expect(
        await screen.findByRole("button", { name: "Add your first provider" }),
      ).toBeInTheDocument()
      expect(
        await screen.findByRole("button", { name: "Connect otari.ai" }),
      ).toBeInTheDocument()
    })

    it("keeps it in the header of the empty page once the welcome was skipped for own keys", async () => {
      localStorage.setItem(WELCOME_ANSWERED_KEY, OWN_KEYS)
      mockApi({ providers: [] })
      const user = userEvent.setup()
      await renderPage({ platform_connect: true })

      expect(
        await screen.findByRole("button", { name: "Add your first provider" }),
      ).toBeInTheDocument()
      expect(
        screen.queryByRole("heading", { name: "Bring your own provider" }),
      ).not.toBeInTheDocument()
      const connects = screen.getAllByRole("button", {
        name: "Connect otari.ai",
      })
      expect(connects).toHaveLength(1)

      await user.click(connects[0] as HTMLElement)
      expect(
        await screen.findByRole("region", { name: "Connect otari.ai" }),
      ).toBeInTheDocument()
    })

    it("opens the Overview once a connection started from that header is approved", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true })
      try {
        localStorage.setItem(WELCOME_ANSWERED_KEY, OWN_KEYS)
        mockApi({
          providers: [],
          connectStatuses: [
            {
              status: "pending",
              user_code: "WXYZ-1234",
              verification_uri_complete: "https://otari.test/connect",
              expires_in: 600,
            },
            { status: "connected" },
          ],
        })
        vi.spyOn(window, "open").mockReturnValue(null)
        const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
        await renderHosted(
          <HostedProvidersPage />,
          "/admin/hosted-providers",
          { platform_connect: true },
          [{ path: "/", element: <p>OVERVIEW</p> }],
        )

        await user.click(
          await screen.findByRole("button", { name: "Connect otari.ai" }),
        )
        const card = await screen.findByRole("region", {
          name: "Connect otari.ai",
        })
        await user.click(
          within(card).getByRole("button", { name: /Connect otari\.ai/ }),
        )
        await vi.advanceTimersByTimeAsync(3_000)
        await vi.advanceTimersByTimeAsync(3_000)

        expect(await screen.findByText("OVERVIEW")).toBeInTheDocument()
      } finally {
        vi.useRealTimers()
      }
    })

    it("follows a Skip pressed in the welcome over the open empty page", async () => {
      mockApi({ providers: [] })
      // The welcome greets only the deployment's operator.
      const served = vi.mocked(globalThis.fetch).getMockImplementation()
      vi.mocked(globalThis.fetch).mockImplementation(async (input, init) =>
        String(input).endsWith("/organizations/me")
          ? jsonResponse(organizationContext())
          : ((await served?.(input, init)) ?? jsonResponse({})),
      )
      const user = userEvent.setup()
      await renderHosted(
        <>
          <HostedProvidersPage />
          <WelcomeDialog />
        </>,
        "/admin/hosted-providers",
        { platform_connect: true },
      )

      await user.click(
        await screen.findByRole("button", {
          name: "Skip, I’ll use my own keys",
        }),
      )

      await waitFor(() =>
        expect(
          screen.queryByRole("heading", { name: "Bring your own provider" }),
        ).not.toBeInTheDocument(),
      )
      expect(
        screen.getAllByRole("button", { name: "Connect otari.ai" }),
      ).toHaveLength(1)
    })

    it("keeps it out of the header of the empty page while the panel offers it", async () => {
      mockApi({ providers: [] })
      await renderPage({ platform_connect: true })

      expect(
        await screen.findByRole("heading", { name: "Bring your own provider" }),
      ).toBeInTheDocument()
      expect(
        await screen.findAllByRole("button", { name: "Connect otari.ai" }),
      ).toHaveLength(1)
    })

    it("neither offers it nor reads its status where the deployment does not", async () => {
      const requests = mockApi()
      await renderPage({ platform_connect: false })

      expect(
        await screen.findByRole("button", { name: "Add provider" }),
      ).toBeInTheDocument()
      expect(
        screen.queryByRole("button", { name: "Connect otari.ai" }),
      ).not.toBeInTheDocument()
      expect(connectRequests(requests)).toEqual([])
    })
  })
})
