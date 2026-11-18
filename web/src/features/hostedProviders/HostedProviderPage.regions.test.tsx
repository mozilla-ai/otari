import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { HostedProviderPage } from "@/features/hostedProviders/HostedProviderPage"
import { API_ROOT } from "@/shared/api/client"
import {
  hostedModel,
  hostedProvider,
  hostedRegion,
  hostedRequests,
  mockApi,
  renderHosted,
} from "@/tests/hostedProviders"
import { pressRowAction } from "@/tests/rowMenu"

function renderProviderPage(name = "openai") {
  return renderHosted(
    <HostedProviderPage name={name} />,
    `/admin/hosted-providers/${name}`,
  )
}

const twoRegions = () =>
  hostedProvider({
    data_location: "CN",
    regions: [
      hostedRegion({ slug: "cn", label: "China", position: 0 }),
      hostedRegion({
        slug: "sg",
        label: "Southeast Asia",
        position: 1,
        data_location: "SG",
        has_api_key: true,
        api_key_last4: "1234",
        api_base: "https://sg.example.com/v1",
      }),
    ],
  })

const regionsTable = () =>
  screen.findByRole("grid", { name: "Regions of openai" })

afterEach(() => {
  vi.restoreAllMocks()
})

describe("HostedProviderPage: regions", () => {
  it("turns the current connection into the first region when adding one", async () => {
    const requests = mockApi({
      providers: [hostedProvider({ data_location: "CN" })],
    })
    const user = userEvent.setup()
    await renderProviderPage()

    expect(await screen.findByText("China")).toBeInTheDocument()
    await user.click(
      screen.getByRole("button", { name: "Add a region to openai" }),
    )
    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).getByRole("textbox", {
        name: "Current connection's name",
      }),
    ).toHaveValue("China")
    expect(
      within(dialog).getByRole("textbox", { name: "Current connection's id" }),
    ).toHaveValue("cn")
    await user.type(
      within(dialog).getByRole("textbox", { name: "Name" }),
      "Southeast Asia",
    )
    await user.type(
      within(dialog).getByRole("textbox", { name: "Region id" }),
      "sg",
    )
    expect(within(dialog).getByText(/openai@sg:<model>/)).toBeInTheDocument()
    await user.click(within(dialog).getByRole("button", { name: "Add region" }))

    await waitFor(() =>
      expect(hostedRequests(requests, "POST")).toHaveLength(1),
    )
    const [post] = hostedRequests(requests, "POST")
    expect(post.url).toBe(`${API_ROOT}/hosted-providers/openai/regions`)
    // The provider's key is used, so none is sent.
    expect(post.body).toEqual({
      slug: "sg",
      label: "Southeast Asia",
      primary_slug: "cn",
      primary_label: "China",
    })
  })

  it("adds a region with a key and location of its own", async () => {
    const requests = mockApi({ providers: [twoRegions()] })
    const user = userEvent.setup()
    await renderProviderPage()

    await user.click(
      await screen.findByRole("button", { name: "Add a region to openai" }),
    )
    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).queryByRole("textbox", {
        name: "Current connection's name",
      }),
    ).toBeNull()
    await user.type(
      within(dialog).getByRole("textbox", { name: "Name" }),
      "West US",
    )
    const slug = within(dialog).getByRole("textbox", { name: "Region id" })
    await user.type(slug, "sg")
    expect(
      within(dialog).getByText("Another region already has this id."),
    ).toBeInTheDocument()
    await user.clear(slug)
    await user.type(slug, "us-west")
    await user.type(
      within(dialog).getByRole("textbox", { name: "Endpoint URL" }),
      "https://us.example.com/v1",
    )
    await user.type(
      within(dialog).getByRole("combobox", { name: "Data hosted in" }),
      "United States",
    )
    await user.click(
      await screen.findByRole("option", { name: "United States (US)" }),
    )
    // The box opens on focus, so focus leaves it rather than Escape closing it.
    await user.click(within(dialog).getByLabelText(/^Name/))
    await user.click(within(dialog).getByRole("radio", { name: "Its own key" }))
    const submit = within(dialog).getByRole("button", { name: "Add region" })
    expect(submit).toBeDisabled()
    await user.type(
      within(dialog).getByLabelText(/Region's API key/),
      "sk-west",
    )
    await user.click(submit)

    await waitFor(() =>
      expect(hostedRequests(requests, "POST")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "POST")[0].body).toEqual({
      slug: "us-west",
      label: "West US",
      api_key: "sk-west",
      api_base: "https://us.example.com/v1",
      data_location: "US",
    })
  })

  it("lists each region with its location, key and endpoint", async () => {
    mockApi({ providers: [twoRegions()] })
    await renderProviderPage()

    const table = await regionsTable()
    const [, china, asia] = within(table).getAllByRole("row")
    expect(within(china).getByText("primary")).toBeInTheDocument()
    expect(within(china).getByText("Provider's key")).toBeInTheDocument()
    expect(within(china).getByText("Provider's endpoint")).toBeInTheDocument()
    expect(within(china).getByText("openai@cn")).toBeInTheDocument()
    expect(within(asia).getByText("Singapore")).toBeInTheDocument()
    expect(within(asia).getByText("Own ••1234")).toBeInTheDocument()
    expect(
      within(asia).getByText("https://sg.example.com/v1"),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Requests go to the first region the workspace allows/),
    ).toHaveTextContent("openai@sg:model pins one.")
  })

  it("puts a region back on the provider's key from its editor", async () => {
    const requests = mockApi({ providers: [twoRegions()] })
    const user = userEvent.setup()
    await renderProviderPage()

    const table = await regionsTable()
    await pressRowAction(
      user,
      "Edit…",
      within(table).getAllByRole("row")[2] as HTMLElement,
    )
    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).getByRole("radio", { name: "Its own key" }),
    ).toBeChecked()
    await user.click(
      within(dialog).getByRole("radio", { name: "Use the provider's key" }),
    )
    await user.clear(
      within(dialog).getByRole("textbox", { name: "Endpoint URL" }),
    )
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    const [patch] = hostedRequests(requests, "PATCH")
    expect(patch.url).toBe(`${API_ROOT}/hosted-providers/openai/regions/sg`)
    expect(patch.body).toEqual({ api_key: null, api_base: null })
  })

  it("moves a region down, switches it off and removes it", async () => {
    const requests = mockApi({ providers: [twoRegions()] })
    const user = userEvent.setup()
    await renderProviderPage()

    const table = await regionsTable()
    const china = within(table).getAllByRole("row")[1] as HTMLElement
    await pressRowAction(user, "Move down", china)
    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "PATCH")[0]).toMatchObject({
      url: `${API_ROOT}/hosted-providers/openai/regions/cn`,
      body: { position: 1 },
    })

    await user.click(
      within(table).getByRole("switch", {
        name: "Serve traffic from Southeast Asia",
      }),
    )
    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(2),
    )
    expect(hostedRequests(requests, "PATCH")[1]).toMatchObject({
      url: `${API_ROOT}/hosted-providers/openai/regions/sg`,
      body: { enabled: false },
    })

    await pressRowAction(
      user,
      "Remove…",
      within(table).getAllByRole("row")[2] as HTMLElement,
    )
    const confirm = await screen.findByRole("alertdialog")
    await user.click(
      within(confirm).getByRole("button", { name: "Remove region" }),
    )
    await waitFor(() =>
      expect(hostedRequests(requests, "DELETE")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "DELETE")[0].url).toBe(
      `${API_ROOT}/hosted-providers/openai/regions/sg`,
    )
  })

  it("offers no region editing on a provider config.yml defines", async () => {
    mockApi({ providers: [{ ...twoRegions(), source: "config" }] })
    await renderProviderPage()

    const table = await regionsTable()
    expect(
      screen.queryByRole("button", { name: "Add a region to openai" }),
    ).toBeNull()
    expect(
      within(table).queryAllByRole("button", { name: /^Actions for / }),
    ).toHaveLength(0)
    expect(
      within(table).getByRole("switch", { name: "Serve traffic from China" }),
    ).toBeDisabled()
  })

  it("sends the provider's data location from its edit dialog", async () => {
    const requests = mockApi({ providers: [hostedProvider()] })
    const user = userEvent.setup()
    await renderProviderPage()

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    const dialog = await screen.findByRole("dialog")
    await user.type(
      within(dialog).getByRole("combobox", { name: "Data hosted in" }),
      "Germ",
    )
    await user.click(
      await screen.findByRole("option", { name: "Germany (DE)" }),
    )
    await user.click(within(dialog).getByLabelText(/^API base URL/))
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "PATCH")[0].body).toEqual({
      api_base: "",
      data_location: "DE",
    })
  })
})

describe("HostedProviderPage: region prices", () => {
  const pricedModels = () => [
    hostedModel({
      region_rates: {
        sg: {
          input_price_per_million: 3,
          output_price_per_million: 12,
          unit: "token",
          source: "deployment",
        },
      },
    }),
  ]
  const pricingRequests = (
    requests: ReturnType<typeof mockApi>,
    method: string,
  ) =>
    requests.filter(
      (request) =>
        request.url.startsWith(`${API_ROOT}/pricing`) &&
        request.method === method,
    )

  it("shows a column per region, its own price or the provider's", async () => {
    mockApi({ providers: [twoRegions()], models: pricedModels() })
    await renderProviderPage()

    expect(
      await screen.findByRole("button", {
        name: "Edit gpt-4o price in China: $2.50 in / $10.00 out, the provider's price",
      }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", {
        name: "Edit gpt-4o price in Southeast Asia: $3.00 in / $12.00 out",
      }),
    ).toBeInTheDocument()
  })

  it("adds no region column to a provider serving from one connection", async () => {
    mockApi({ providers: [hostedProvider()], models: pricedModels() })
    await renderProviderPage()

    await screen.findByText("gpt-4o")
    expect(
      screen.queryByRole("button", { name: /^Edit gpt-4o price in / }),
    ).toBeNull()
  })

  it("prices a model in one region under that region's key", async () => {
    const requests = mockApi({
      providers: [twoRegions()],
      models: pricedModels(),
    })
    const user = userEvent.setup()
    await renderProviderPage()

    await user.click(
      await screen.findByRole("button", {
        name: /^Edit gpt-4o price in China/,
      }),
    )
    const dialog = await screen.findByRole("dialog", {
      name: "Edit price in China",
    })
    // It charges the provider's price now, so there is nothing to clear.
    expect(
      within(dialog).queryByRole("button", { name: "Use provider price" }),
    ).toBeNull()
    const input = within(dialog).getByRole("textbox", {
      name: /Input, per 1M tokens/,
    })
    expect(input).toHaveValue("2.5")
    await user.clear(input)
    await user.type(input, "4")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(pricingRequests(requests, "POST")).toHaveLength(1),
    )
    expect(pricingRequests(requests, "POST")[0].body).toMatchObject({
      model_key: "openai@cn:gpt-4o",
      input_price_per_million: 4,
      output_price_per_million: 10,
      unit: "token",
    })
  })

  it("puts a region back on the provider's price", async () => {
    const requests = mockApi({
      providers: [twoRegions()],
      models: pricedModels(),
    })
    const user = userEvent.setup()
    await renderProviderPage()

    await user.click(
      await screen.findByRole("button", {
        name: /^Edit gpt-4o price in Southeast Asia/,
      }),
    )
    const dialog = await screen.findByRole("dialog", {
      name: "Edit price in Southeast Asia",
    })
    expect(
      within(dialog).getByRole("textbox", { name: /Input, per 1M tokens/ }),
    ).toHaveValue("3")
    await user.click(
      within(dialog).getByRole("button", { name: "Use provider price" }),
    )

    await waitFor(() =>
      expect(pricingRequests(requests, "DELETE")).toHaveLength(1),
    )
    expect(pricingRequests(requests, "DELETE")[0].url).toBe(
      `${API_ROOT}/pricing/${encodeURIComponent("openai@sg:gpt-4o")}?kind=model`,
    )
  })
})
