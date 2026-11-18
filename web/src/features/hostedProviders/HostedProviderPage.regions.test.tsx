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
      hostedRegion({
        slug: "cn",
        label: "China",
        position: 0,
        data_location: "CN",
      }),
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
    // Named in one line, with nothing to fill in unless it is renamed.
    expect(
      within(dialog).getByText(/Your current connection becomes/),
    ).toHaveTextContent("Your current connection becomes China (openai@cn)")
    expect(
      within(dialog).queryByRole("textbox", {
        name: "Current connection's name",
      }),
    ).toBeNull()
    await user.type(
      within(dialog).getByRole("textbox", { name: "Name" }),
      "Southeast Asia",
    )
    // The id follows the name until it is changed.
    expect(
      within(dialog).getByText("openai@southeast-asia"),
    ).toBeInTheDocument()
    await user.click(
      within(dialog).getByRole("button", { name: "Change region id" }),
    )
    const slug = within(dialog).getByRole("textbox", { name: "Region id" })
    expect(slug).toHaveValue("southeast-asia")
    await user.clear(slug)
    await user.type(slug, "sg")
    expect(within(dialog).getByText(/openai@sg:<model>/)).toBeInTheDocument()
    // A new region starts from the provider's location, as an editable copy.
    expect(
      within(dialog).getByRole("combobox", { name: "Data hosted in" }),
    ).toHaveValue("China")
    await user.click(within(dialog).getByRole("button", { name: "Add region" }))

    await waitFor(() =>
      expect(hostedRequests(requests, "POST")).toHaveLength(1),
    )
    const [post] = hostedRequests(requests, "POST")
    expect(post.url).toBe(`${API_ROOT}/hosted-providers/openai/regions`)
    // No key, so the shared one is used. The current connection's id is the
    // server's own default; its name is the location's name, not its code.
    expect(post.body).toEqual({
      slug: "sg",
      label: "Southeast Asia",
      data_location: "CN",
      primary_label: "China",
    })
  })

  it("renames the current connection only when asked", async () => {
    const requests = mockApi({ providers: [hostedProvider()] })
    const user = userEvent.setup()
    await renderProviderPage()

    await user.click(
      await screen.findByRole("button", { name: "Add a region to openai" }),
    )
    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).getByText(/Your current connection becomes/),
    ).toHaveTextContent("Primary (openai@primary)")
    await user.click(within(dialog).getByRole("button", { name: "Rename" }))
    const id = within(dialog).getByRole("textbox", {
      name: "Current connection's id",
    })
    await user.clear(id)
    await user.type(id, "us")
    const name = within(dialog).getByRole("textbox", {
      name: "Current connection's name",
    })
    await user.clear(name)
    await user.type(name, "United States")
    await user.type(
      within(dialog).getByRole("textbox", { name: "Name" }),
      "Europe",
    )
    await user.click(within(dialog).getByRole("button", { name: "Add region" }))

    await waitFor(() =>
      expect(hostedRequests(requests, "POST")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "POST")[0].body).toEqual({
      slug: "europe",
      label: "Europe",
      data_location: null,
      primary_slug: "us",
      primary_label: "United States",
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
    const name = within(dialog).getByRole("textbox", { name: "Name" })
    // A name whose id is taken opens the id field by itself.
    await user.type(name, "SG")
    expect(
      within(dialog).getByText("Another region already has this id."),
    ).toBeInTheDocument()
    const slug = within(dialog).getByRole("textbox", { name: "Region id" })
    await user.clear(slug)
    await user.type(slug, "us-west")
    await user.clear(name)
    await user.type(name, "West US")
    await user.type(
      within(dialog).getByRole("textbox", { name: "Endpoint URL" }),
      "https://us.example.com/v1",
    )
    await user.type(
      within(dialog).getByRole("combobox", { name: "Data hosted in" }),
      "States",
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
    // Numbered in the order requests try them.
    expect(within(china).getByText("1")).toBeInTheDocument()
    expect(within(asia).getByText("2")).toBeInTheDocument()
    expect(within(china).queryByText("primary")).toBeNull()
    expect(within(china).getAllByText("China")).toHaveLength(2)
    expect(within(china).getByText("Shared key")).toBeInTheDocument()
    expect(within(china).getByText("Shared endpoint")).toBeInTheDocument()
    expect(within(china).getByText("Every country")).toBeInTheDocument()
    expect(within(china).getByText("openai@cn")).toBeInTheDocument()
    expect(within(asia).getByText("Singapore")).toBeInTheDocument()
    expect(within(asia).getByText("Own ••1234")).toBeInTheDocument()
    expect(
      within(asia).getByText("https://sg.example.com/v1"),
    ).toBeInTheDocument()
    expect(screen.getByText(/^Tried in this order/)).toHaveTextContent(
      "openai@sg:model pins one.",
    )
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
      within(dialog).getByRole("radio", { name: "Use the shared key" }),
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
    // The answer is the provider in its new order, shown before any reread.
    const regionRow = (label: string) =>
      within(table)
        .getAllByRole("row")
        .find((row) => within(row).queryByText(label) !== null) as HTMLElement
    await waitFor(() =>
      expect(
        within(within(table).getAllByRole("row")[1]).getByText(
          "Southeast Asia",
        ),
      ).toBeInTheDocument(),
    )

    // Switching off says where the region's requests go next, then asks.
    await user.click(
      within(table).getByRole("switch", {
        name: "Serve traffic from Southeast Asia",
      }),
    )
    const confirmOff = await screen.findByRole("alertdialog")
    expect(
      within(confirmOff).getByText(/first other serving region that admits/),
    ).toHaveTextContent(
      "China, the next serving region in order, is the likely one. A request that pins openai@sg is refused while it is off.",
    )
    expect(hostedRequests(requests, "PATCH")).toHaveLength(1)
    await user.click(
      within(confirmOff).getByRole("button", { name: "Switch off" }),
    )
    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(2),
    )
    expect(hostedRequests(requests, "PATCH")[1]).toMatchObject({
      url: `${API_ROOT}/hosted-providers/openai/regions/sg`,
      body: { enabled: false },
    })

    await pressRowAction(user, "Remove…", regionRow("Southeast Asia"))
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

  it("keeps only provider-wide settings once regions exist, and what they share", async () => {
    mockApi({
      providers: [
        {
          ...twoRegions(),
          api_base: "https://cn.example.com/v1",
          client_args: { region_name: "ap-east-1" },
        },
      ],
    })
    await renderProviderPage()

    const grid = await screen.findByLabelText("openai settings")
    expect(within(grid).getByText("Serves")).toBeInTheDocument()
    expect(within(grid).getByText("Zero data retention")).toBeInTheDocument()
    expect(within(grid).queryByText("Endpoint")).toBeNull()
    expect(within(grid).queryByText("Data hosted in")).toBeNull()
    expect(within(grid).queryByText("Requests from")).toBeNull()
    const shared = screen.getByLabelText("Shared by openai's regions")
    expect(within(shared).getByText("••••abcd")).toBeInTheDocument()
    expect(
      within(shared).getByText("https://cn.example.com/v1"),
    ).toBeInTheDocument()
    expect(within(shared).getByText("region_name")).toBeInTheDocument()
  })

  it("shows a provider without regions no region concepts", async () => {
    mockApi({ providers: [hostedProvider({ data_location: "DE" })] })
    await renderProviderPage()

    const grid = await screen.findByLabelText("openai settings")
    expect(within(grid).getByText("Endpoint")).toBeInTheDocument()
    expect(within(grid).getByText("Data hosted in")).toBeInTheDocument()
    expect(within(grid).getByText("Germany")).toBeInTheDocument()
    expect(screen.queryByText("Shared by regions")).toBeNull()
    expect(screen.queryByRole("grid", { name: "Regions of openai" })).toBeNull()
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

  it("shows a region's column once it prices a model, and every region to compare", async () => {
    mockApi({ providers: [twoRegions()], models: pricedModels() })
    const user = userEvent.setup()
    await renderProviderPage()

    expect(
      await screen.findByRole("button", {
        name: "Edit gpt-4o price in Southeast Asia: $3.00 in / $12.00 out",
      }),
    ).toBeInTheDocument()
    expect(screen.getByText("Default in")).toBeInTheDocument()
    // China sets no price of its own, so it has no column until compared.
    expect(
      screen.queryByRole("button", { name: /^Edit gpt-4o price in China/ }),
    ).toBeNull()
    await user.click(
      screen.getByRole("switch", {
        name: "Compare regions: a price column for every region",
      }),
    )
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
    expect(screen.queryByText("Default in")).toBeNull()
    expect(
      screen.queryByRole("switch", { name: /^Compare regions/ }),
    ).toBeNull()
    expect(
      screen.queryByRole("button", { name: /^Edit gpt-4o price in / }),
    ).toBeNull()
  })

  it("keeps a lone region's price column, so its own price stays editable", async () => {
    const [, asia] = twoRegions().regions ?? []
    mockApi({
      providers: [hostedProvider({ regions: [{ ...asia, position: 0 }] })],
      models: pricedModels(),
    })
    await renderProviderPage()

    expect(
      await screen.findByRole("button", {
        name: "Edit gpt-4o price in Southeast Asia: $3.00 in / $12.00 out",
      }),
    ).toBeInTheDocument()
  })

  it("prices a model in one region under that region's key", async () => {
    const requests = mockApi({
      providers: [twoRegions()],
      models: pricedModels(),
    })
    const user = userEvent.setup()
    await renderProviderPage()

    await user.click(
      await screen.findByRole("switch", {
        name: "Compare regions: a price column for every region",
      }),
    )
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
      missingPrices: ["openai@sg/gpt-4o"],
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

    // Both spellings a region override may be stored under, and never the
    // provider's own key. The older one is already gone here, a 404.
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    expect(pricingRequests(requests, "DELETE").map((r) => r.url)).toEqual([
      `${API_ROOT}/pricing/${encodeURIComponent("openai@sg:gpt-4o")}?kind=model`,
      `${API_ROOT}/pricing/${encodeURIComponent("openai@sg/gpt-4o")}?kind=model`,
    ])
  })
})
