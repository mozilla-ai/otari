import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ProvidersPage } from "@/features/providers/ProvidersPage"
import { API_ROOT } from "@/shared/api/client"
import { mockApi, renderPage, storedProvider } from "@/tests/providersPage"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ProvidersPage edit form", () => {
  it("splits a stored provider's registered options out of the JSON box on edit", async () => {
    mockApi({
      stored: [
        storedProvider("bedrock", "0000", true, {
          region_name: "us-east-1",
          timeout: 1800,
        }),
      ],
      catalog: [
        {
          id: "bedrock",
          name: "Bedrock",
          env_key: "AWS_BEARER_TOKEN_BEDROCK",
          default_api_base: null,
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(await screen.findByRole("button", { name: "Edit" }))

    expect(
      await screen.findByRole("textbox", { name: /AWS region/ }),
    ).toHaveValue("us-east-1")
    expect(
      screen.getByRole("textbox", { name: "Client options (JSON)" }),
    ).toHaveValue('{\n  "timeout": 1800\n}')
  })

  it("keeps a split-out option when the provider type is retyped mid-edit", async () => {
    // The field list follows the provider type, which is an editable box here,
    // while the values were split out of client_args at mount. A field that
    // stops rendering must not take the stored option with it.
    const fetchMock = mockApi({
      stored: [
        storedProvider("bedrock", "0000", true, {
          region_name: "us-east-1",
          aws_secret_access_key: "***",
        }),
      ],
      catalog: [
        {
          id: "bedrock",
          name: "Bedrock",
          env_key: "AWS_BEARER_TOKEN_BEDROCK",
          default_api_base: null,
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(await screen.findByRole("button", { name: "Edit" }))
    await screen.findByRole("textbox", { name: /AWS region/ })
    await user.type(
      screen.getByRole("textbox", { name: "Provider type" }),
      "openai",
    )
    expect(
      screen.queryByRole("textbox", { name: /AWS region/ }),
    ).not.toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Save" }))

    const patch = await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([, init]) => (init?.method ?? "") === "PATCH",
      )
      expect(call).toBeDefined()
      return call!
    })
    expect(JSON.parse(String(patch[1]?.body))).toMatchObject({
      client_args: {
        region_name: "us-east-1",
        aws_secret_access_key: "***",
      },
    })
  })

  it("sends client options entered on the custom-endpoint form", async () => {
    // otari#517: client_args is the only way to give the provider client a
    // timeout, which a slow self-hosted backend needs; it was reachable only
    // through config.yml or the raw API before.
    const fetchMock = mockApi({ meta: [], stored: [] })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add your first provider" }),
    )
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.type(screen.getByLabelText("Name"), "homelab")
    await user.type(
      screen.getByLabelText("API base"),
      "https://my-box.example.net",
    )
    await user.type(
      screen.getByLabelText("Client options (JSON)"),
      '{{"timeout": 1800}',
    )
    await user.click(screen.getByRole("button", { name: "Add provider" }))

    const post = await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([u, init]) =>
          String(u).endsWith(`${API_ROOT}/provider-credentials`) &&
          (init?.method ?? "") === "POST",
      )
      expect(call).toBeDefined()
      return call!
    })
    expect(JSON.parse(String(post[1]?.body))).toMatchObject({
      instance: "homelab",
      client_args: { timeout: 1800 },
    })
  })

  it("rejects client options that are not a JSON object instead of sending them", async () => {
    const fetchMock = mockApi({ meta: [], stored: [] })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add your first provider" }),
    )
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.type(screen.getByLabelText("Name"), "homelab")
    await user.type(
      screen.getByLabelText("API base"),
      "https://my-box.example.net",
    )

    const clientArgs = screen.getByLabelText("Client options (JSON)")
    await user.type(clientArgs, "timeout: 1800")
    expect(await screen.findByText("Not valid JSON.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add provider" })).toBeDisabled()
    // A "Test connection" would hit the provider with the same bad options.
    expect(
      screen.getByRole("button", { name: "Test connection" }),
    ).toBeDisabled()

    // Valid JSON, but not an object: the API takes a mapping of client kwargs.
    // "[[" is userEvent's escape for a literal "[".
    await user.clear(clientArgs)
    await user.type(clientArgs, "[[1800]")
    expect(
      await screen.findByText('Must be a JSON object, like {"timeout": 1800}.'),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add provider" })).toBeDisabled()

    expect(
      fetchMock.mock.calls.some(
        ([u, init]) =>
          String(u).endsWith(`${API_ROOT}/provider-credentials`) &&
          (init?.method ?? "") === "POST",
      ),
    ).toBe(false)
  })

  it("holds Advanced open while invalid client options are blocking the submit", async () => {
    // Otherwise collapsing the section leaves "Add provider" disabled with the
    // reason, and the field to fix it, off screen.
    mockApi({
      stored: [],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: true,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add your first provider" }),
    )
    await user.type(screen.getByPlaceholderText("Search providers…"), "OpenAI")
    await user.click(await screen.findByRole("option", { name: /OpenAI/ }))
    // Close the combobox popover, which otherwise aria-hides the rest of the form.
    await user.keyboard("{Escape}")
    await user.click(
      screen.getByRole("button", {
        name: "Advanced (API base, rename, client options)",
      }),
    )
    await user.type(screen.getByLabelText("Client options (JSON)"), "oops")
    expect(await screen.findByText("Not valid JSON.")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Hide advanced" }))
    expect(screen.getByLabelText("Client options (JSON)")).toBeInTheDocument()
    expect(screen.getByText("Not valid JSON.")).toBeInTheDocument()

    // The hide the operator asked for takes effect once the section is no longer
    // the thing blocking the submit.
    await user.clear(screen.getByLabelText("Client options (JSON)"))
    await waitFor(() =>
      expect(
        screen.queryByLabelText("Client options (JSON)"),
      ).not.toBeInTheDocument(),
    )
  })

  it("opens the edit form in a dialog, naming the instance", async () => {
    mockApi({ stored: [storedProvider("homelab", "1234", true)] })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Edit" }))

    // A dialog rather than a band above the table: the row keeps its place, and
    // the page under it does not shift by the height of a form (otari-ai#2125).
    const dialog = await screen.findByRole("dialog", { name: "Edit provider" })
    expect(within(dialog).getByText("homelab")).toBeInTheDocument()
  })

  it("prefills stored client options on edit and clears them when emptied", async () => {
    const fetchMock = mockApi({
      stored: [storedProvider("homelab", "1234", true, { timeout: 1800 })],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Edit" }))
    const clientArgs = screen.getByLabelText("Client options (JSON)")
    expect(clientArgs).toHaveValue(JSON.stringify({ timeout: 1800 }, null, 2))

    // Emptying the field clears the stored options: an explicit null, not an
    // omission, which the API would read as "leave them alone".
    await user.clear(clientArgs)
    await user.click(screen.getByRole("button", { name: "Save" }))

    const patch = await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([, init]) => (init?.method ?? "") === "PATCH",
      )
      expect(call).toBeDefined()
      return call!
    })
    expect(JSON.parse(String(patch[1]?.body)).client_args).toBeNull()
  })

  it("keeps a save from going out while the edited client options are invalid", async () => {
    const fetchMock = mockApi({
      stored: [storedProvider("homelab", "1234", true, { timeout: 1800 })],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.type(screen.getByLabelText("Client options (JSON)"), "oops")

    expect(await screen.findByText("Not valid JSON.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    expect(
      fetchMock.mock.calls.some(([, init]) => (init?.method ?? "") === "PATCH"),
    ).toBe(false)
  })
})
