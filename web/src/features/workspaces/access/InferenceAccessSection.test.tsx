import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type {
  OrgProviderKey,
  WorkspaceHostedProvider,
  WorkspaceProviderKeyOverride,
} from "@/client"
import { InferenceAccessSection } from "@/features/workspaces/access/InferenceAccessSection"
import { API_ROOT } from "@/shared/api/client"
import {
  orgProviderKey,
  workspaceHostedProvider,
  workspaceProviderKeyOverride,
} from "@/tests/fixtures"

const WORKSPACE = "44444444-4444-4444-4444-444444444444"
const OPENAI_KEY = "66666666-6666-6666-6666-666666666666"
const ANTHROPIC_KEY = "77777777-7777-7777-7777-777777777777"
const SECOND_OPENAI_KEY = "88888888-8888-8888-8888-888888888888"

interface Request {
  url: string
  method: string
  body: unknown
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi(
  opts: {
    keys?: OrgProviderKey[]
    overrides?: WorkspaceProviderKeyOverride[]
    hosted?: WorkspaceHostedProvider[]
    /** Catalog ids, as `/models` spells them: `provider:model`. */
    catalog?: string[]
    /** A refusal for whichever write matches, for the error paths. */
    writeRefusal?: { method: string; status: number; detail: string }
    /** A refusal for the organization's own key list. */
    keysRefusal?: { status: number; detail: string }
    /** A refusal for the model catalog the suggestions come from. */
    catalogRefusal?: { status: number; detail: string }
    /** Whether the workspace requires zero data retention. */
    requireZeroRetention?: boolean
  } = {},
) {
  const keys = opts.keys ?? [orgProviderKey()]
  const overrides = opts.overrides ?? [workspaceProviderKeyOverride()]
  const hosted = opts.hosted ?? []
  const catalog = opts.catalog ?? []
  const requests: Request[] = []

  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })

    if (opts.writeRefusal && method === opts.writeRefusal.method) {
      return jsonResponse(
        { detail: opts.writeRefusal.detail },
        opts.writeRefusal.status,
      )
    }
    if (url.startsWith(`${API_ROOT}/models`)) {
      if (opts.catalogRefusal) {
        return jsonResponse(
          { detail: opts.catalogRefusal.detail },
          opts.catalogRefusal.status,
        )
      }
      return jsonResponse({
        data: catalog.map((id) => ({ id, owned_by: id.split(":")[0] })),
      })
    }
    if (url.startsWith(`${API_ROOT}/catalog/models`)) {
      // The dialog reads one provider's models from the grouped catalog, each
      // as a selector; no vendor, so the name before a slash stands in.
      const provider = new URL(url, "http://localhost").searchParams.get(
        "provider",
      )
      return jsonResponse({
        models: catalog
          .filter((id) => id.startsWith(`${provider}:`))
          .map((id) => ({ selectors: [id], vendor: null })),
        count: catalog.length,
      })
    }
    if (url.includes("/provider-policy")) {
      if (method !== "GET")
        return jsonResponse(init?.body ? JSON.parse(String(init.body)) : {})
      return jsonResponse({
        require_zero_data_retention: opts.requireZeroRetention ?? false,
      })
    }
    if (url.includes("/hosted-providers")) {
      if (method !== "GET") return jsonResponse(hosted[0] ?? {})
      return jsonResponse({ data: hosted })
    }
    if (url.includes(`${API_ROOT}/workspaces/`)) {
      if (method !== "GET") return jsonResponse({ message: "ok" })
      return jsonResponse({ data: overrides })
    }
    if (url.includes(`${API_ROOT}/organizations/me/provider-keys`)) {
      if (opts.keysRefusal) {
        return jsonResponse(
          { detail: opts.keysRefusal.detail },
          opts.keysRefusal.status,
        )
      }
      return jsonResponse({ data: keys, count: keys.length })
    }
    return jsonResponse({})
  })

  return requests
}

function renderSection({
  canManage = true,
  canManageOrganization = true,
}: {
  canManage?: boolean
  canManageOrganization?: boolean
} = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <InferenceAccessSection
        workspaceId={WORKSPACE}
        canManage={canManage}
        canManageOrganization={canManageOrganization}
      />
    </QueryClientProvider>,
  )
}

/** Open a key's model dialog from its row's scope control. */
async function openKeyModels(
  user: ReturnType<typeof userEvent.setup>,
  name = "openai / Production",
) {
  await user.click(
    await screen.findByRole("button", { name: new RegExp(`on ${name}$`) }),
  )
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("InferenceAccessSection: the organization's own keys", () => {
  it("names every inherited key and says nothing is narrowed", async () => {
    mockApi()
    renderSection()

    expect(await screen.findByText("openai / Production")).toBeInTheDocument()
    expect(screen.getByText("Your key")).toBeInTheDocument()
    expect(
      screen.getByRole("button", {
        name: "Models: All on openai / Production",
      }),
    ).toBeInTheDocument()
    expect(screen.queryByText("Customized")).toBeNull()
  })

  it("shows a migrated restriction as the narrowing it is", async () => {
    mockApi({
      overrides: [
        workspaceProviderKeyOverride({
          allowed_models: ["gpt-4o", "gpt-4o-mini"],
        }),
      ],
    })
    const user = userEvent.setup()
    renderSection()

    expect(
      await screen.findByRole("button", {
        name: "Models: 2 selected on openai / Production",
      }),
    ).toBeInTheDocument()
    expect(screen.getByText("Customized")).toBeInTheDocument()
    await openKeyModels(user)
    expect(await screen.findByText("gpt-4o")).toBeInTheDocument()
    expect(screen.getByText("gpt-4o-mini")).toBeInTheDocument()
  })

  it("says a key is unreadable", async () => {
    mockApi({ overrides: [workspaceProviderKeyOverride({ usable: false })] })
    renderSection()

    expect(await screen.findByText(/Unreadable credential/)).toBeInTheDocument()
  })

  it("reads the allow-list off the row rather than asking per key", async () => {
    const requests = mockApi({
      overrides: [workspaceProviderKeyOverride({ allowed_models: ["gpt-4o"] })],
    })
    renderSection()

    await screen.findByText("openai / Production")
    expect(
      requests.filter((request) => request.url.includes("/provider-keys/")),
    ).toEqual([])
  })

  it("narrows a key to the models ticked, adding and removing to match", async () => {
    const requests = mockApi({
      catalog: ["openai:gpt-4o", "openai:gpt-4o-mini", "anthropic:claude"],
      overrides: [workspaceProviderKeyOverride({ allowed_models: ["gpt-4o"] })],
    })
    const user = userEvent.setup()
    renderSection()

    await openKeyModels(user)
    // One vendor, so its models are already open; another provider's are not offered.
    await user.click(
      await screen.findByRole("checkbox", { name: "gpt-4o-mini" }),
    )
    await user.click(screen.getByRole("checkbox", { name: "gpt-4o" }))
    expect(screen.queryByRole("checkbox", { name: "claude" })).toBeNull()
    await user.click(screen.getByRole("button", { name: "Save" }))

    await vi.waitFor(() =>
      expect(
        requests
          .filter((request) => request.method !== "GET")
          .map((request) => [request.method, request.url, request.body]),
      ).toEqual([
        [
          "POST",
          `${API_ROOT}/workspaces/${WORKSPACE}/provider-keys/${OPENAI_KEY}/models`,
          { model: "gpt-4o-mini" },
        ],
        [
          "DELETE",
          `${API_ROOT}/workspaces/${WORKSPACE}/provider-keys/${OPENAI_KEY}/models/gpt-4o`,
          undefined,
        ],
      ]),
    )
  })

  it("lifts a key's restriction by allowing every model again", async () => {
    const requests = mockApi({
      catalog: ["openai:gpt-4o"],
      overrides: [workspaceProviderKeyOverride({ allowed_models: ["gpt-4o"] })],
    })
    const user = userEvent.setup()
    renderSection()

    await openKeyModels(user)
    await user.click(
      await screen.findByRole("radio", {
        name: "All models, including ones added later",
      }),
    )
    await user.click(screen.getByRole("button", { name: "Save" }))

    await vi.waitFor(() =>
      expect(requests.find((request) => request.method === "DELETE")?.url).toBe(
        `${API_ROOT}/workspaces/${WORKSPACE}/provider-keys/${OPENAI_KEY}/models/gpt-4o`,
      ),
    )
  })

  it("selects every model of one vendor at once", async () => {
    // A provider that routes to others names the vendor before the slash.
    const requests = mockApi({
      catalog: [
        "openai:deepseek-ai/DeepSeek-R1",
        "openai:deepseek-ai/DeepSeek-V3",
        "openai:meta-llama/Llama-3",
      ],
    })
    const user = userEvent.setup()
    renderSection()

    await openKeyModels(user)
    await user.click(
      await screen.findByRole("radio", { name: "Only these models" }),
    )
    // Two vendors, so each is folded until opened.
    expect(screen.queryByRole("checkbox", { name: /DeepSeek-R1/ })).toBeNull()
    await user.click(
      screen.getByRole("checkbox", { name: "All deepseek-ai models" }),
    )
    expect(screen.getByText("2 of 2")).toBeInTheDocument()
    expect(screen.getByText("0 of 1")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Save" }))

    await vi.waitFor(() =>
      expect(
        requests
          .filter((request) => request.method === "POST")
          .map((request) => request.body),
      ).toEqual([
        { model: "deepseek-ai/DeepSeek-R1" },
        { model: "deepseek-ai/DeepSeek-V3" },
      ]),
    )
  })

  it("marks a vendor with some of its models picked as partly selected", async () => {
    mockApi({
      catalog: [
        "openai:deepseek-ai/DeepSeek-R1",
        "openai:deepseek-ai/DeepSeek-V3",
        "openai:meta-llama/Llama-3",
      ],
      overrides: [
        workspaceProviderKeyOverride({
          allowed_models: ["deepseek-ai/DeepSeek-R1"],
        }),
      ],
    })
    const user = userEvent.setup()
    renderSection()

    await openKeyModels(user)
    const vendor = await screen.findByRole("checkbox", {
      name: "All deepseek-ai models",
    })
    expect(vendor).toBePartiallyChecked()
    await user.click(
      screen.getByRole("button", { name: "Show deepseek-ai models" }),
    )
    expect(
      screen.getByRole("checkbox", { name: "deepseek-ai/DeepSeek-R1" }),
    ).toBeChecked()
  })

  it("takes typed names where the catalog lists nothing for the key", async () => {
    const requests = mockApi({
      keys: [orgProviderKey({ provider: "ollama", name: "Local" })],
    })
    const user = userEvent.setup()
    renderSection()

    await openKeyModels(user, "ollama / Local")
    await user.click(
      await screen.findByRole("radio", { name: "Only these models" }),
    )
    const field = screen.getByRole("textbox", { name: "Model name" })
    await user.type(field, "ollama:llama3")
    expect(
      screen.getByText('Name the model without its "ollama:" prefix.'),
    ).toBeInTheDocument()
    await user.clear(field)
    // A colon inside a model's own name is not a provider prefix.
    // Enter adds the name; it neither saves nor closes the dialog.
    await user.type(field, "llama3:8b{Enter}")
    expect(
      screen.getByRole("button", { name: "Remove llama3:8b" }),
    ).toBeInTheDocument()
    expect(requests.filter((request) => request.method === "POST")).toEqual([])
    await user.click(screen.getByRole("button", { name: "Save" }))

    await vi.waitFor(() =>
      expect(
        requests.find((request) => request.method === "POST")?.body,
      ).toEqual({ model: "llama3:8b" }),
    )
  })

  it("switches a key off with the one flag that means it", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    renderSection()

    await user.click(
      await screen.findByRole("switch", {
        name: "Use openai / Production in this workspace",
      }),
    )

    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.url).toBe(
      `${API_ROOT}/workspaces/${WORKSPACE}/provider-keys/${OPENAI_KEY}`,
    )
    expect(patch?.body).toEqual({ disabled: true })
  })

  it("offers no model list on a key switched off", async () => {
    // Disabling deletes the key's allow-list server-side and refuses a later
    // add, so the control goes with it rather than refusing on press.
    mockApi({ overrides: [workspaceProviderKeyOverride({ disabled: true })] })
    renderSection()

    expect(
      await screen.findByRole("switch", {
        name: "Use openai / Production in this workspace",
      }),
    ).not.toBeChecked()
    expect(
      screen.queryByRole("button", { name: /^Models: .* on openai/ }),
    ).toBeNull()
  })

  it("offers to always use a key only where its provider has another", async () => {
    mockApi()
    renderSection()

    await screen.findByText("openai / Production")
    expect(
      screen.queryByRole("button", { name: "More for openai / Production" }),
    ).toBeNull()
  })

  it("pins one of a provider's keys from the row's menu", async () => {
    const requests = mockApi({
      keys: [
        orgProviderKey(),
        orgProviderKey({ id: SECOND_OPENAI_KEY, name: "Staging" }),
      ],
      overrides: [
        workspaceProviderKeyOverride(),
        workspaceProviderKeyOverride({
          org_provider_key_id: SECOND_OPENAI_KEY,
          is_effective_default: false,
        }),
      ],
    })
    const user = userEvent.setup()
    renderSection()

    // Two keys of one provider fold under its row, which opens in place.
    const group = await screen.findByRole("button", { name: /^openai/ })
    expect(group).toHaveTextContent("2 keys")
    expect(group).toHaveAttribute("aria-expanded", "false")
    await user.click(group)
    await user.click(
      await screen.findByRole("button", { name: "More for openai / Staging" }),
    )
    await user.click(
      await screen.findByRole("menuitem", { name: "Always use this key" }),
    )

    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.url).toBe(
      `${API_ROOT}/workspaces/${WORKSPACE}/provider-keys/${SECOND_OPENAI_KEY}`,
    )
    expect(patch?.body).toEqual({ is_default: true })
  })

  it("resets a key by deleting its override and its allow-list", async () => {
    const requests = mockApi({
      overrides: [
        workspaceProviderKeyOverride({
          is_default: true,
          allowed_models: ["gpt-4o"],
        }),
      ],
    })
    const user = userEvent.setup()
    renderSection()

    await user.click(
      await screen.findByRole("button", {
        name: "More for openai / Production",
      }),
    )
    await user.click(await screen.findByRole("menuitem", { name: "Reset" }))

    await vi.waitFor(() =>
      expect(
        requests
          .filter((request) => request.method === "DELETE")
          .map((request) => request.url),
      ).toEqual([
        `${API_ROOT}/workspaces/${WORKSPACE}/provider-keys/${OPENAI_KEY}`,
        `${API_ROOT}/workspaces/${WORKSPACE}/provider-keys/${OPENAI_KEY}/models/gpt-4o`,
      ]),
    )
  })

  it("reports a refused write rather than showing it as applied", async () => {
    mockApi({
      writeRefusal: {
        method: "PATCH",
        status: 403,
        detail: "Not authorized to manage this workspace",
      },
    })
    const user = userEvent.setup()
    renderSection()

    const toggle = await screen.findByRole("switch", {
      name: "Use openai / Production in this workspace",
    })
    await user.click(toggle)

    expect(await screen.findByRole("alert")).toHaveTextContent(/Not authorized/)
    expect(toggle).toBeChecked()
  })

  it("reports a refused key list rather than naming keys by id in silence", async () => {
    mockApi({
      keysRefusal: { status: 403, detail: "Organization admins only" },
    })
    renderSection()

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Organization admins only/,
    )
  })

  it("does not ask a workspace admin for the organization's key list", async () => {
    const requests = mockApi()
    renderSection({ canManageOrganization: false })

    expect(await screen.findByText("Unknown key 66666666")).toBeInTheDocument()
    expect(
      requests.filter((request) =>
        request.url.includes("/organizations/me/provider-keys"),
      ),
    ).toEqual([])
  })

  it("names a key the organization list does not carry as unknown", async () => {
    mockApi({
      overrides: [
        workspaceProviderKeyOverride({ org_provider_key_id: ANTHROPIC_KEY }),
      ],
    })
    renderSection()

    expect(await screen.findByText("Unknown key 77777777")).toBeInTheDocument()
  })

  it("holds every control for a caller who cannot manage the workspace", async () => {
    mockApi()
    renderSection({ canManage: false, canManageOrganization: false })

    expect(
      await screen.findByRole("switch", {
        name: "Use Unknown key 66666666 in this workspace",
      }),
    ).toBeDisabled()
  })
})

describe("InferenceAccessSection: zero data retention", () => {
  it("requires it from the switch above the list", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    renderSection()

    await user.click(
      await screen.findByRole("switch", {
        name: "Require zero data retention",
      }),
    )

    const put = requests.find((request) => request.method === "PUT")
    expect(put?.url).toBe(`${API_ROOT}/workspaces/${WORKSPACE}/provider-policy`)
    expect(put?.body).toEqual({ require_zero_data_retention: true })
  })

  it("says which sources it blocks and which are declared", async () => {
    mockApi({
      requireZeroRetention: true,
      overrides: [workspaceProviderKeyOverride({ zero_data_retention: false })],
      hosted: [workspaceHostedProvider({ zero_data_retention: true })],
    })
    renderSection()

    expect(
      await screen.findByText(/Blocked, not zero data retention/),
    ).toBeInTheDocument()
    expect(
      await screen.findByText("Billed by deployment · Zero data retention"),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("switch", { name: "Require zero data retention" }),
    ).toBeChecked()
  })

  it("holds every control on a source it blocks, and none on a declared one", async () => {
    mockApi({
      requireZeroRetention: true,
      overrides: [workspaceProviderKeyOverride({ zero_data_retention: false })],
      hosted: [workspaceHostedProvider({ zero_data_retention: true })],
    })
    renderSection()

    expect(
      await screen.findByRole("switch", {
        name: "Use openai / Production in this workspace",
      }),
    ).toBeDisabled()
    expect(
      screen.getByRole("button", {
        name: "Models: All on openai / Production",
      }),
    ).toBeDisabled()
    expect(
      screen.getByRole("switch", { name: "Use gemini in this workspace" }),
    ).toBeEnabled()
    expect(
      screen.getByRole("button", { name: "Models: All on gemini" }),
    ).toBeEnabled()
  })

  it("holds the switch for a caller who cannot manage the workspace", async () => {
    mockApi()
    renderSection({ canManage: false })

    expect(
      await screen.findByRole("switch", {
        name: "Require zero data retention",
      }),
    ).toBeDisabled()
  })
})

describe("InferenceAccessSection: a long list", () => {
  const providers = Array.from({ length: 9 }, (_, index) => `provider-${index}`)

  it("counts what is set and offers no search under the threshold", async () => {
    mockApi({
      overrides: [],
      hosted: providers
        .slice(0, 3)
        .map((provider) => workspaceHostedProvider({ provider })),
    })
    renderSection()

    expect(await screen.findByText("3 providers")).toBeInTheDocument()
    expect(screen.queryByLabelText("Search providers")).toBeNull()
  })

  it("narrows a long list to what is switched off, then by name", async () => {
    mockApi({
      overrides: [],
      hosted: providers.map((provider, index) =>
        workspaceHostedProvider({
          provider,
          enabled: index !== 4,
          customized: index === 4,
        }),
      ),
    })
    const user = userEvent.setup()
    renderSection()

    expect(
      await screen.findByText("9 providers · 1 customized · 1 off"),
    ).toBeInTheDocument()
    await user.click(screen.getByRole("radio", { name: "Off" }))

    expect(screen.getByText("provider-4")).toBeInTheDocument()
    expect(screen.queryByText("provider-0")).toBeNull()

    await user.click(screen.getByRole("radio", { name: "All" }))
    await user.type(screen.getByLabelText("Search providers"), "provider-7")

    expect(await screen.findByText("provider-7")).toBeInTheDocument()
    await vi.waitFor(() => expect(screen.queryByText("provider-0")).toBeNull())
  })
})

describe("InferenceAccessSection: hosted providers", () => {
  it("lists a hosted provider as billed by the deployment", async () => {
    mockApi({ overrides: [], hosted: [workspaceHostedProvider()] })
    renderSection()

    expect(await screen.findByText("gemini")).toBeInTheDocument()
    expect(screen.getByText("Billed by deployment")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Models: All on gemini" }),
    ).toBeInTheDocument()
  })

  it("says a hosted provider the organization's key serves is not used", async () => {
    mockApi({
      overrides: [],
      hosted: [workspaceHostedProvider({ served_by_own_key: true })],
    })
    renderSection()

    expect(
      await screen.findByText(/Not used, your key serves this provider/),
    ).toBeInTheDocument()
  })

  it("switches a hosted provider off, keeping its model list", async () => {
    const requests = mockApi({
      overrides: [],
      hosted: [workspaceHostedProvider({ models: ["gemini-2.5-pro"] })],
    })
    const user = userEvent.setup()
    renderSection()

    await user.click(
      await screen.findByRole("switch", {
        name: "Use gemini in this workspace",
      }),
    )

    const put = requests.find((request) => request.method === "PUT")
    expect(put?.url).toBe(
      `${API_ROOT}/workspaces/${WORKSPACE}/hosted-providers/gemini`,
    )
    expect(put?.body).toEqual({ enabled: false, models: ["gemini-2.5-pro"] })
  })

  it("narrows a hosted provider to the models ticked, and not before one is", async () => {
    const requests = mockApi({
      overrides: [],
      hosted: [workspaceHostedProvider()],
    })
    const user = userEvent.setup()
    renderSection()

    await user.click(
      await screen.findByRole("button", { name: "Models: All on gemini" }),
    )
    await user.click(
      await screen.findByRole("radio", { name: "Only these models" }),
    )
    const save = screen.getByRole("button", { name: "Save" })
    // "Only" with nothing picked is unfinished rather than a way to block.
    expect(save).toBeDisabled()
    await user.click(screen.getByRole("checkbox", { name: "gemini-2.5-pro" }))
    await user.click(save)

    const put = requests.find((request) => request.method === "PUT")
    expect(put?.body).toEqual({ enabled: true, models: ["gemini-2.5-pro"] })
  })

  it("takes typed names where the deployment advertises no roster", async () => {
    const requests = mockApi({
      overrides: [],
      hosted: [workspaceHostedProvider({ served_models: null })],
    })
    const user = userEvent.setup()
    renderSection()

    await user.click(
      await screen.findByRole("button", { name: "Models: All on gemini" }),
    )
    await user.click(
      await screen.findByRole("radio", { name: "Only these models" }),
    )
    await user.type(
      screen.getByRole("textbox", { name: "Model name" }),
      "gemini-3",
    )
    await user.click(screen.getByRole("button", { name: "Add" }))
    await user.click(screen.getByRole("button", { name: "Save" }))

    const put = requests.find((request) => request.method === "PUT")
    expect(put?.body).toEqual({ enabled: true, models: ["gemini-3"] })
  })

  it("resets a customized hosted provider", async () => {
    const requests = mockApi({
      overrides: [],
      hosted: [workspaceHostedProvider({ enabled: false, customized: true })],
    })
    const user = userEvent.setup()
    renderSection()

    await user.click(
      await screen.findByRole("button", { name: "More for gemini" }),
    )
    await user.click(await screen.findByRole("menuitem", { name: "Reset" }))

    const reset = requests.find((request) => request.method === "DELETE")
    expect(reset?.url).toBe(
      `${API_ROOT}/workspaces/${WORKSPACE}/hosted-providers/gemini`,
    )
  })
})
