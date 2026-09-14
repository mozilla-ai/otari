import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { OrgProviderKey, WorkspaceProviderKeyOverride } from "@/client"
import { WorkspaceProviderKeys } from "@/features/workspaces/WorkspaceProviderKeys"
import { API_ROOT } from "@/shared/api/client"
import { orgProviderKey, workspaceProviderKeyOverride } from "@/tests/fixtures"
import { pickOption } from "@/tests/select"

const WORKSPACE = "44444444-4444-4444-4444-444444444444"
const OPENAI_KEY = "66666666-6666-6666-6666-666666666666"
const ANTHROPIC_KEY = "77777777-7777-7777-7777-777777777777"

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
    /** Catalog ids, as `/models` spells them: `provider:model`. */
    catalog?: string[]
    /** A refusal for whichever write matches, for the error paths. */
    writeRefusal?: { method: string; status: number; detail: string }
    /** A refusal for the organization's own key list. */
    keysRefusal?: { status: number; detail: string }
    /** A refusal for the model catalog the suggestions come from. */
    catalogRefusal?: { status: number; detail: string }
  } = {},
) {
  const keys = opts.keys ?? [orgProviderKey()]
  const overrides = opts.overrides ?? [workspaceProviderKeyOverride()]
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

function renderSection() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <WorkspaceProviderKeys workspaceId={WORKSPACE} />
    </QueryClientProvider>,
  )
}

/** The combo box that adds one model to a key's allow-list. */
function allowField() {
  return screen.findByRole("combobox", {
    name: "Allow a model on openai / Production",
  })
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("WorkspaceProviderKeys", () => {
  it("names every inherited key and says nothing is narrowed", async () => {
    mockApi()
    renderSection()

    expect(await screen.findByText("openai / Production")).toBeInTheDocument()
    expect(
      await screen.findByText("Every model this key serves is allowed."),
    ).toBeInTheDocument()
    // The key resolving for this workspace, which the flags alone do not say.
    expect(screen.getByText("In use")).toBeInTheDocument()
  })

  it("says the section saves as it goes, unlike the form around it", async () => {
    // The fields above this section wait for Save, so an admin who disables a
    // key and then presses Cancel would otherwise expect it undone.
    mockApi()
    renderSection()

    expect(
      await screen.findByText(
        /Each change in this section is saved as you make it/,
      ),
    ).toBeInTheDocument()
  })

  it("shows a migrated restriction as the narrowing it is", async () => {
    // The state the cutover creates: a workspace whose catalog is narrowed to
    // named models, which until now was invisible in the dashboard.
    mockApi({
      overrides: [
        workspaceProviderKeyOverride({
          allowed_models: ["gpt-4o", "gpt-4o-mini"],
        }),
      ],
    })
    renderSection()

    expect(await screen.findByText("gpt-4o")).toBeInTheDocument()
    expect(screen.getByText("gpt-4o-mini")).toBeInTheDocument()
    expect(
      screen.queryByText("Every model this key serves is allowed."),
    ).toBeNull()
  })

  it("reads the allow-list off the row rather than asking per key", async () => {
    // The narrowing travels with the flags, so the section makes one read for
    // the workspace instead of one more for every key it holds.
    const requests = mockApi({
      overrides: [workspaceProviderKeyOverride({ allowed_models: ["gpt-4o"] })],
    })
    renderSection()

    await screen.findByText("gpt-4o")
    expect(
      requests.filter((request) => request.url.includes("/provider-keys/")),
    ).toEqual([])
  })

  it("lifts a restriction through the model the operator names", async () => {
    const requests = mockApi({
      overrides: [workspaceProviderKeyOverride({ allowed_models: ["gpt-4o"] })],
    })
    const user = userEvent.setup()
    renderSection()

    await user.click(
      await screen.findByRole("button", {
        name: "Stop allowing gpt-4o on openai / Production",
      }),
    )

    const removed = requests.find((request) => request.method === "DELETE")
    expect(removed?.url).toBe(
      `${API_ROOT}/workspaces/${WORKSPACE}/provider-keys/${OPENAI_KEY}/models/gpt-4o`,
    )
  })

  it("narrows a key to one more model", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    renderSection()

    await user.type(await allowField(), "gpt-4o")
    // The suggestion popover is open on what was typed, and react-aria marks
    // everything outside it aria-hidden, so it is dismissed before the button
    // beside it is reached for.
    await user.keyboard("{Escape}")
    await user.click(
      screen.getByRole("button", {
        name: "Allow a model on openai / Production",
      }),
    )

    const added = requests.find((request) => request.method === "POST")
    expect(added?.url).toBe(
      `${API_ROOT}/workspaces/${WORKSPACE}/provider-keys/${OPENAI_KEY}/models`,
    )
    expect(added?.body).toEqual({ model: "gpt-4o" })
  })

  it("suggests the catalog entries of this key's own provider", async () => {
    // A shortcut, not a whitelist: the models of another provider are not this
    // key's to allow, and one already allowed is not worth offering twice.
    mockApi({
      catalog: ["openai:gpt-4o", "openai:gpt-4o-mini", "anthropic:claude"],
      overrides: [
        workspaceProviderKeyOverride({ allowed_models: ["gpt-4o-mini"] }),
      ],
    })
    const user = userEvent.setup()
    renderSection()

    await user.type(await allowField(), "gpt")

    expect(
      await screen.findByRole("option", { name: "gpt-4o" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("option", { name: "claude" })).toBeNull()
    expect(screen.queryByRole("option", { name: "gpt-4o-mini" })).toBeNull()
  })

  it("says an exhausted list is exhausted, not that the provider has no models", async () => {
    mockApi({
      catalog: ["openai:gpt-4o"],
      overrides: [workspaceProviderKeyOverride({ allowed_models: ["gpt-4o"] })],
    })
    const user = userEvent.setup()
    renderSection()

    await user.type(await allowField(), "z")

    expect(
      await screen.findByText(
        "Every model the catalog lists for this provider is already allowed.",
      ),
    ).toBeInTheDocument()
  })

  it("says a refused catalog was refused, not that the provider has no models", async () => {
    // Both leave the same empty popover behind, and only one of them is a fact
    // about the provider.
    mockApi({ catalogRefusal: { status: 503, detail: "Discovery is down" } })
    const user = userEvent.setup()
    renderSection()

    await user.type(await allowField(), "g")

    expect(
      await screen.findByText(/The model catalog could not be read/),
    ).toBeInTheDocument()
  })

  it("refuses a model still carrying its provider prefix", async () => {
    // The catalog spells a model `openai:gpt-4o` and the allow-list stores the
    // bare name, so the id copied off the models page is the one entry that
    // looks right and narrows the workspace to a model that does not exist.
    const requests = mockApi({ catalog: ["openai:gpt-4o"] })
    const user = userEvent.setup()
    renderSection()

    await user.type(await allowField(), "openai:gpt-4o")

    expect(
      await screen.findByText('Name the model without its "openai:" prefix.'),
    ).toBeInTheDocument()
    await user.keyboard("{Escape}")
    expect(
      screen.getByRole("button", {
        name: "Allow a model on openai / Production",
      }),
    ).toBeDisabled()
    expect(requests.filter((request) => request.method === "POST")).toEqual([])
  })

  it("refuses another provider's prefix, not just this key's own", async () => {
    // `anthropic:claude` on an openai key is the same mistake one step further
    // along: stored as written, it narrows the workspace to a model that does
    // not exist, and it is not this key's to allow either way.
    const requests = mockApi({
      keys: [
        orgProviderKey(),
        orgProviderKey({
          id: ANTHROPIC_KEY,
          provider: "anthropic",
          name: "Backup",
        }),
      ],
    })
    const user = userEvent.setup()
    renderSection()

    await user.type(await allowField(), "anthropic:claude")

    expect(
      await screen.findByText(
        '"anthropic:" names another provider. Name a model this key serves.',
      ),
    ).toBeInTheDocument()
    expect(requests.filter((request) => request.method === "POST")).toEqual([])
  })

  it("accepts a model whose own name carries a colon", async () => {
    // `llama3:8b` is a whole model name on Ollama, not a prefixed one, so the
    // check matches the providers this deployment names rather than any text
    // before a colon.
    const requests = mockApi({
      keys: [orgProviderKey({ provider: "ollama", name: "Local" })],
    })
    const user = userEvent.setup()
    renderSection()

    await user.type(
      await screen.findByRole("combobox", {
        name: "Allow a model on ollama / Local",
      }),
      "llama3:8b",
    )
    await user.keyboard("{Escape}")
    await user.click(
      screen.getByRole("button", { name: "Allow a model on ollama / Local" }),
    )

    const added = requests.find((request) => request.method === "POST")
    expect(added?.body).toEqual({ model: "llama3:8b" })
  })

  it("refuses a model the key already allows", async () => {
    mockApi({
      overrides: [workspaceProviderKeyOverride({ allowed_models: ["gpt-4o"] })],
    })
    const user = userEvent.setup()
    renderSection()

    await user.type(await allowField(), "gpt-4o")

    expect(
      await screen.findByText("This model is already allowed on this key."),
    ).toBeInTheDocument()
    await user.keyboard("{Escape}")
    expect(
      screen.getByRole("button", {
        name: "Allow a model on openai / Production",
      }),
    ).toBeDisabled()
  })

  it("pins a key as this workspace's default with one flag", async () => {
    // Both flags true is the one combination the gateway refuses, so the
    // picker sends the one it means and leaves the other to auto-resolve.
    const requests = mockApi()
    const user = userEvent.setup()
    renderSection()

    await screen.findByText("openai / Production")
    await pickOption(
      user,
      "This workspace's use of openai / Production",
      "Always use this key",
    )

    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.url).toBe(
      `${API_ROOT}/workspaces/${WORKSPACE}/provider-keys/${OPENAI_KEY}`,
    )
    expect(patch?.body).toEqual({ is_default: true })
  })

  it("disables a key, and stops offering an allow-list it cannot write", async () => {
    // Disabling deletes the key's allow-list server-side and refuses a later
    // add, so the controls go with it rather than refusing on press.
    mockApi({
      overrides: [workspaceProviderKeyOverride({ disabled: true })],
    })
    renderSection()

    expect(
      await screen.findByText(
        "No model of this key is available to this workspace.",
      ),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", {
        name: "Allow a model on openai / Production",
      }),
    ).toBeNull()
  })

  it("reverts to inheritance by deleting the override", async () => {
    const requests = mockApi({
      overrides: [workspaceProviderKeyOverride({ is_default: true })],
    })
    const user = userEvent.setup()
    renderSection()

    await screen.findByText("openai / Production")
    await pickOption(
      user,
      "This workspace's use of openai / Production",
      "Follow the organization",
    )

    const reset = requests.find((request) => request.method === "DELETE")
    expect(reset?.url).toBe(
      `${API_ROOT}/workspaces/${WORKSPACE}/provider-keys/${OPENAI_KEY}`,
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

    await screen.findByText("openai / Production")
    await pickOption(
      user,
      "This workspace's use of openai / Production",
      "Never use this key",
    )

    expect(await screen.findByRole("alert")).toHaveTextContent(/Not authorized/)
    // The row still shows what the gateway holds, not what was asked for.
    expect(
      screen.queryByText(
        "No model of this key is available to this workspace.",
      ),
    ).toBeNull()
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

  it("names a key the organization list does not carry as unknown", async () => {
    // Not a state the gateway produces, since both reads cover the same
    // non-archived set, but a bare UUID named nothing a person could act on.
    mockApi({
      overrides: [
        workspaceProviderKeyOverride({ org_provider_key_id: ANTHROPIC_KEY }),
      ],
    })
    renderSection()

    expect(await screen.findByText("Unknown key 77777777")).toBeInTheDocument()
  })
})
