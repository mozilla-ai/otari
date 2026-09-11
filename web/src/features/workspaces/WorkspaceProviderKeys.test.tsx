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
    /** The allow-list each key carries, by key id. Absent means every model. */
    models?: Record<string, string[]>
    /** A refusal for whichever write matches, for the error paths. */
    writeRefusal?: { method: string; status: number; detail: string }
    /** A refusal for the organization's own key list. */
    keysRefusal?: { status: number; detail: string }
    /** A refusal for a key's allow-list read. */
    modelsRefusal?: { status: number; detail: string }
  } = {},
) {
  const keys = opts.keys ?? [orgProviderKey()]
  const overrides = opts.overrides ?? [workspaceProviderKeyOverride()]
  const models = opts.models ?? {}
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
    if (url.includes("/models")) {
      if (method !== "GET") return jsonResponse({ message: "ok" })
      if (opts.modelsRefusal) {
        return jsonResponse(
          { detail: opts.modelsRefusal.detail },
          opts.modelsRefusal.status,
        )
      }
      const keyId = url.split("/provider-keys/")[1]?.split("/")[0] ?? ""
      return jsonResponse({ models: models[keyId] ?? [] })
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

  it("shows a migrated restriction as the narrowing it is", async () => {
    // The state the cutover creates: a workspace whose catalog is narrowed to
    // named models, which until now was invisible in the dashboard.
    mockApi({ models: { [OPENAI_KEY]: ["gpt-4o", "gpt-4o-mini"] } })
    renderSection()

    expect(await screen.findByText("gpt-4o")).toBeInTheDocument()
    expect(screen.getByText("gpt-4o-mini")).toBeInTheDocument()
    expect(
      screen.queryByText("Every model this key serves is allowed."),
    ).toBeNull()
  })

  it("lifts a restriction through the model the operator names", async () => {
    const requests = mockApi({ models: { [OPENAI_KEY]: ["gpt-4o"] } })
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

    await user.type(
      await screen.findByLabelText("Allow a model on openai / Production"),
      "gpt-4o",
    )
    await user.click(screen.getByRole("button", { name: "Allow" }))

    const added = requests.find((request) => request.method === "POST")
    expect(added?.url).toBe(
      `${API_ROOT}/workspaces/${WORKSPACE}/provider-keys/${OPENAI_KEY}/models`,
    )
    expect(added?.body).toEqual({ model: "gpt-4o" })
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
      "Workspace default",
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
    expect(screen.queryByRole("button", { name: "Allow" })).toBeNull()
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
      "Inherited",
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
      "Disabled",
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

  it("says a refused allow-list was refused, not that it is open or loading", async () => {
    // An empty list is the answer "every model is allowed", so a read that
    // failed must not borrow it: that would report a narrowing still in force as
    // lifted. Nor may it sit on "loading", which claims an answer is still
    // coming from a read that already gave one.
    mockApi({
      modelsRefusal: { status: 403, detail: "Not a member of this workspace" },
    })
    renderSection()

    expect(await screen.findByRole("alert")).toHaveTextContent(/Not a member/)
    expect(
      await screen.findByText(
        "The allowed models for this key could not be read.",
      ),
    ).toBeInTheDocument()
    expect(
      screen.queryByText("Every model this key serves is allowed."),
    ).toBeNull()
    expect(screen.queryByText("Loading allowed models…")).toBeNull()
  })

  it("names a key the organization list does not carry by its id", async () => {
    // Not a state the gateway produces, since both reads cover the same
    // non-archived set, but the row stays addressable if it ever does.
    mockApi({
      overrides: [
        workspaceProviderKeyOverride({ org_provider_key_id: ANTHROPIC_KEY }),
      ],
    })
    renderSection()

    expect(await screen.findByText(ANTHROPIC_KEY)).toBeInTheDocument()
  })
})
