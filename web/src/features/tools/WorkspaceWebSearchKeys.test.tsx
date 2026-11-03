import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  SearchProviderInfo,
  WorkspaceWebSearchKey,
  WorkspaceWebSearchKeys,
} from "@/client"
import {
  WorkspaceProviderOptions,
  WorkspaceWebSearchKeyChoice,
} from "@/features/tools/WorkspaceWebSearchKeys"
import { pickOption } from "@/tests/select"

const WORKSPACE = "11111111-1111-1111-1111-111111111111"
const AAA = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
const BBB = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
const MOJEEK = "cccccccc-cccc-cccc-cccc-cccccccccccc"

function key(overrides: Partial<WorkspaceWebSearchKey>): WorkspaceWebSearchKey {
  return {
    org_web_search_key_id: AAA,
    workspace_id: WORKSPACE,
    kind: "search",
    provider: "exa",
    name: "AAA",
    last4: "a1b2",
    is_default: false,
    is_org_default: false,
    disabled: false,
    is_effective: false,
    usable: true,
    ...overrides,
  }
}

function listing(
  keys: WorkspaceWebSearchKey[],
  overrides: Partial<WorkspaceWebSearchKeys> = {},
): WorkspaceWebSearchKeys {
  return {
    data: keys,
    count: keys.length,
    search_provider: "exa",
    fetch_provider: "builtin",
    deployment_search_tool: "exa",
    deployment_fetch_tool: "builtin_fetch",
    ...overrides,
  }
}

const CATALOG: SearchProviderInfo[] = [
  {
    id: "exa",
    kind: "search",
    requires_api_key: true,
    requires_api_base: false,
    tier: "production",
    options: [
      {
        name: "type",
        type: "string",
        enum: ["auto", "fast"],
        default: "auto",
        description: "How Exa searches.",
        operator_only: false,
      },
      {
        name: "internal",
        type: "string",
        description: "Only an instance or a key sets this.",
        operator_only: true,
      },
    ],
  },
]

function mockApi(
  answer: WorkspaceWebSearchKeys,
  { failing }: { failing?: string } = {},
) {
  const calls: { url: string; method: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    calls.push({
      url,
      method,
      body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
    })
    if (url.includes("/search-tools/providers")) return Response.json(CATALOG)
    if (method === failing) {
      return Response.json({ detail: "Not found." }, { status: 404 })
    }
    return Response.json(answer)
  })
  return calls
}

function renderWith(ui: ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

function writes(
  calls: { url: string; method: string; body: unknown }[],
  method: string,
) {
  return calls.filter((call) => call.method === method)
}

describe("WorkspaceWebSearchKeyChoice", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("lists the deployment's default tool and each key, with the one in use chosen", async () => {
    mockApi(
      listing([
        key({ is_default: true, is_effective: true }),
        key({
          org_web_search_key_id: MOJEEK,
          provider: "mojeek",
          name: "M",
          last4: "e5f6",
          is_org_default: true,
        }),
      ]),
    )
    renderWith(
      <WorkspaceWebSearchKeyChoice workspaceId={WORKSPACE} canChoose />,
    )

    expect(
      await screen.findByRole("radio", { name: /exa · AAA/ }),
    ).toBeChecked()
    expect(screen.getByText("••••a1b2 · Pinned")).toBeInTheDocument()
    expect(
      screen.getByText("••••e5f6 · Your organization's default"),
    ).toBeInTheDocument()
    // The organization has a default, so the deployment's tool cannot be
    // pinned over it.
    expect(
      screen.getByRole("radio", {
        name: /This deployment's default tool · exa/,
      }),
    ).toBeDisabled()
    expect(
      screen.getByText("Used when your organization has no default."),
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "This deployment's default tool · builtin_fetch. Your organization has no fetch keys.",
      ),
    ).toBeInTheDocument()
  })

  it("pins the key chosen", async () => {
    const user = userEvent.setup()
    const calls = mockApi(
      listing([
        key({ is_effective: true, is_org_default: true }),
        key({ org_web_search_key_id: BBB, name: "BBB" }),
      ]),
    )
    renderWith(
      <WorkspaceWebSearchKeyChoice workspaceId={WORKSPACE} canChoose />,
    )

    await user.click(await screen.findByRole("radio", { name: /exa · BBB/ }))

    await waitFor(() => expect(writes(calls, "PATCH")).toHaveLength(1))
    expect(writes(calls, "PATCH")[0].url).toContain(
      `/workspaces/${WORKSPACE}/web-search-keys/${BBB}`,
    )
    expect(writes(calls, "PATCH")[0].body).toEqual({ is_default: true })
  })

  it("uses the organization's default again by clearing the pin and the default's turned-off mark", async () => {
    const user = userEvent.setup()
    const calls = mockApi(
      listing([
        key({ is_default: true, is_effective: true }),
        key({ org_web_search_key_id: BBB, name: "BBB", disabled: true }),
        key({
          org_web_search_key_id: MOJEEK,
          name: "M",
          is_org_default: true,
          disabled: true,
        }),
      ]),
    )
    renderWith(
      <WorkspaceWebSearchKeyChoice workspaceId={WORKSPACE} canChoose />,
    )

    await user.click(
      await screen.findByRole("button", {
        name: /Use the organization.s default/,
      }),
    )

    // BBB's mark changes nothing while the workspace follows, so it is kept.
    await waitFor(() => expect(writes(calls, "DELETE")).toHaveLength(2))
    expect(
      writes(calls, "DELETE").map((call) => call.url.split("/").at(-1)),
    ).toEqual([AAA, MOJEEK])
  })

  it("offers no way back to a default the workspace already uses", async () => {
    mockApi(
      listing([
        key({ is_org_default: true, is_effective: true }),
        key({ org_web_search_key_id: BBB, name: "BBB", disabled: true }),
      ]),
    )
    renderWith(
      <WorkspaceWebSearchKeyChoice workspaceId={WORKSPACE} canChoose />,
    )

    expect(
      await screen.findByRole("radio", { name: /exa · AAA/ }),
    ).toBeChecked()
    expect(
      screen.queryByRole("button", { name: /Use the organization/ }),
    ).toBeNull()
  })

  it("lets the workspace leave a default this deployment cannot decrypt for its own tool", async () => {
    const user = userEvent.setup()
    const calls = mockApi(
      listing([
        key({ is_default: true, is_effective: true }),
        key({
          org_web_search_key_id: MOJEEK,
          name: "M",
          is_org_default: true,
          usable: false,
        }),
      ]),
    )
    renderWith(
      <WorkspaceWebSearchKeyChoice workspaceId={WORKSPACE} canChoose />,
    )

    const deployment = await screen.findByRole("radio", {
      name: /This deployment's default tool/,
    })
    expect(deployment).toBeEnabled()
    expect(
      screen.queryByRole("button", { name: /Use the organization/ }),
    ).toBeNull()
    await user.click(deployment)

    await waitFor(() => expect(writes(calls, "DELETE")).toHaveLength(1))
    expect(writes(calls, "DELETE")[0].url).toContain(AAA)
  })

  it("reads the listing again after a reset that failed", async () => {
    const user = userEvent.setup()
    const calls = mockApi(
      listing([
        key({ is_default: true, is_effective: true }),
        key({ org_web_search_key_id: MOJEEK, name: "M", is_org_default: true }),
      ]),
      { failing: "DELETE" },
    )
    renderWith(
      <WorkspaceWebSearchKeyChoice workspaceId={WORKSPACE} canChoose />,
    )
    await user.click(
      await screen.findByRole("button", {
        name: /Use the organization.s default/,
      }),
    )

    expect(await screen.findByText("Not found.")).toBeInTheDocument()
    const listings = calls.filter(
      (call) => call.method === "GET" && call.url.includes("/web-search-keys"),
    )
    expect(listings.length).toBeGreaterThanOrEqual(2)
    expect(listings[0].url).toContain("?limit=1000")
  })

  it("says when the lists stop short of every key", async () => {
    mockApi(listing([key({ is_effective: true })], { count: 1500 }))
    renderWith(
      <WorkspaceWebSearchKeyChoice workspaceId={WORKSPACE} canChoose />,
    )

    expect(
      await screen.findByText(
        "These lists show the first 1 of your organization's 1,500 keys.",
      ),
    ).toBeInTheDocument()
  })

  it("follows the organization when its default is chosen, so no pin is left", async () => {
    const user = userEvent.setup()
    const calls = mockApi(
      listing([
        key({ is_default: true, is_effective: true }),
        key({
          org_web_search_key_id: MOJEEK,
          provider: "mojeek",
          name: "M",
          is_org_default: true,
        }),
      ]),
    )
    renderWith(
      <WorkspaceWebSearchKeyChoice workspaceId={WORKSPACE} canChoose />,
    )

    await user.click(await screen.findByRole("radio", { name: /mojeek · M/ }))

    await waitFor(() => expect(writes(calls, "DELETE")).toHaveLength(1))
    expect(writes(calls, "DELETE")[0].url).toContain(AAA)
    expect(writes(calls, "PATCH")).toHaveLength(0)
  })

  it("chooses the deployment's tool by unpinning when the organization has no default", async () => {
    const user = userEvent.setup()
    const calls = mockApi(
      listing([key({ is_default: true, is_effective: true })]),
    )
    renderWith(
      <WorkspaceWebSearchKeyChoice workspaceId={WORKSPACE} canChoose />,
    )

    // With no organization default there is none to go back to, so no button.
    expect(
      await screen.findByRole("radio", {
        name: /This deployment's default tool/,
      }),
    ).toBeEnabled()
    expect(
      screen.queryByRole("button", { name: /Use the organization/ }),
    ).toBeNull()
    await user.click(
      screen.getByRole("radio", { name: /This deployment's default tool/ }),
    )

    await waitFor(() => expect(writes(calls, "DELETE")).toHaveLength(1))
    expect(writes(calls, "DELETE")[0].url).toContain(AAA)
  })

  it("says nothing searches when neither a key nor the deployment serves the workspace", async () => {
    mockApi(
      listing([], { search_provider: null, deployment_search_tool: null }),
    )
    renderWith(
      <WorkspaceWebSearchKeyChoice workspaceId={WORKSPACE} canChoose />,
    )

    expect(
      await screen.findByText(
        "This deployment has no search tool, so nothing searches for this workspace.",
      ),
    ).toBeInTheDocument()
  })

  it("shows a member the choice without letting them change it", async () => {
    mockApi(listing([key({ is_effective: true, is_org_default: true })]))
    renderWith(
      <WorkspaceWebSearchKeyChoice workspaceId={WORKSPACE} canChoose={false} />,
    )

    expect(
      await screen.findByRole("radio", { name: /exa · AAA/ }),
    ).toBeDisabled()
    expect(
      screen.queryByRole("button", { name: /Use the organization/ }),
    ).toBeNull()
  })
})

describe("WorkspaceProviderOptions", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("saves the options a workspace may set for the provider its searches run on", async () => {
    const user = userEvent.setup()
    mockApi(listing([]))
    const save = vi.fn().mockResolvedValue(undefined)
    renderWith(
      <WorkspaceProviderOptions
        workspaceId={WORKSPACE}
        stored={null}
        disabled={false}
        save={save}
      />,
    )

    expect(await screen.findByText("exa options")).toBeInTheDocument()
    // An operator-only option is not one a workspace may set.
    expect(screen.queryByText("internal")).toBeNull()
    await pickOption(user, "type", "fast")
    await user.click(screen.getByRole("button", { name: "Save options" }))

    await waitFor(() => expect(save).toHaveBeenCalledWith({ type: "fast" }))
  })
})
