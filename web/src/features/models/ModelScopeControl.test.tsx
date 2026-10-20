import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ModelScopeControl } from "@/features/models/ModelScopeControl"
import { API_ROOT } from "@/shared/api/client"
import { organizationContext } from "@/tests/fixtures"

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

// An organization admin on a hosted deployment: not the deployment operator, so
// the operator-only discovery and provider-config reads refuse, and GET /models
// is the read that answers with the organization's own providers and the
// deployment's managed models (otari-ai#2392). `operator` flips the caller to
// the deployment operator, who may read the configured providers.
function mockApi({
  models = [] as string[],
  modelsStatus = 200,
  operator = false,
  providers = [] as string[],
  hosted = [] as string[],
  aliases = [] as { name: string; target: string }[],
} = {}) {
  const urls: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    urls.push(url)
    if (url.includes(`${API_ROOT}/organizations/me/aliases`)) {
      return json(aliases)
    }
    if (url.endsWith(`${API_ROOT}/organizations/me`)) {
      return json(organizationContext({ deployment_operator: operator }))
    }
    if (operator && url.endsWith(`${API_ROOT}/aliases`)) {
      return json([])
    }
    if (operator && url.endsWith(`${API_ROOT}/providers`)) {
      return json({ providers: providers.map((instance) => ({ instance })) })
    }
    if (operator && url.includes(`${API_ROOT}/hosted-providers?`)) {
      return json({
        data: hosted.map((name) => ({ name, provider: "openai" })),
        count: hosted.length,
      })
    }
    if (url.endsWith(`${API_ROOT}/models`)) {
      if (modelsStatus !== 200) return json({ detail: "nope" }, modelsStatus)
      return json({
        object: "list",
        data: models.map((id) => ({ id, object: "model", owned_by: "x" })),
      })
    }
    return json({ detail: "Forbidden" }, 403)
  })
  return urls
}

function renderControl() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const onChange = vi.fn()
  render(
    <QueryClientProvider client={client}>
      <ModelScopeControl initial={["byok-openai:gpt-4o"]} onChange={onChange} />
    </QueryClientProvider>,
  )
  return onChange
}

describe("ModelScopeControl", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("offers the organization's own and managed models to a non-operator", async () => {
    const urls = mockApi({
      models: [
        "byok-openai:gpt-4o",
        "byok-openai:gpt-4o-mini",
        "byok-anthropic:claude-haiku-4-5",
        "mzai:llama-4",
        // A bare name resolves through no instance, so it is no selector.
        "fast",
      ],
      aliases: [{ name: "fast", target: "byok-openai:gpt-4o-mini" }],
    })
    const user = userEvent.setup()
    const onChange = renderControl()

    // Every entry here has an "a" in it, so the filter keeps them all.
    await user.type(screen.getByLabelText("Add a model"), "a")
    // The stored entry is not offered again.
    expect(
      screen.queryByRole("option", { name: "byok-openai:gpt-4o" }),
    ).not.toBeInTheDocument()
    for (const name of [
      "byok-anthropic:* · all byok-anthropic models",
      "byok-openai:* · all byok-openai models",
      "mzai:* · all mzai models",
      "byok-anthropic:claude-haiku-4-5",
      "byok-openai:gpt-4o-mini",
      "mzai:llama-4",
    ]) {
      expect(await screen.findByRole("option", { name })).toBeInTheDocument()
    }
    expect(
      screen.queryByRole("option", { name: "fast" }),
    ).not.toBeInTheDocument()
    // The alias resolves to a model already listed, which keeps its own name.
    expect(
      screen.queryByRole("option", { name: "fast · alias" }),
    ).not.toBeInTheDocument()

    await user.click(screen.getByRole("option", { name: "mzai:llama-4" }))
    expect(onChange).toHaveBeenLastCalledWith(
      ["byok-openai:gpt-4o", "mzai:llama-4"],
      true,
    )
    expect(urls.some((url) => url.includes("/models/discoverable"))).toBe(false)
    expect(urls.some((url) => url.endsWith(`${API_ROOT}/providers`))).toBe(
      false,
    )
  })

  it("offers an operator a wildcard for a configured provider /models cannot list", async () => {
    const urls = mockApi({
      operator: true,
      models: ["byok-openai:gpt-4o-mini"],
      // Discovery unsupported and nothing priced: no model of its own to list.
      providers: ["byok-openai", "local-vllm"],
    })
    const user = userEvent.setup()
    renderControl()

    await user.type(screen.getByLabelText("Add a model"), "l")
    expect(
      await screen.findByRole("option", {
        name: "local-vllm:* · all local-vllm models",
      }),
    ).toBeInTheDocument()
    expect(urls.some((url) => url.endsWith(`${API_ROOT}/providers`))).toBe(true)
  })

  it("offers an operator a wildcard for a hosted provider /models cannot list", async () => {
    mockApi({
      operator: true,
      models: ["byok-openai:gpt-4o-mini"],
      hosted: ["my-vllm"],
    })
    const user = userEvent.setup()
    renderControl()

    await user.type(screen.getByLabelText("Add a model"), "m")
    expect(
      await screen.findByRole("option", {
        name: "my-vllm:* · all my-vllm models",
      }),
    ).toBeInTheDocument()
  })

  it("says nothing is configured only when the catalog is really empty", async () => {
    mockApi()
    renderControl()

    expect(
      await screen.findByText(/No providers or models discovered yet/),
    ).toBeInTheDocument()
  })

  it("reports a failed catalog read rather than an empty one", async () => {
    mockApi({ modelsStatus: 500 })
    renderControl()

    expect(
      await screen.findByText(/Could not load the model catalog/),
    ).toBeInTheDocument()
    expect(
      screen.queryByText(/No providers or models discovered yet/),
    ).not.toBeInTheDocument()
  })

  it("still says the read failed when aliases fill the list", async () => {
    mockApi({
      modelsStatus: 500,
      aliases: [{ name: "fast", target: "byok-openai:gpt-4o-mini" }],
    })
    const user = userEvent.setup()
    renderControl()

    expect(
      await screen.findByText(/Could not load the model catalog/),
    ).toBeInTheDocument()
    // What did load stays pickable.
    await user.type(await screen.findByLabelText("Add a model"), "fast")
    expect(
      await screen.findByRole("option", { name: "fast · alias" }),
    ).toBeInTheDocument()
  })
})
