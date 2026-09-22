import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext, OrganizationPricingOverride } from "@/client"
import { PricingOverrideDialog } from "@/features/organization/PricingOverrideDialog"
import { API_ROOT } from "@/shared/api/client"
import { organizationContext } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

interface RecordedRequest {
  url: string
  method: string
  body: unknown
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(status === 204 ? null : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function pricingOverride(
  fields: Partial<OrganizationPricingOverride> = {},
): OrganizationPricingOverride {
  return {
    id: "11111111-1111-1111-1111-111111111111",
    organization_id: "org-1",
    model_key: "openai:gpt-4o",
    input_price_per_million: 2.5,
    output_price_per_million: 5,
    cache_read_price_per_million: null,
    cache_write_price_per_million: null,
    cache_write_1h_price_per_million: null,
    pricing_tiers: [],
    unit: "tokens",
    effective_from: "2026-08-01T00:00:00Z",
    effective_to: null,
    created_at: "2026-08-01T00:00:00Z",
    updated_at: "2026-08-01T00:00:00Z",
    ...fields,
  }
}

// Mocked at the `@/client` boundary (a real `fetch`), which is what the standards
// call for: the hooks and their invalidation are part of what is under test, so
// stubbing them would leave the interesting half uncovered.
function catalogModel(id: string, deployment_managed = false) {
  return {
    id,
    object: "model",
    created: 0,
    owned_by: id.split(":")[0],
    pricing_source: "none",
    deployment_managed,
  }
}

function mockApi({
  context = organizationContext(),
  overrides = [] as OrganizationPricingOverride[],
  writeStatus = 201,
  writeBody = pricingOverride() as unknown,
  models = [] as string[],
  managedModels = [] as string[],
}: {
  context?: OrganizationContext
  overrides?: OrganizationPricingOverride[]
  writeStatus?: number
  writeBody?: unknown
  /** What GET /v1/models serves, which is where the model-key picker looks. */
  models?: string[]
  /** Catalog entries the deployment supplies the credential for. */
  managedModels?: string[]
} = {}) {
  const requests: RecordedRequest[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })
    if (url.includes(`${API_ROOT}/organizations/me/pricing`)) {
      if (method === "GET") {
        // Honors the window, like the endpoint: a test that ignored it could
        // not tell a paged read from a read of everything.
        const params = new URL(url, "http://localhost").searchParams
        const skip = Number(params.get("skip") ?? 0)
        const limit = Number(params.get("limit") ?? 100)
        return jsonResponse({
          data: overrides.slice(skip, skip + limit),
          count: overrides.length,
        })
      }
      return jsonResponse(writeBody, writeStatus)
    }
    // The picker's own source, folded by model and narrowed by the server.
    if (url.includes(`${API_ROOT}/catalog/models`)) {
      const term = new URL(url, "http://localhost").searchParams.get("search")
      const selectors = [...models, ...managedModels].filter(
        (id) => !term || id.toLowerCase().includes(term.toLowerCase()),
      )
      return jsonResponse({
        default_pricing: false,
        defaults_as_of: null,
        metadata_available: true,
        count: selectors.length,
        models: selectors.map((id) => ({
          id,
          name: id,
          selectors: [id],
        })),
      })
    }
    if (url.endsWith(`${API_ROOT}/models`)) {
      return jsonResponse({
        object: "list",
        data: [
          ...models.map((id) => catalogModel(id)),
          ...managedModels.map((id) => catalogModel(id, true)),
        ],
      })
    }
    return jsonResponse(context)
  })
  return requests
}

// The real router, per the frontend standards: the dialog reads the catalog and
// the caller's standing, and both travel through hooks the app mounts a router
// around. Awaited, because the router resolves its first location asynchronously
// and a synchronous DOM read would race it.
//
// Rendered open, because this file is about the form rather than about whoever
// opens it: the merged Providers page owns that, and its own test covers it.
function renderDialog(
  existing: OrganizationPricingOverride[] = [],
  props: {
    editing?: OrganizationPricingOverride
    initialModelKey?: string
    onSaved?: () => void
  } = {},
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <QueryClientProvider client={client}>
      <PricingOverrideDialog
        isOpen
        onOpenChange={vi.fn()}
        existing={existing}
        onSaved={props.onSaved ?? vi.fn()}
        editing={props.editing}
        initialModelKey={props.initialModelKey}
      />
    </QueryClientProvider>,
    { url: "/organization/provider-keys" },
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("PricingOverrideDialog", () => {
  it("sends a create with the rates that were typed", async () => {
    const requests = mockApi({ overrides: [] })
    const user = userEvent.setup()

    await renderDialog()

    await user.type(
      await screen.findByRole("combobox", { name: /model key/i }),
      "anthropic:claude-sonnet-5",
    )
    await user.type(screen.getByLabelText(/input, per 1m tokens/i), "3")
    await user.type(screen.getByLabelText(/output, per 1m tokens/i), "15")
    await user.click(screen.getByRole("button", { name: /^add override$/i }))

    await waitFor(() => {
      const write = requests.find(
        (request) =>
          request.method === "POST" &&
          request.url.includes(`${API_ROOT}/organizations/me/pricing`),
      )
      expect(write?.body).toMatchObject({
        model_key: "anthropic:claude-sonnet-5",
        input_price_per_million: 3,
        output_price_per_million: 15,
      })
    })
  })

  it("fills the model key from the catalog, so a rate is not stored under a typo", async () => {
    // The catalog rather than /v1/models/discoverable: this form answers to an
    // organization admin, who is refused the deployment-operator read.
    const requests = mockApi({
      overrides: [],
      models: ["anthropic:claude-sonnet-5"],
    })
    const user = userEvent.setup()

    await renderDialog()

    // The trigger, not the input: the field opens on typing, so that an
    // autofocused list does not hide the rest of the form from a screen reader.
    await user.click(
      await screen.findByRole("button", { name: /show suggestions/i }),
    )
    await user.click(
      await screen.findByRole("option", { name: "anthropic:claude-sonnet-5" }),
    )
    await user.type(screen.getByLabelText(/input, per 1m tokens/i), "3")
    await user.type(screen.getByLabelText(/output, per 1m tokens/i), "15")
    await user.click(screen.getByRole("button", { name: /^add override$/i }))

    await waitFor(() => {
      const write = requests.find(
        (request) =>
          request.method === "POST" &&
          request.url.includes(`${API_ROOT}/organizations/me/pricing`),
      )
      expect(write?.body).toMatchObject({
        model_key: "anthropic:claude-sonnet-5",
      })
    })
  })

  it("opens on the selector it was handed, so a link lands on the right model", async () => {
    // The Models detail page links with `?override=<selector>`, and the merged
    // Providers page passes that through as `initialModelKey`.
    mockApi({ overrides: [] })

    await renderDialog([], { initialModelKey: "openai:gpt-4o" })

    expect(
      await screen.findByRole("combobox", { name: /model key/i }),
    ).toHaveValue("openai:gpt-4o")
  })

  it("refuses a model key with no provider prefix before sending it", async () => {
    // A rate stored under a bare name bills nothing: resolution builds
    // `provider:model` from the request and would never match it.
    const requests = mockApi({ overrides: [] })
    const user = userEvent.setup()

    await renderDialog()

    await user.type(
      await screen.findByRole("combobox", { name: /model key/i }),
      "gpt-4o",
    )
    await user.type(screen.getByLabelText(/input, per 1m tokens/i), "3")
    await user.type(screen.getByLabelText(/output, per 1m tokens/i), "15")

    expect(screen.getByText(/needs the provider prefix/i)).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /^add override$/i }),
    ).toBeDisabled()
    expect(requests.some((request) => request.method === "POST")).toBe(false)
  })

  it("refuses a period that overlaps a stored override before sending it", async () => {
    // The server refuses this with a 409. Saying so before the request is sent
    // means the admin sees which period they collided with, not just a failure.
    mockApi({ overrides: [] })
    const user = userEvent.setup()

    await renderDialog([
      pricingOverride({
        effective_from: "2020-01-01T00:00:00Z",
        effective_to: null,
      }),
    ])

    await user.type(
      await screen.findByRole("combobox", { name: /model key/i }),
      "openai:gpt-4o",
    )
    await user.type(screen.getByLabelText(/input, per 1m tokens/i), "3")
    await user.type(screen.getByLabelText(/output, per 1m tokens/i), "15")

    expect(
      screen.getByText(/overlaps an override already stored/i),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /^add override$/i }),
    ).toBeDisabled()
  })

  it("sends a replace without the model key when an override is edited", async () => {
    // The endpoint refuses to repoint an override at another model, so the
    // update body carries no `model_key` at all.
    const stored = pricingOverride()
    const requests = mockApi({ overrides: [stored], writeStatus: 200 })
    const user = userEvent.setup()

    await renderDialog([stored], { editing: stored })

    const input = await screen.findByLabelText(/input, per 1m tokens/i)
    await user.clear(input)
    await user.type(input, "4")
    await user.click(screen.getByRole("button", { name: /save override/i }))

    await waitFor(() => {
      const write = requests.find((request) => request.method === "PUT")
      expect(write?.body).toMatchObject({ input_price_per_million: 4 })
      expect(write?.body).not.toHaveProperty("model_key")
    })
  })
  it("shows a refused save rather than closing over it", async () => {
    // `RateOverridesCard` asserted this before the merge and took the assertion
    // with it. The mutation lives inside this dialog, so a refusal has nowhere
    // else to surface.
    mockApi({
      overrides: [],
      writeStatus: 409,
      writeBody: { detail: "A period already covers that instant" },
    })
    const user = userEvent.setup()

    // Handed the selector, so the combobox never opens: this is about what a
    // refusal does, not about picking a model.
    await renderDialog([], { initialModelKey: "openai:gpt-4o" })

    await user.type(await screen.findByLabelText(/input, per 1m tokens/i), "1")
    await user.type(screen.getByLabelText(/output, per 1m tokens/i), "2")
    await user.click(screen.getByRole("button", { name: /^add override$/i }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "A period already covers that instant",
    )
    expect(screen.getByRole("dialog")).toBeInTheDocument()
  })

  it("puts focus in the first rate when editing, not on the frame's close", async () => {
    // Editing a rate is a numeric change to a form already filled in, so the
    // caret belongs where the change goes. The model key is fixed for an edit.
    mockApi({ overrides: [] })
    const stored = pricingOverride()

    await renderDialog([stored], { editing: stored })

    await waitFor(() =>
      expect(screen.getByLabelText(/input, per 1m tokens/i)).toHaveFocus(),
    )
  })
  it("blocks an edit whose start has been cleared", async () => {
    // A period with no start resolves for nothing, so the form refuses it here
    // rather than letting the server answer.
    const requests = mockApi({ overrides: [] })
    const stored = pricingOverride()
    const user = userEvent.setup()

    await renderDialog([stored], { editing: stored })

    await user.clear(await screen.findByLabelText(/applies from/i))

    expect(screen.getByText(/an edit needs a start/i)).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /save override/i }),
    ).toBeDisabled()
    expect(requests.some((request) => request.method === "PUT")).toBe(false)
  })

  it("does not greet the next open with the last attempt's refusal", async () => {
    // The mutation lives below the caller's key, so remounting the dialog on a
    // different model drops it. Asserted as two renders, which is what the page
    // does when `?override=` changes.
    mockApi({
      overrides: [],
      writeStatus: 409,
      writeBody: { detail: "A period already covers that instant" },
    })
    const user = userEvent.setup()

    const first = await renderDialog([], { initialModelKey: "openai:gpt-4o" })
    await user.type(await screen.findByLabelText(/input, per 1m tokens/i), "1")
    await user.type(screen.getByLabelText(/output, per 1m tokens/i), "2")
    await user.click(screen.getByRole("button", { name: /^add override$/i }))
    expect(await screen.findByRole("alert")).toBeInTheDocument()
    first.unmount()

    await renderDialog([], { initialModelKey: "openai:gpt-4o-mini" })

    expect(
      await screen.findByLabelText(/input, per 1m tokens/i),
    ).toBeInTheDocument()
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("seeds each opener's dialog fresh, whichever one was used last", async () => {
    // Keyed on the model in the page above, so a second opener starts from that
    // model's stored rate and not from whatever the last one left typed.
    mockApi({ overrides: [] })
    const stored = pricingOverride()
    const user = userEvent.setup()

    const first = await renderDialog([], { initialModelKey: "openai:gpt-4o" })
    await user.type(await screen.findByLabelText(/input, per 1m tokens/i), "9")
    first.unmount()

    await renderDialog([stored], { editing: stored })

    expect(await screen.findByLabelText(/input, per 1m tokens/i)).toHaveValue(
      String(stored.input_price_per_million),
    )
  })
})
