import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  OrganizationContext,
  OrganizationPricingOverride,
  OrgProviderKey,
  OrgProviderModel,
} from "@/client"
import { OrganizationProvidersPage } from "@/features/organization/providers/OrganizationProvidersPage"
import { API_ROOT } from "@/shared/api/client"
import {
  organizationContext,
  organizationPricingOverride,
  orgProviderKey,
  orgProviderModel,
} from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

const KEY_ID = "66666666-6666-6666-6666-666666666666"

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

interface MockOpts {
  keys?: OrgProviderKey[]
  context?: OrganizationContext
  catalog?: { id: string; name: string }[]
  // Refuse the create, so a test can read what a refusal leaves behind.
  createFails?: boolean
  // Refuse the catalog read, which is what a deployment that still gates it on
  // the operator does to this page's audience.
  catalogFails?: boolean
  // The models offered on whichever key a test expands.
  models?: OrgProviderModel[]
  // The organization's own stored rates, which the editor seeds from.
  overrides?: OrganizationPricingOverride[]
}

function mockApi(opts: MockOpts = {}) {
  const keys = opts.keys ?? []
  const models = opts.models ?? []
  const requests: Request[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })

    // Ahead of the key routes, because every one of these is nested under a key
    // and `/provider-keys` matches their URLs too.
    if (url.includes("/models") || url.includes("/available-models")) {
      return jsonResponse({ count: models.length, data: models, models: [] })
    }
    if (url.includes("/provider-keys")) {
      if (method === "GET") {
        return jsonResponse({ count: keys.length, data: keys })
      }
      if (method === "POST" && opts.createFails) {
        return jsonResponse({ detail: "Provider key already exists" }, 409)
      }
      // Every write answers with a key-shaped body the page only re-reads
      // through the invalidated list, so one row is enough for all of them.
      return jsonResponse(keys[0] ?? orgProviderKey())
    }
    // What the deployment actually answers this page's audience: /api/v1/settings
    // is operator-only and an organization owner is not one. Mocked as the
    // refusal rather than as a body, so a page that went back to reading it
    // would fail here rather than pass on a fixture no tenant ever receives.
    if (url.includes(`${API_ROOT}/settings`)) {
      return jsonResponse({ detail: "Not authorized" }, 403)
    }
    if (url.includes(`${API_ROOT}/providers/catalog`)) {
      if (opts.catalogFails) {
        return jsonResponse({ detail: "Not authorized" }, 403)
      }
      return jsonResponse(
        opts.catalog ?? [{ id: "anthropic", name: "Anthropic" }],
      )
    }
    // Ahead of the context catch-all, because that path is a prefix of this one.
    if (url.includes(`${API_ROOT}/organizations/me/pricing`)) {
      // Narrowed the way the server narrows it, so a test can open the editor on
      // a second model and see what that model's own read answers.
      const asked = new URL(url, "http://x").searchParams.get("model_key")
      const overrides = (opts.overrides ?? []).filter(
        (row) => asked === null || row.model_key === asked,
      )
      return jsonResponse({ count: overrides.length, data: overrides })
    }
    return jsonResponse(opts.context ?? organizationContext())
  })
  return requests
}

/**
 * Answer the organization's own rate list, which the page reads to seed its rate
 * editor and to refuse an overlapping period before sending one.
 *
 * Answered explicitly rather than left to a catch-all, so a test about one of
 * this page's *other* reads failing is not also a test about this one failing:
 * a second error banner would make an assertion on "the alert" ambiguous, and
 * whether that race lands depends on which query settles first.
 */
function organizationRatesAnswer(url: string): Response | undefined {
  if (url.includes(`${API_ROOT}/organizations/me/pricing`)) {
    return jsonResponse({ count: 0, data: [] })
  }
  return undefined
}

// A live router, not a stub: the page keeps which provider's models are open and
// which model's rate is being edited in the URL, so `useUrlState` needs a real
// location to read. `renderWithRouter` awaits the router's first resolution,
// which is why every caller is awaited.
function renderPage(ui: ReactElement, url = "/organization/provider-keys") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
    { url },
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("OrganizationProvidersPage", () => {
  it("keeps the page's add action visible while the dialog is open", async () => {
    // It used to hide while the form was a band on the page. The form is over
    // the page now, and the trigger is where focus returns when it closes.
    mockApi({})
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    const trigger = await screen.findByRole("button", {
      name: "Add provider key",
    })
    await user.click(trigger)
    await screen.findByRole("dialog", { name: "New provider key" })
    expect(trigger).toBeInTheDocument()
  })

  it("marks a key the deployment cannot decrypt and says how to fix it", async () => {
    // Every other signal on the row reads normal: the key is stored, it has a
    // last4, nothing disabled it. What is wrong is that the ciphertext no longer
    // decrypts, so it serves nothing and its models leave the catalog. Without
    // this the page is indistinguishable from a working one.
    mockApi({ keys: [orgProviderKey({ name: "Production", usable: false })] })
    await renderPage(<OrganizationProvidersPage />)

    expect(await screen.findByText("UNREADABLE")).toBeInTheDocument()
    expect(
      screen.getByText(/can't be decrypted on this deployment/i),
    ).toBeInTheDocument()
    expect(screen.getByText(/Re-enter the API key/i)).toBeInTheDocument()
  })

  it("says nothing about decryption when every key is readable", async () => {
    mockApi({ keys: [orgProviderKey({ name: "Production" })] })
    await renderPage(<OrganizationProvidersPage />)

    await screen.findByText("Production")
    expect(screen.queryByText("UNREADABLE")).not.toBeInTheDocument()
    expect(
      screen.queryByText(/can't be decrypted on this deployment/i),
    ).not.toBeInTheDocument()
  })

  it("says the provider catalog failed rather than that there are no providers", async () => {
    // A refused read leaves the query with no data, which is the same empty
    // array a deployment with no providers would give. Reported as "No provider
    // to offer here.", it read as a deployment with nothing to offer, and the
    // real cause (the catalog sat behind the deployment-operator gate, which
    // this page's owners and admins do not pass) was invisible from the form.
    mockApi({ catalogFails: true })
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider key" }),
    )
    const dialog = await screen.findByRole("dialog", {
      name: "New provider key",
    })
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))

    expect(
      await screen.findByText(/catalog could not be loaded/i),
    ).toBeInTheDocument()
    expect(
      screen.queryByText("No provider to offer here."),
    ).not.toBeInTheDocument()
  })

  it("opens on a blank draft after a create", async () => {
    // Nothing unmounts this form, so the remount on the way in is the only
    // thing that clears it, and what it holds includes the plaintext secret.
    // A surviving draft also reports itself dirty against the empty snapshot
    // `seeded` still holds, so the guard arms before anything is typed.
    mockApi()
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider key" }),
    )
    await user.click(screen.getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(screen.getByRole("textbox", { name: /Name/ }), "Production")
    await user.type(screen.getByLabelText("API key"), "sk-secret")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add provider key",
      }),
    )
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "New provider key" }),
      ).toBeNull(),
    )

    await user.click(screen.getByRole("button", { name: "Add provider key" }))
    const reopened = await screen.findByRole("dialog", {
      name: "New provider key",
    })
    expect(within(reopened).getByRole("textbox", { name: /Name/ })).toHaveValue(
      "",
    )
    expect(within(reopened).getByLabelText("API key")).toHaveValue("")
    expect(
      within(reopened).getByRole("combobox", { name: "Provider" }),
    ).toHaveValue("")
    // And not dirty on arrival: Escape closes it rather than arming the guard.
    await user.keyboard("{Escape}")
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "New provider key" }),
      ).toBeNull(),
    )
  })

  it("does not carry a refused create's banner into the next open", async () => {
    mockApi({ createFails: true })
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider key" }),
    )
    await user.click(screen.getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(screen.getByRole("textbox", { name: /Name/ }), "Production")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add provider key",
      }),
    )
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Provider key already exists",
    )

    // Out through the guard, which is the only way out of a dirty form.
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Discard" }))
    await user.click(screen.getByRole("button", { name: "Add provider key" }))

    const reopened = await screen.findByRole("dialog", {
      name: "New provider key",
    })
    expect(within(reopened).queryByRole("alert")).toBeNull()
  })

  it("lists the organization's keys with the default marked", async () => {
    mockApi({
      keys: [
        orgProviderKey({ name: "Production", is_org_default: true }),
        orgProviderKey({
          id: "77777777-7777-7777-7777-777777777777",
          name: "Staging",
          provider: "anthropic",
          api_base: "https://proxy.example.com/v1",
        }),
      ],
    })
    await renderPage(<OrganizationProvidersPage />)

    expect(await screen.findByText("Production")).toBeInTheDocument()
    expect(screen.getByText("DEFAULT")).toBeInTheDocument()
    expect(screen.getByText("Staging")).toBeInTheDocument()
    // The column shows the vendor's name; the row's provider is still the id.
    expect(screen.getByText("Anthropic")).toBeInTheDocument()
    expect(screen.getByText("https://proxy.example.com/v1")).toBeInTheDocument()
    // The credential itself never comes back, so the page can only ever show
    // the tail the API publishes.
    expect(screen.getAllByText("••••abcd").length).toBeGreaterThan(0)
  })

  it("reads the organization's own keys, not the deployment's credentials", async () => {
    // The whole reason this page exists: /api/v1/provider-credentials is keyed on
    // an instance name and belongs to the process, so a page that read it would
    // be showing every tenant the same rows.
    const requests = mockApi({ keys: [orgProviderKey()] })
    await renderPage(<OrganizationProvidersPage />)

    await screen.findByText("Production")
    expect(
      requests.some((request) =>
        request.url.includes(`${API_ROOT}/organizations/me/provider-keys`),
      ),
    ).toBe(true)
    expect(
      requests.some((request) =>
        request.url.includes(`${API_ROOT}/provider-credentials`),
      ),
    ).toBe(false)
  })

  it("creates a key for the provider that was picked", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider key" }),
    )
    await user.click(screen.getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(screen.getByRole("textbox", { name: /Name/ }), "Production")
    await user.type(screen.getByLabelText("API key"), "sk-secret")
    await user.click(
      screen.getByRole("button", { name: "Add provider key", hidden: false }),
    )

    await waitFor(() => {
      expect(
        requests.some(
          (request) =>
            request.method === "POST" &&
            request.url.endsWith(`${API_ROOT}/organizations/me/provider-keys`),
        ),
      ).toBe(true)
    })
    const post = requests.find(
      (request) =>
        request.method === "POST" &&
        request.url.endsWith(`${API_ROOT}/organizations/me/provider-keys`),
    )
    expect(post?.body).toMatchObject({
      provider: "anthropic",
      name: "Production",
      api_key: "sk-secret",
    })
  })

  it("asks Bedrock for its region by name and sends it in client_args", async () => {
    const requests = mockApi({ catalog: [{ id: "bedrock", name: "Bedrock" }] })
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider key" }),
    )
    await user.click(screen.getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Bedrock" }))
    await user.type(screen.getByRole("textbox", { name: /Name/ }), "Prod")
    // Not called an API key here: Bedrock's credential is a bearer token or an
    // IAM secret, depending on which shape the organization uses.
    await user.type(screen.getByLabelText(/Bedrock API key/), "bearer-token")
    await user.type(
      screen.getByRole("textbox", { name: /AWS region/ }),
      "us-east-1",
    )
    await user.click(
      screen.getByRole("button", { name: "Add provider key", hidden: false }),
    )

    await waitFor(() => {
      expect(requests.some((request) => request.method === "POST")).toBe(true)
    })
    expect(
      requests.find((request) => request.method === "POST")?.body,
    ).toMatchObject({
      provider: "bedrock",
      api_key: "bearer-token",
      client_args: { region_name: "us-east-1" },
    })
  })

  it("will not add a Bedrock key until the region is there and looks like one", async () => {
    const requests = mockApi({ catalog: [{ id: "bedrock", name: "Bedrock" }] })
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider key" }),
    )
    await user.click(screen.getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Bedrock" }))
    await user.type(screen.getByRole("textbox", { name: /Name/ }), "Prod")

    const submit = screen.getByRole("button", {
      name: "Add provider key",
      hidden: false,
    })
    expect(submit).toBeDisabled()
    expect(
      screen.getByText("AWS region is required for this provider."),
    ).toBeInTheDocument()

    const region = screen.getByRole("textbox", { name: /AWS region/ })
    await user.type(region, "US East 1")
    expect(
      screen.getByText(
        "A region is lowercase letters, digits and hyphens, like us-east-1.",
      ),
    ).toBeInTheDocument()
    expect(submit).toBeDisabled()

    await user.clear(region)
    await user.type(region, "us-east-1")
    await waitFor(() => expect(submit).toBeEnabled())
    expect(requests.some((request) => request.method === "POST")).toBe(false)
  })

  it("will not add a Bedrock key with half of an IAM key pair", async () => {
    const requests = mockApi({ catalog: [{ id: "bedrock", name: "Bedrock" }] })
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider key" }),
    )
    await user.click(screen.getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Bedrock" }))
    await user.type(screen.getByRole("textbox", { name: /Name/ }), "Prod")
    await user.type(
      screen.getByRole("textbox", { name: /AWS region/ }),
      "us-east-1",
    )

    const submit = screen.getByRole("button", {
      name: "Add provider key",
      hidden: false,
    })
    // The region alone is the bearer-token shape, which is complete.
    await waitFor(() => expect(submit).toBeEnabled())

    await user.type(
      screen.getByRole("textbox", { name: /AWS access key ID/ }),
      "AKIAIOSFODNN7EXAMPLE",
    )
    expect(
      screen.getByText(
        "AWS secret access key is required alongside AWS access key ID.",
      ),
    ).toBeInTheDocument()
    expect(submit).toBeDisabled()

    await user.type(screen.getByLabelText(/AWS secret access key/), "s3cret")
    await waitFor(() => expect(submit).toBeEnabled())
    expect(requests.some((request) => request.method === "POST")).toBe(false)
  })

  it("does not advise against the credential Bedrock's IAM shape requires", async () => {
    // The old copy said to keep secrets out of client_args, which is the only
    // supported place for Bedrock's aws_secret_access_key.
    mockApi()
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider key" }),
    )

    expect(screen.queryByText(/keep secrets out/)).not.toBeInTheDocument()
    expect(
      screen.getByText(
        /masked when read back, but nothing here is encrypted at rest/,
      ),
    ).toBeInTheDocument()
  })

  it("does not offer a provider whose stored credential can never authenticate", async () => {
    mockApi({
      catalog: [
        { id: "anthropic", name: "Anthropic" },
        { id: "sagemaker", name: "SageMaker" },
      ],
    })
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider key" }),
    )
    await user.click(screen.getByRole("combobox", { name: "Provider" }))

    expect(
      await screen.findByRole("option", { name: "Anthropic" }),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("option", { name: "SageMaker" }),
    ).not.toBeInTheDocument()
  })

  it("keeps a stored client_args credential the form was never shown", async () => {
    // The gateway masks a credential-shaped option on read, by key name, so
    // both halves of the IAM pair come back as the mask and neither has a value
    // to prefill with. Sending the mask back is what tells the gateway to keep
    // what it holds.
    const requests = mockApi({
      catalog: [{ id: "bedrock", name: "Bedrock" }],
      keys: [
        orgProviderKey({
          provider: "bedrock",
          client_args: {
            region_name: "us-east-1",
            aws_access_key_id: "***",
            aws_secret_access_key: "***",
            timeout: 1800,
          },
        }),
      ],
    })
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    await user.click(await screen.findByRole("button", { name: "Edit" }))
    expect(
      await screen.findByRole("textbox", { name: /AWS region/ }),
    ).toHaveValue("us-east-1")
    // Never prefilled with the mask: three characters in a control read as a
    // real value, whether or not the control is a password box.
    expect(screen.getByLabelText(/AWS secret access key/)).toHaveValue("")
    const accessKeyId = screen.getByRole("textbox", {
      name: /AWS access key ID/,
    })
    expect(accessKeyId).toHaveValue("")
    expect(
      screen.getAllByText(/Set already, and never shown again/),
    ).toHaveLength(2)
    // Only what has no typed field of its own is left in the JSON escape hatch.
    expect(
      screen.getByRole("textbox", { name: "Client options (JSON)" }),
    ).toHaveValue('{\n  "timeout": 1800\n}')

    // A stored half counts as filled in, so the pair does not read as half done.
    expect(screen.queryByText(/is required alongside/)).not.toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(requests.some((request) => request.method === "PATCH")).toBe(true)
    })
    expect(
      requests.find((request) => request.method === "PATCH")?.body,
    ).toMatchObject({
      client_args: {
        region_name: "us-east-1",
        aws_access_key_id: "***",
        aws_secret_access_key: "***",
        timeout: 1800,
      },
    })
  })

  it("keeps the stored credential when an edit leaves the key box blank", async () => {
    // The box is never prefilled, because the gateway returns no plaintext to
    // prefill it with. Sending it as an explicit null would clear the key the
    // operator did not touch.
    const requests = mockApi({ keys: [orgProviderKey()] })
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    await user.click(await screen.findByRole("button", { name: "Edit" }))
    await user.click(await screen.findByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(requests.some((request) => request.method === "PATCH")).toBe(true)
    })
    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.body).not.toHaveProperty("api_key")
  })

  it("makes a key the organization default", async () => {
    const requests = mockApi({ keys: [orgProviderKey()] })
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Make default" }),
    )

    await waitFor(() => {
      expect(requests.some((request) => request.url.endsWith("/default"))).toBe(
        true,
      )
    })
  })

  it("hides archived keys until they are asked for, and offers delete only there", async () => {
    mockApi({
      keys: [
        orgProviderKey(),
        orgProviderKey({
          id: "88888888-8888-8888-8888-888888888888",
          name: "Retired",
          archived_at: "2026-08-20T00:00:00+00:00",
        }),
      ],
    })
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    await screen.findByText("Production")
    expect(screen.queryByText("Retired")).toBeNull()
    // Delete is permanent and the API accepts it for an archived key alone, so
    // it is never offered beside a live one.
    expect(screen.queryByRole("button", { name: "Delete" })).toBeNull()

    await user.click(screen.getByText("Show archived (1)"))

    expect(await screen.findByText("Retired")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Restore" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument()
  })

  it("deletes an archived key only through the confirm dialog", async () => {
    // otari-ai#2110. Archive keeps its two-click row confirm, because it is the
    // reversible step; the delete that follows it is the one behind a modal.
    const requests = mockApi({
      keys: [
        orgProviderKey({
          id: "88888888-8888-8888-8888-888888888888",
          name: "Retired",
          archived_at: "2026-08-20T00:00:00+00:00",
        }),
      ],
    })
    const user = userEvent.setup()
    await renderPage(<OrganizationProvidersPage />)

    await user.click(await screen.findByText("Show archived (1)"))
    await user.click(screen.getByRole("button", { name: "Delete" }))

    const dialog = await screen.findByRole("alertdialog")
    expect(within(dialog).getByText(/Retired and its stored/)).toBeVisible()
    expect(requests.some((request) => request.method === "DELETE")).toBe(false)

    await user.click(
      within(dialog).getByRole("button", { name: "Delete permanently" }),
    )

    await waitFor(() =>
      expect(
        requests.some(
          (request) =>
            request.method === "DELETE" &&
            request.url.includes("88888888-8888-8888-8888-888888888888"),
        ),
      ).toBe(true),
    )
  })

  it("withholds the keys and their read from a member who cannot manage the organization", async () => {
    // The list read is organization owner/admin-gated on the server
    // (otari-ai#1944), so a member reaching this URL is answered 403. The read
    // is not made, and the table goes with it: an empty one would say the
    // organization has no keys rather than that they cannot see them.
    const requests = mockApi({
      keys: [orgProviderKey()],
      // Both flags, not just the role: the fixture's default identity is a
      // deployment operator, and this page's gate is `canManage` alone, so
      // saying only the role would leave the case under test ambiguous.
      context: organizationContext({
        role: "member",
        deployment_operator: false,
      }),
    })
    await renderPage(<OrganizationProvidersPage />)

    expect(
      await screen.findByText(/Only organization owners and admins/),
    ).toBeInTheDocument()
    expect(screen.queryByText("Production")).toBeNull()
    // HeroUI's table is a `grid`, so this is the table itself being absent and
    // not merely empty of the row above.
    expect(
      screen.queryByRole("grid", { name: "Organization provider keys" }),
    ).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Add provider key" }),
    ).toBeNull()
    expect(screen.queryByRole("button", { name: "Edit" })).toBeNull()
    expect(
      requests.some((request) =>
        request.url.includes(`${API_ROOT}/organizations/me/provider-keys`),
      ),
    ).toBe(false)
  })

  it("disables adding a key when the deployment cannot encrypt one", async () => {
    // Same gate the deployment-wide providers page applies: without
    // OTARI_SECRET_KEY the write would fail at submit time.
    mockApi({
      context: organizationContext({
        provider_key_encryption_available: false,
      }),
    })
    await renderPage(<OrganizationProvidersPage />)

    expect(
      await screen.findByRole("button", { name: "Add provider key" }),
    ).toBeDisabled()
    expect(screen.getByText(/OTARI_SECRET_KEY/)).toBeInTheDocument()
  })

  it("keeps adding available for an owner the operator-only settings read refuses", async () => {
    // The bug this page shipped with (#839): the flag was inferred from
    // /api/v1/settings, which 403s for every organization owner, so the banner
    // reported a missing key on a deployment where the write path works.
    //
    // An owner who is *not* the deployment's operator, which is the whole case:
    // the deployment price bands below carry their own `/settings` read and an
    // operator legitimately makes it, so a context claiming both authorities
    // would prove nothing about where this page gets the encryption flag.
    const requests = mockApi({
      context: organizationContext({ deployment_operator: false }),
    })
    await renderPage(<OrganizationProvidersPage />)

    expect(
      await screen.findByRole("button", { name: "Add provider key" }),
    ).toBeEnabled()
    expect(screen.queryByText(/OTARI_SECRET_KEY/)).toBeNull()
    expect(
      requests.some((request) => request.url.includes(`${API_ROOT}/settings`)),
    ).toBe(false)
  })

  it("says nothing about the key when the context read is what failed", async () => {
    // The shape the original bug had, from the other direction: with no context
    // the encryption state is unknown, so the page must report the read that
    // failed rather than claim the key is missing. It behaves correctly today
    // only because `canManage(undefined)` is false and the banner is gated on
    // it, which is a role check standing in for an encryption one; this pins the
    // outcome so a later change to either gate cannot quietly restore the lie.
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      return (
        organizationRatesAnswer(url) ??
        (url.includes("/provider-keys")
          ? jsonResponse({ count: 0, data: [] })
          : jsonResponse({ detail: "Tenancy is unavailable" }, 500))
      )
    })
    await renderPage(<OrganizationProvidersPage />)

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Tenancy is unavailable",
    )
    expect(screen.queryByText(/OTARI_SECRET_KEY/)).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Add provider key" }),
    ).toBeNull()
    // Nor is the role claimed either way: with the context unread the page
    // cannot say the caller is a member, so the refusal banner stays off and the
    // table, whose read is gated on that same role, is not drawn empty.
    expect(screen.queryByText(/Only organization owners and admins/)).toBeNull()
    expect(
      screen.queryByRole("grid", { name: "Organization provider keys" }),
    ).toBeNull()
  })

  it("reports a list that could not be read instead of an empty table", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      return (
        organizationRatesAnswer(url) ??
        (url.includes("/provider-keys")
          ? jsonResponse({ detail: "Tenancy is unavailable" }, 500)
          : jsonResponse(organizationContext()))
      )
    })
    await renderPage(<OrganizationProvidersPage />)

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Tenancy is unavailable",
    )
  })

  it("opens the rate editor on the rate the model already has, not on a blank form", async () => {
    // The deep link is in the URL on the first render and the rates are not, so
    // a dialog opened before they land seeds itself from nothing and keeps that:
    // its fields read once, at mount. An admin following "Set your rate" from a
    // model page would then see an empty form over a rate that exists and
    // replace it by saving.
    mockApi({
      keys: [orgProviderKey({ id: KEY_ID, name: "Production" })],
      overrides: [organizationPricingOverride()],
    })

    await renderPage(
      <OrganizationProvidersPage />,
      "/organization/provider-keys?override=openai%3Agpt-4o",
    )

    const dialog = await screen.findByRole("dialog")
    expect(within(dialog).getByLabelText("Input, per 1M tokens")).toHaveValue(
      "2.5",
    )
    expect(within(dialog).getByLabelText("Output, per 1M tokens")).toHaveValue(
      "10",
    )
  })

  it("reseeds the rate editor when it is opened on a second model", async () => {
    // The rates query keeps the previous model's rows as placeholder data while
    // the next model's read is in flight, so "the data has arrived" is true and
    // wrong: the dialog remounts on the new model and seeds from the old one's
    // answer, which holds no row for it. Blank fields over a rate that exists,
    // the same failure the deep link had, reached by opening a second model.
    const user = userEvent.setup()
    mockApi({
      keys: [orgProviderKey({ id: KEY_ID, name: "Production" })],
      models: [
        orgProviderModel({ id: "m1", model: "gpt-4o" }),
        orgProviderModel({ id: "m2", model: "gpt-4o-mini" }),
      ],
      overrides: [
        organizationPricingOverride(),
        organizationPricingOverride({
          id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
          model_key: "openai:gpt-4o-mini",
          input_price_per_million: 0.15,
          output_price_per_million: 0.6,
        }),
      ],
    })

    await renderPage(
      <OrganizationProvidersPage />,
      `/organization/provider-keys?provider=${KEY_ID}&override=openai%3Agpt-4o`,
    )
    const first = await screen.findByRole("dialog")
    expect(within(first).getByLabelText("Input, per 1M tokens")).toHaveValue(
      "2.5",
    )
    await user.keyboard("{Escape}")
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())

    await user.click(
      await screen.findByRole("button", {
        name: "Set your rate for gpt-4o-mini",
      }),
    )

    const second = await screen.findByRole("dialog")
    await waitFor(() =>
      expect(within(second).getByLabelText("Input, per 1M tokens")).toHaveValue(
        "0.15",
      ),
    )
  })

  it("expands the provider named by the URL", async () => {
    mockApi({
      keys: [orgProviderKey({ id: KEY_ID, name: "Production" })],
      models: [orgProviderModel({ model: "gpt-4o" })],
    })

    await renderPage(
      <OrganizationProvidersPage />,
      `/organization/provider-keys?provider=${KEY_ID}`,
    )

    expect(
      await screen.findByRole("grid", { name: "Models on Production" }),
    ).toBeInTheDocument()
  })

  it("reports a rate list that could not be read instead of opening an empty editor", async () => {
    // The editor is held shut until the rates land, so a read that never lands
    // has to say so: silence plus no dialog reads as a link that did nothing.
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes(`${API_ROOT}/organizations/me/pricing`)) {
        return jsonResponse({ detail: "Pricing is unavailable" }, 500)
      }
      if (url.includes("/provider-keys")) {
        return jsonResponse({ count: 0, data: [] })
      }
      return jsonResponse(organizationContext())
    })

    await renderPage(
      <OrganizationProvidersPage />,
      "/organization/provider-keys?override=openai%3Agpt-4o",
    )

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Pricing is unavailable",
    )
    expect(screen.queryByRole("dialog")).toBeNull()
  })

  it("does not open the rate editor for a member who cannot manage the organization", async () => {
    // The deep link is a URL anybody can type. Rates decide what every member is
    // billed, so the editor answers to the same role the rest of the page does.
    mockApi({
      // Both flags, for the reason the keys test above gives.
      context: organizationContext({
        role: "member",
        deployment_operator: false,
      }),
      overrides: [organizationPricingOverride()],
    })

    await renderPage(
      <OrganizationProvidersPage />,
      "/organization/provider-keys?override=openai%3Agpt-4o",
    )

    expect(
      await screen.findByText(/Only organization owners and admins/),
    ).toBeInTheDocument()
    expect(screen.queryByRole("dialog")).toBeNull()
  })

  it("reads the models page from the URL, so an expanded panel is shareable to the row", async () => {
    // The page an expanded panel is on travels with the link, the way `provider`
    // and `override` already do. Snapped to an offered size, so a stale or
    // hand-edited `models_size` cannot reach the API as a limit it never offers.
    const requests = mockApi({
      keys: [orgProviderKey({ id: KEY_ID, name: "Production" })],
      models: [orgProviderModel({ model: "gpt-4o" })],
    })

    await renderPage(
      <OrganizationProvidersPage />,
      `/organization/provider-keys?provider=${KEY_ID}&models_page=2&models_size=999`,
    )
    await screen.findByRole("grid", { name: "Models on Production" })

    const read = requests.find((request) => request.url.includes("/models?"))
    expect(read?.url).toContain("skip=50")
    expect(read?.url).toContain("limit=25")
  })
})
