import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { StoredGuardrail } from "@/client"
import { LocalGuardrailDialog } from "@/features/tools/LocalGuardrailDialog"
import { API_ROOT } from "@/shared/api/client"
import {
  builtInGuardrail,
  organizationContext,
  storedGuardrail,
} from "@/tests/fixtures"
import { pickOption } from "@/tests/select"

const LAKERA = builtInGuardrail({
  create_parameters: [
    {
      name: "api_key",
      type: "string",
      required: true,
      secret: true,
      storable: true,
      description: "The Lakera API key.",
    },
    {
      name: "endpoint",
      type: "string",
      required: false,
      secret: false,
      storable: true,
    },
    // watsonx's shape: upstream takes a live client here, so nothing can be
    // written down and the gateway refuses a value for it.
    {
      name: "api_client",
      type: "json",
      required: false,
      secret: true,
      storable: false,
    },
  ],
})

// A local guardrail needs no credential at all, which is the case the gate has
// to leave alone.
const LOCAL = builtInGuardrail({
  guardrail_name: "prompt_guard_2",
  display_name: "Prompt Guard 2",
  vendor: "Meta",
  backend: "local_encoder",
  requires_api_key: false,
  create_parameters: [
    {
      name: "model_id",
      type: "string",
      required: false,
      secret: false,
      storable: true,
    },
  ],
})

const CATALOG = [LAKERA, LOCAL]

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi({ encryption = true }: { encryption?: boolean } = {}) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url.includes(`${API_ROOT}/organizations/me`)) {
      return jsonResponse(
        organizationContext({ provider_key_encryption_available: encryption }),
      )
    }
    return jsonResponse(storedGuardrail(), 201)
  })
}

function renderDialog(ui: ReactElement): ReturnType<typeof render> {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

function open(editing?: StoredGuardrail, taken: string[] = []) {
  return renderDialog(
    <LocalGuardrailDialog
      isOpen
      onClose={vi.fn()}
      guardrails={CATALOG}
      takenNames={taken}
      editing={editing}
    />,
  )
}

/** Everything a create or update actually put on the wire. */
function writeBody(spy: ReturnType<typeof mockApi>): Record<string, unknown> {
  const call = spy.mock.calls.find(
    ([url, init]) =>
      String(url).includes(`${API_ROOT}/guardrail-credentials`) &&
      init?.method !== undefined &&
      init.method !== "GET",
  )
  return JSON.parse(String(call?.[1]?.body)) as Record<string, unknown>
}

async function chooseLakera(user: ReturnType<typeof userEvent.setup>) {
  await pickOption(user, "What do you want checked?", "Prompt injection")
  await user.click(screen.getByRole("combobox", { name: /Which guardrail/ }))
  await user.click(await screen.findByRole("option", { name: /Lakera Guard/ }))
}

describe("LocalGuardrailDialog", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("shows no fields until a guardrail is chosen", async () => {
    mockApi()
    open()

    expect(screen.queryByLabelText("Name")).not.toBeInTheDocument()
    await chooseLakera(userEvent.setup())
    expect(await screen.findByLabelText("Name")).toBeInTheDocument()
    expect(screen.getByLabelText("Api key")).toBeInTheDocument()
  })

  it("suggests a name from the task, and stops once one is typed", async () => {
    mockApi()
    const user = userEvent.setup()
    open()
    await chooseLakera(user)

    const name = await screen.findByLabelText("Name")
    expect(name).toHaveValue("prompt-injection")
    await user.clear(name)
    await user.type(name, "lakera-in")
    expect(name).toHaveValue("lakera-in")
  })

  it("steps the suggestion past a name already in use", async () => {
    mockApi()
    open(undefined, ["prompt-injection"])
    await chooseLakera(userEvent.setup())

    expect(await screen.findByLabelText("Name")).toHaveValue(
      "prompt-injection-2",
    )
  })

  it("offers no value for an argument that cannot be stored, and sends none", async () => {
    const spy = mockApi()
    const user = userEvent.setup()
    open()
    await chooseLakera(user)

    const unstorable = await screen.findByLabelText("Api client")
    expect(unstorable).toBeDisabled()
    expect(
      screen.getByText(/Cannot be stored: this argument takes a live client/),
    ).toBeInTheDocument()

    await user.type(screen.getByLabelText("Api key"), "lakera-live-1")
    await user.click(screen.getByRole("button", { name: "Add guardrail" }))

    await waitFor(() => {
      expect(writeBody(spy)).toMatchObject({
        name: "prompt-injection",
        guardrail_name: "lakera_guard",
        create_kwargs: { api_key: "lakera-live-1" },
      })
    })
    expect(
      (writeBody(spy).create_kwargs as Record<string, unknown>).api_client,
    ).toBeUndefined()
  })

  it("refuses a guardrail that needs a credential when nothing can encrypt one", async () => {
    mockApi({ encryption: false })
    const user = userEvent.setup()
    open()
    await chooseLakera(user)

    expect(
      await screen.findByText(/no OTARI_SECRET_KEY set to encrypt one with/),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add guardrail" })).toBeDisabled()
  })

  it("still allows a guardrail that needs no credential", async () => {
    mockApi({ encryption: false })
    const user = userEvent.setup()
    open()
    await pickOption(user, "What do you want checked?", "Prompt injection")
    await user.click(screen.getByRole("combobox", { name: /Which guardrail/ }))
    await user.click(
      await screen.findByRole("option", { name: /Prompt Guard 2/ }),
    )

    expect(await screen.findByLabelText("Name")).toBeInTheDocument()
    expect(
      screen.queryByText(/no OTARI_SECRET_KEY set to encrypt one with/),
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Add guardrail" }),
    ).not.toBeDisabled()
  })

  it("will not change which guardrail an existing definition runs", async () => {
    mockApi()
    open(storedGuardrail())

    expect(await screen.findByText(/Lakera Guard/)).toBeInTheDocument()
    expect(
      screen.getByText(/Which guardrail runs cannot be changed here/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("combobox", { name: /Which guardrail/ }),
    ).not.toBeInTheDocument()
    expect(screen.getByLabelText("Name")).toBeDisabled()
  })

  it("says which secrets are already set, without showing one", async () => {
    mockApi()
    open(storedGuardrail())

    expect(
      await screen.findByText(
        /Set already, and never shown again. Leave blank to keep it./,
      ),
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Api key")).toHaveValue("")
  })

  it("keeps a stored secret when a save only changed something else", async () => {
    const spy = mockApi()
    const user = userEvent.setup()
    open(storedGuardrail())

    // The PATCH replaces create_kwargs, so a save that leaves the key out
    // deletes it. The mask is what says "keep the one you have".
    await user.type(
      await screen.findByLabelText("Endpoint"),
      "https://api.lakera.ai",
    )
    await user.click(screen.getByRole("button", { name: "Save guardrail" }))

    await waitFor(() => {
      expect(writeBody(spy)).toMatchObject({
        create_kwargs: {
          api_key: "***",
          endpoint: "https://api.lakera.ai",
        },
        expected_updated_at: storedGuardrail().updated_at,
      })
    })
  })

  it("rotates a secret when a new one is typed", async () => {
    const spy = mockApi()
    const user = userEvent.setup()
    open(storedGuardrail())

    await user.type(await screen.findByLabelText("Api key"), "lakera-live-2")
    await user.click(screen.getByRole("button", { name: "Save guardrail" }))

    await waitFor(() => {
      expect(
        (writeBody(spy).create_kwargs as Record<string, unknown>).api_key,
      ).toBe("lakera-live-2")
    })
  })

  it("reports a required argument the operator left blank", async () => {
    mockApi()
    const user = userEvent.setup()
    open()
    await chooseLakera(user)

    await user.click(
      await screen.findByRole("button", { name: "Add guardrail" }),
    )
    expect(
      await screen.findByText("This guardrail needs a value here."),
    ).toBeInTheDocument()
  })
})
