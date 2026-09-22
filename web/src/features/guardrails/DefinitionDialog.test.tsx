import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  BuiltInGuardrailCatalog,
  BuiltInGuardrailSpec,
  OrganizationGuardrailDefinition,
} from "@/client"
import { DefinitionDialog } from "@/features/guardrails/DefinitionDialog"
import { REDACTED_SECRET } from "@/shared/helpers/redaction"
import { pickOption, selectTrigger } from "@/tests/select"

function guardrail(
  overrides: Partial<BuiltInGuardrailSpec> &
    Pick<BuiltInGuardrailSpec, "guardrail_name" | "display_name">,
): BuiltInGuardrailSpec {
  return {
    backend: "hosted_api",
    categories: [],
    default_license: "",
    description: "",
    multilingual: false,
    multimodal: false,
    output_shapes: [],
    primary_category: "content_safety",
    requires_api_key: true,
    stages: [],
    supports_batch: false,
    ...overrides,
  } as BuiltInGuardrailSpec
}

// Shaped as the installed library reports these two, trimmed.
const CATALOG: BuiltInGuardrailCatalog = {
  guardrails: [
    guardrail({
      guardrail_name: "lakera_guard",
      display_name: "Lakera Guard",
      vendor: "Lakera",
      primary_category: "prompt_injection",
      categories: ["content_safety", "pii", "prompt_injection"],
      create_parameters: [
        {
          name: "api_key",
          type: "string",
          required: true,
          secret: true,
          storable: true,
        },
        {
          name: "endpoint",
          type: "string",
          required: false,
          secret: false,
          storable: true,
        },
      ],
    }),
    guardrail({
      guardrail_name: "bedrock_guardrails",
      display_name: "Bedrock Guardrails",
      vendor: "Amazon",
      categories: ["content_safety"],
      create_parameters: [
        {
          name: "guardrail_identifier",
          type: "string",
          required: true,
          secret: false,
          storable: true,
        },
        {
          name: "region_name",
          type: "string",
          required: false,
          secret: false,
          storable: true,
        },
        {
          name: "boto3_session",
          type: "json",
          required: false,
          secret: true,
          storable: false,
        },
      ],
    }),
    guardrail({
      guardrail_name: "any_llm",
      display_name: "Any LLM",
      vendor: "Mozilla AI",
      primary_category: "general_judge",
      categories: ["general_judge"],
    }),
  ],
}

function stored(
  overrides: Partial<OrganizationGuardrailDefinition> = {},
): OrganizationGuardrailDefinition {
  return {
    id: "d1",
    organization_id: "o1",
    name: "prod-lakera",
    guardrail_name: "lakera_guard",
    enabled: true,
    build_state: "built",
    create_kwargs: { endpoint: "https://api.lakera.ai" },
    create_secrets: { api_key: REDACTED_SECRET },
    secrets_decryptable: true,
    created_at: "2026-09-22T00:00:00Z",
    updated_at: "2026-09-22T00:00:00Z",
    ...overrides,
  }
}

function mockApi() {
  const calls: { url: string; method: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    calls.push({
      url: String(input),
      method: init?.method ?? "GET",
      body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
    })
    return Response.json(stored())
  })
  return calls
}

function renderDialog(
  props: Partial<Parameters<typeof DefinitionDialog>[0]> = {},
) {
  const onSaved = vi.fn()
  const onClose = vi.fn()
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <DefinitionDialog
        isOpen
        onClose={onClose}
        catalog={CATALOG}
        isCatalogPending={false}
        takenNames={[]}
        onSaved={onSaved}
        {...props}
      />
    </QueryClientProvider>,
  )
  return { onSaved, onClose }
}

function dialog() {
  return within(screen.getByRole("dialog"))
}

async function listedOptions(label: string) {
  const user = userEvent.setup()
  await user.click(selectTrigger(label))
  const options = (await screen.findAllByRole("option")).map(
    (option) => option.textContent,
  )
  await user.keyboard("{Escape}")
  return options
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("DefinitionDialog, setting one up", () => {
  it("asks for the check before the guardrail", () => {
    renderDialog()
    expect(selectTrigger("Which guardrail?")).toBeDisabled()
  })

  it("offers neither any_llm nor its general_judge check", async () => {
    renderDialog()
    expect(await listedOptions("What do you want checked?")).toEqual([
      "Content safety",
      "Personally identifiable information",
      "Prompt injection",
    ])
  })

  it("suggests a name, asks for the key, and posts the definition", async () => {
    const calls = mockApi()
    const { onSaved } = renderDialog({ takenNames: ["lakera-guard"] })
    const user = userEvent.setup()

    await pickOption(user, "What do you want checked?", "Prompt injection")
    await pickOption(user, "Which guardrail?", "Lakera Guard · Lakera")
    expect(dialog().getByLabelText("Name")).toHaveValue("lakera-guard-2")
    await user.type(dialog().getByLabelText("Api key"), "lk-secret")
    await user.click(dialog().getByRole("button", { name: "Set up guardrail" }))

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(calls.find((call) => call.method === "POST")?.body).toEqual({
      name: "lakera-guard-2",
      guardrail_name: "lakera_guard",
      create_kwargs: { api_key: "lk-secret" },
    })
  })

  it("refuses a save with the required key missing", async () => {
    const calls = mockApi()
    renderDialog()
    const user = userEvent.setup()

    await pickOption(user, "What do you want checked?", "Prompt injection")
    await pickOption(user, "Which guardrail?", "Lakera Guard · Lakera")
    await user.click(dialog().getByRole("button", { name: "Set up guardrail" }))

    expect(
      await screen.findByText("This guardrail needs a value here."),
    ).toBeInTheDocument()
    expect(calls).toEqual([])
  })

  it("draws no Advanced section when it would hold no field", async () => {
    // Nothing here is optional. An argument the form
    // cannot store is no reason to draw a section with no control in it.
    renderDialog({
      catalog: {
        guardrails: [
          ...(CATALOG.guardrails ?? []),
          guardrail({
            guardrail_name: "watsonx_guardian",
            display_name: "watsonx Guardian",
            vendor: "IBM",
            categories: ["pii"],
            create_parameters: [
              {
                name: "api_key",
                type: "string",
                required: false,
                secret: true,
                storable: true,
              },
              {
                name: "api_client",
                type: "json",
                required: false,
                secret: true,
                storable: false,
              },
            ],
          }),
        ],
      },
    })
    const user = userEvent.setup()

    await pickOption(
      user,
      "What do you want checked?",
      "Personally identifiable information",
    )
    await pickOption(user, "Which guardrail?", "watsonx Guardian · IBM")

    expect(dialog().getByLabelText("Api key")).toBeInTheDocument()
    expect(dialog().queryByRole("button", { name: "Advanced" })).toBeNull()
  })

  it("draws no control for an argument it cannot store, and names it", async () => {
    renderDialog()
    const user = userEvent.setup()

    await pickOption(user, "What do you want checked?", "Content safety")
    await pickOption(user, "Which guardrail?", "Bedrock Guardrails · Amazon")
    await user.click(dialog().getByRole("button", { name: "Advanced" }))

    expect(dialog().queryByLabelText(/Boto3 session/)).toBeNull()
    expect(dialog().getByText(/Not offered here: Boto3 session/)).toBeVisible()
  })
})

describe("DefinitionDialog, editing one", () => {
  it("never prefills the stored key", () => {
    renderDialog({ definition: stored() })
    expect(dialog().getByLabelText("Api key")).toHaveValue("")
    expect(dialog().getByLabelText("Endpoint")).toHaveValue(
      "https://api.lakera.ai",
    )
  })

  it("renames without sending, or reading, the stored arguments", async () => {
    const calls = mockApi()
    renderDialog({ definition: stored() })
    const user = userEvent.setup()

    await user.clear(dialog().getByLabelText("Name"))
    await user.type(dialog().getByLabelText("Name"), "eu-lakera")
    await user.click(dialog().getByRole("button", { name: "Save guardrail" }))

    await waitFor(() =>
      expect(calls.some((call) => call.method === "PATCH")).toBe(true),
    )
    expect(calls.find((call) => call.method === "PATCH")?.body).toEqual({
      name: "eu-lakera",
    })
  })

  it("keeps the stored key when another argument changes", async () => {
    // Leaving the key out would clear it: `create_kwargs` replaces the
    // arguments whole.
    const calls = mockApi()
    renderDialog({ definition: stored() })
    const user = userEvent.setup()

    await user.clear(dialog().getByLabelText("Endpoint"))
    await user.type(dialog().getByLabelText("Endpoint"), "https://eu.lakera.ai")
    await user.click(dialog().getByRole("button", { name: "Save guardrail" }))

    await waitFor(() =>
      expect(calls.some((call) => call.method === "PATCH")).toBe(true),
    )
    expect(calls.find((call) => call.method === "PATCH")?.body).toEqual({
      create_kwargs: {
        api_key: REDACTED_SECRET,
        endpoint: "https://eu.lakera.ai",
      },
    })
  })

  it("asks for the key again when the stored one cannot be read", async () => {
    const calls = mockApi()
    renderDialog({
      definition: stored({ create_secrets: {}, secrets_decryptable: false }),
    })
    const user = userEvent.setup()

    expect(
      dialog().getByText(/cannot read the stored credentials/),
    ).toBeVisible()
    await user.click(dialog().getByRole("button", { name: "Save guardrail" }))

    expect(
      await screen.findByText("This guardrail needs a value here."),
    ).toBeInTheDocument()
    expect(calls).toEqual([])
  })
})
