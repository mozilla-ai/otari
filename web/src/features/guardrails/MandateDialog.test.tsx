import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  BuiltInGuardrailCatalog,
  OrganizationGuardrailDefinition,
} from "@/client"
import { MandateDialog } from "@/features/guardrails/MandateDialog"
import { REDACTED_SECRET } from "@/shared/helpers/redaction"
import { organizationGuardrail } from "@/tests/fixtures"
import { pickOption, selectTrigger } from "@/tests/select"

const DEFINITION: OrganizationGuardrailDefinition = {
  id: "d1",
  organization_id: "o1",
  name: "prod-patronus",
  guardrail_name: "patronus",
  enabled: true,
  build_state: "built",
  create_kwargs: {},
  create_secrets: { api_key: REDACTED_SECRET },
  secrets_decryptable: true,
  created_at: "2026-09-22T00:00:00Z",
  updated_at: "2026-09-22T00:00:00Z",
}

// Patronus takes typed per-call arguments, which is where they are asked for.
const BUILT_IN = {
  guardrails: [
    {
      guardrail_name: "patronus",
      display_name: "Patronus",
      categories: ["prompt_injection"],
      validate_parameters: [
        {
          name: "output_text",
          type: "string",
          required: false,
          secret: false,
          storable: true,
        },
      ],
    },
  ],
} as unknown as BuiltInGuardrailCatalog

function mockApi() {
  const calls: { method: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
    calls.push({
      method: init?.method ?? "GET",
      body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
    })
    return Response.json(organizationGuardrail())
  })
  return calls
}

function renderDialog(
  props: Partial<Parameters<typeof MandateDialog>[0]> = {},
) {
  const onSaved = vi.fn()
  const onSetUpDefinition = vi.fn()
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <MandateDialog
        isOpen
        onClose={() => {}}
        definitions={[DEFINITION]}
        isDefinitionsSettled
        builtInCatalog={BUILT_IN}
        remoteCatalog={{ available: false, reason: "unset", profiles: [] }}
        isRemoteCatalogPending={false}
        workspaces={[]}
        onSetUpDefinition={onSetUpDefinition}
        onSaved={onSaved}
        {...props}
      />
    </QueryClientProvider>,
  )
  return { onSaved, onSetUpDefinition }
}

function dialog() {
  return within(screen.getByRole("dialog"))
}

function posted(calls: { method: string; body: unknown }[], method: string) {
  return calls.find((call) => call.method === method)?.body
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("MandateDialog, creating one", () => {
  it("starts on a guardrail you set up, when there is one", () => {
    renderDialog()
    expect(selectTrigger("Runs on")).toHaveTextContent("A guardrail you set up")
  })

  it("holds the shape until the definitions read has answered", () => {
    renderDialog({ definitions: [], isDefinitionsSettled: false })
    expect(selectTrigger("Runs on")).toBeDisabled()
  })

  it("offers only your own service, with the way to set one up, when there is none", async () => {
    const { onSetUpDefinition } = renderDialog({ definitions: [] })

    expect(selectTrigger("Runs on")).toHaveTextContent(
      "Your own guardrails service",
    )
    await userEvent.click(
      dialog().getByRole("button", { name: "Configure a guardrail" }),
    )
    expect(onSetUpDefinition).toHaveBeenCalled()
  })

  it("mandates a definition, with no endpoint or credential to send", async () => {
    const calls = mockApi()
    const { onSaved } = renderDialog()
    const user = userEvent.setup()

    expect(dialog().queryByLabelText("Endpoint")).toBeNull()
    expect(dialog().queryByLabelText("Credential")).toBeNull()
    await pickOption(user, "Guardrail", "prod-patronus")
    // The profile follows the definition's name until the user types one.
    expect(dialog().getByLabelText("Profile a caller sends")).toHaveValue(
      "prod-patronus",
    )
    await pickOption(user, "Mode", "Block")
    await user.type(dialog().getByLabelText("Output text"), "the answer")
    await user.click(
      dialog().getByRole("button", { name: "Mandate a guardrail" }),
    )

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(posted(calls, "POST")).toEqual({
      profile: "prod-patronus",
      definition_id: "d1",
      mode: "block",
      on_unavailable: "block",
      validate_kwargs: { output_text: "the answer" },
      applies_to_all_workspaces: false,
      workspace_ids: [],
    })
  })

  it("says how each shape fails, in its own words", async () => {
    renderDialog()
    expect(selectTrigger("If it can't run")).toBeInTheDocument()

    await pickOption(
      userEvent.setup(),
      "Runs on",
      "Your own guardrails service",
    )
    expect(selectTrigger("If unreachable")).toBeInTheDocument()
    expect(dialog().getByLabelText("Endpoint")).toBeInTheDocument()
    expect(dialog().queryByText("Guardrail", { selector: "label" })).toBeNull()
  })
})

describe("MandateDialog, editing one", () => {
  it("shows the shape as text, since it cannot change", () => {
    renderDialog({
      mandate: organizationGuardrail({ definition_id: "d1", profile: "p" }),
    })
    expect(dialog().getByText("a guardrail you set up")).toBeInTheDocument()
    expect(dialog().queryByRole("button", { name: /Runs on/ })).toBeNull()
  })

  it("sends a remote mandate's mode without touching its endpoint or credential", async () => {
    const calls = mockApi()
    const { onSaved } = renderDialog({
      mandate: organizationGuardrail({
        profile: "pii",
        url: "https://guardrails.example/validate",
        has_credential: true,
        mode: "monitor",
      }),
    })
    const user = userEvent.setup()

    await pickOption(user, "Mode", "Block")
    await user.click(dialog().getByRole("button", { name: "Save mandate" }))

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    const body = posted(calls, "PATCH") as Record<string, unknown>
    expect(body.mode).toBe("block")
    // Omitted, so the stored endpoint and credential stay as they are.
    expect(body).not.toHaveProperty("url")
    expect(body).not.toHaveProperty("credential")
    expect(body).not.toHaveProperty("definition_id")
  })

  it("leaves a definition link alone on a save that did not move it", async () => {
    const calls = mockApi()
    const { onSaved } = renderDialog({
      mandate: organizationGuardrail({ definition_id: "d1", profile: "p" }),
    })

    await userEvent.click(
      dialog().getByRole("button", { name: "Save mandate" }),
    )

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(posted(calls, "PATCH")).not.toHaveProperty("definition_id")
  })
})
