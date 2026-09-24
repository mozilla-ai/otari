import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { BuiltInGuardrailCatalog, HostedGuardrail } from "@/client"
import {
  hostedGuardrailLabel,
  MandateDialog,
} from "@/features/guardrails/MandateDialog"
import { organizationGuardrail } from "@/tests/fixtures"
import { pickOption, selectTrigger } from "@/tests/select"

const HOSTED: HostedGuardrail = {
  id: "h1",
  name: "Prompt injection",
  guardrail_name: "patronus",
  description: "Patronus, run by the deployment",
  price_per_check: 0.0005,
}

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
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <MandateDialog
        isOpen
        onClose={() => {}}
        definitions={[]}
        isDefinitionsSettled
        builtInCatalog={BUILT_IN}
        remoteCatalog={{ available: false, reason: "unset", profiles: [] }}
        isRemoteCatalogPending={false}
        hostedGuardrails={[HOSTED]}
        workspaces={[]}
        onSetUpDefinition={() => {}}
        onSaved={onSaved}
        {...props}
      />
    </QueryClientProvider>,
  )
  return { onSaved }
}

function dialog() {
  return within(screen.getByRole("dialog"))
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("hostedGuardrailLabel", () => {
  it("says what one check costs, and only the name when it is free", () => {
    expect(hostedGuardrailLabel(HOSTED)).toBe(
      "Prompt injection · $0.0005 per check",
    )
    expect(hostedGuardrailLabel({ ...HOSTED, price_per_check: null })).toBe(
      "Prompt injection",
    )
  })
})

describe("MandateDialog, with hosted guardrails on offer", () => {
  it("mandates one, with its arguments described by the built-in catalog", async () => {
    const calls = mockApi()
    const { onSaved } = renderDialog()
    const user = userEvent.setup()

    await pickOption(user, "Runs on", "A guardrail the deployment hosts")
    expect(dialog().queryByLabelText("Endpoint")).toBeNull()
    await pickOption(
      user,
      "Hosted guardrail",
      "Prompt injection · $0.0005 per check",
    )
    expect(dialog().getByLabelText("Profile a caller sends")).toHaveValue(
      "Prompt injection",
    )
    await user.type(dialog().getByLabelText("Output text"), "the answer")
    await user.click(
      dialog().getByRole("button", { name: "Mandate a guardrail" }),
    )

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(calls.find((call) => call.method === "POST")?.body).toEqual({
      profile: "Prompt injection",
      hosted_guardrail_id: "h1",
      mode: "monitor",
      on_unavailable: "block",
      validate_kwargs: { output_text: "the answer" },
      applies_to_all_workspaces: false,
      workspace_ids: [],
    })
  })

  it("offers no hosted choice where the deployment hosts none", async () => {
    renderDialog({ hostedGuardrails: [] })

    await userEvent.click(selectTrigger("Runs on"))
    expect(
      screen.queryByRole("option", {
        name: "A guardrail the deployment hosts",
      }),
    ).toBeNull()
  })

  it("edits a hosted mandate without resending its guardrail", async () => {
    const calls = mockApi()
    const { onSaved } = renderDialog({
      mandate: organizationGuardrail({
        hosted_guardrail_id: "h1",
        profile: "prompt-injection",
      }),
    })
    const user = userEvent.setup()

    expect(
      dialog().getByText("a guardrail the deployment hosts"),
    ).toBeInTheDocument()
    await pickOption(user, "Mode", "Block")
    await user.click(dialog().getByRole("button", { name: "Save mandate" }))

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    const body = calls.find((call) => call.method === "PATCH")?.body as Record<
      string,
      unknown
    >
    expect(body.mode).toBe("block")
    expect(body).not.toHaveProperty("hosted_guardrail_id")
    expect(body).not.toHaveProperty("url")
  })
})
