import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  GuardrailCatalog,
  GuardrailParameterSpec,
  OrganizationGuardrail,
  Workspace,
} from "@/client"
import { MandateDialog } from "@/features/guardrails/MandateDialog"
import { organizationGuardrail } from "@/tests/fixtures"
import { pickOption, selectTrigger } from "@/tests/select"

// The mandate dialog on its "your own guardrails service" branch, which is the
// form the organization card carried before the page replaced it. These are
// that card's assertions, kept against the component that now owns them.

const ALPHA = "11111111-1111-1111-1111-111111111111"
const BETA = "22222222-2222-2222-2222-222222222222"
const WORKSPACES = [
  { id: ALPHA, name: "Alpha" },
  { id: BETA, name: "Beta" },
] as Workspace[]

// What the operator's guardrails service answered with, joined to the parameter
// schema of the any-guardrail class each profile is built from.
const CATALOG: GuardrailCatalog = {
  available: true,
  reason: null,
  profiles: [
    {
      profile: "house-policy",
      guardrail: "any_llm",
      model_id: null,
      parameters_known: true,
      parameters: [
        {
          name: "policy",
          type: "string",
          required: true,
          secret: false,
          storable: true,
          description: "Natural-language policy to validate against.",
        },
        {
          name: "threshold",
          type: "number",
          required: false,
          secret: false,
          storable: true,
          default: 0.5,
        },
        {
          name: "prompt_version",
          type: "enum",
          required: false,
          secret: false,
          storable: true,
          choices: ["v1", "v2"],
        },
      ],
    },
    {
      profile: "prompt-injection",
      guardrail: "injec_guard",
      model_id: "leolee99/InjecGuard",
      parameters_known: true,
      parameters: [],
    },
  ],
}

// Two profiles of one class differing only in the model they pin, so nothing
// but the name separates them.
const TWIN_PARAMETERS: GuardrailParameterSpec[] = [
  {
    name: "policy",
    type: "string",
    required: true,
    secret: false,
    storable: true,
  },
]
const TWIN_CATALOG: GuardrailCatalog = {
  available: true,
  reason: null,
  profiles: [
    {
      profile: "house-policy-fast",
      guardrail: "any_llm",
      model_id: "openai/gpt-4o-mini",
      parameters_known: true,
      parameters: TWIN_PARAMETERS,
    },
    {
      profile: "house-policy-strict",
      guardrail: "any_llm",
      model_id: "openai/gpt-4o",
      parameters_known: true,
      parameters: TWIN_PARAMETERS,
    },
  ],
}

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

function renderDialog({
  mandate,
  catalog = CATALOG,
  isCatalogPending = false,
}: {
  mandate?: OrganizationGuardrail
  catalog?: GuardrailCatalog
  isCatalogPending?: boolean
} = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const onSaved = vi.fn()
  render(
    <QueryClientProvider client={client}>
      <MandateDialog
        isOpen
        onClose={() => {}}
        mandate={mandate}
        // None, so a new mandate opens on "your own guardrails service".
        definitions={[]}
        isDefinitionsSettled
        builtInCatalog={{ guardrails: [] }}
        remoteCatalog={catalog}
        isRemoteCatalogPending={isCatalogPending}
        workspaces={WORKSPACES}
        onSetUpDefinition={() => {}}
        onSaved={onSaved}
      />
    </QueryClientProvider>,
  )
  return onSaved
}

function dialog() {
  return within(screen.getByRole("dialog"))
}

function sent(calls: { method: string; body: unknown }[], method: string) {
  return calls.find((call) => call.method === method)?.body as
    | Record<string, unknown>
    | undefined
}

const submitLabel = (mandate?: OrganizationGuardrail) =>
  mandate ? "Save mandate" : "Mandate a guardrail"

async function submit(mandate?: OrganizationGuardrail) {
  await userEvent.click(
    dialog().getByRole("button", { name: submitLabel(mandate) }),
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("MandateDialog, creating one on your own service", () => {
  it("mandates a profile named by hand, sending no endpoint or credential", async () => {
    const calls = mockApi()
    const onSaved = renderDialog()

    await userEvent.click(
      dialog().getByRole("button", { name: "Name a profile by hand" }),
    )
    await userEvent.type(dialog().getByLabelText("Guardrail profile"), "pii")
    await submit()

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(sent(calls, "POST")).toMatchObject({
      profile: "pii",
      mode: "monitor",
      url: null,
      credential: null,
      applies_to_all_workspaces: false,
      workspace_ids: [],
    })
  })

  it("offers no profile to pick until the guardrails service has answered", () => {
    renderDialog({ isCatalogPending: true })

    // The picker throughout, never a text box that turns into one.
    const waiting = selectTrigger("Guardrail profile")
    expect(waiting).toBeDisabled()
    expect(waiting).toHaveAccessibleDescription(
      /Reading the guardrails service/,
    )
    expect(
      dialog().queryByRole("button", { name: "Name a profile by hand" }),
    ).toBeNull()
  })

  it("writes the chosen profile's typed parameters into validate_kwargs", async () => {
    const calls = mockApi()
    const onSaved = renderDialog()
    const user = userEvent.setup()

    await pickOption(user, "Guardrail profile", "house-policy")
    await user.type(
      await dialog().findByLabelText("Policy"),
      "No personal data.",
    )
    await user.type(dialog().getByLabelText("Threshold"), "0.8")
    await pickOption(user, "Prompt version", "v2")
    await submit()

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(sent(calls, "POST")).toMatchObject({
      profile: "house-policy",
      // Coerced back to its JSON-native type, not left as the typed string.
      validate_kwargs: {
        policy: "No personal data.",
        threshold: 0.8,
        prompt_version: "v2",
      },
    })
  })

  it("refuses a mandate whose guardrail needs a parameter it has not got", async () => {
    const calls = mockApi()
    renderDialog()
    const user = userEvent.setup()

    await pickOption(user, "Guardrail profile", "house-policy")
    await submit()

    expect(
      await screen.findByText("This guardrail needs a value here."),
    ).toBeInTheDocument()
    expect(calls).toEqual([])
  })

  it("omits a parameter left blank, so the profile's own default still applies", async () => {
    const calls = mockApi()
    const onSaved = renderDialog()
    const user = userEvent.setup()

    await pickOption(user, "Guardrail profile", "house-policy")
    await user.type(
      await dialog().findByLabelText("Policy"),
      "No personal data.",
    )
    await submit()

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(sent(calls, "POST")?.validate_kwargs).toEqual({
      policy: "No personal data.",
    })
  })

  it("falls back to naming a profile by hand, and says why, when the service cannot be listed", async () => {
    const calls = mockApi()
    const onSaved = renderDialog({
      catalog: {
        available: false,
        reason: "The guardrails service could not be reached.",
        profiles: [],
      },
    })

    expect(
      dialog().getByText("The guardrails service could not be reached."),
    ).toBeInTheDocument()
    await userEvent.type(dialog().getByLabelText("Guardrail profile"), "pii")
    await submit()

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(sent(calls, "POST")).toMatchObject({ profile: "pii" })
  })

  it("clears a typed parameter when the picker moves to a profile with the same schema", async () => {
    // otari-ai#2119: re-seeding on the schema alone would send what was typed
    // for the first profile under the second one's name.
    renderDialog({ catalog: TWIN_CATALOG })
    const user = userEvent.setup()

    await pickOption(user, "Guardrail profile", "house-policy-fast")
    await user.type(
      await dialog().findByLabelText("Policy"),
      "No personal data.",
    )
    await pickOption(user, "Guardrail profile", "house-policy-strict")

    await waitFor(() =>
      expect(dialog().getByLabelText("Policy")).toHaveValue(""),
    )
  })

  it("keeps what is filled in while a profile the catalog does not describe is typed", async () => {
    // The other half of otari-ai#2119: a name typed by hand arrives one
    // character at a time, and none of them spell a described profile.
    renderDialog()
    const user = userEvent.setup()

    await user.click(
      dialog().getByRole("button", { name: "Name a profile by hand" }),
    )
    await user.click(
      dialog().getByRole("button", {
        name: /Other parameters for the new guardrail/,
      }),
    )
    await user.type(
      dialog().getByLabelText("Parameters (JSON)"),
      '{{"threshold": 0.8}',
    )
    await user.type(dialog().getByLabelText("Guardrail profile"), "pii")

    expect(dialog().getByLabelText("Parameters (JSON)")).toHaveValue(
      '{"threshold": 0.8}',
    )
  })

  it("names each workspace box inside a group named for the guardrail", async () => {
    // A named group rather than a per-box aria-label: the box keeps the
    // workspace name a reader sees, and the group says whose it is.
    renderDialog({
      mandate: organizationGuardrail({
        profile: "pii",
        workspace_ids: [ALPHA],
      }),
    })

    const group = dialog().getByRole("group", { name: "pii" })
    expect(within(group).getByLabelText("Alpha")).toBeChecked()
    expect(within(group).getByLabelText("Beta")).not.toBeChecked()
  })
})

describe("MandateDialog, editing one on your own service", () => {
  it("never renders a stored credential back", () => {
    renderDialog({ mandate: organizationGuardrail({ has_credential: true }) })

    const field = dialog().getByLabelText("Credential")
    expect(field).toHaveValue("")
    expect(field).toHaveAttribute("placeholder", "stored; blank keeps it")
  })

  it("omits the credential, and the workspace list, from a save that did not touch them", async () => {
    const calls = mockApi()
    const mandate = organizationGuardrail({
      has_credential: true,
      applies_to_all_workspaces: true,
    })
    const onSaved = renderDialog({ mandate })

    await submit(mandate)

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    const body = sent(calls, "PATCH")
    expect(body).not.toHaveProperty("credential")
    // The server refuses a list beside "every workspace".
    expect(body).not.toHaveProperty("workspace_ids")
  })

  it("sends the chosen workspaces when the mandate does not apply to all of them", async () => {
    const calls = mockApi()
    const mandate = organizationGuardrail({ workspace_ids: [ALPHA] })
    const onSaved = renderDialog({ mandate })

    await userEvent.click(
      within(
        dialog().getByRole("group", { name: "prompt-injection" }),
      ).getByLabelText("Beta"),
    )
    await submit(mandate)

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(sent(calls, "PATCH")).toMatchObject({
      applies_to_all_workspaces: false,
      workspace_ids: [ALPHA, BETA],
    })
  })

  it("rewrites the endpoint in place, and clears it with an empty box", async () => {
    // Three states on the wire: omitted leaves it, "" clears it, a value
    // replaces it.
    const calls = mockApi()
    const mandate = organizationGuardrail({
      url: "https://guardrails.example/validate",
    })
    const onSaved = renderDialog({ mandate })

    await userEvent.clear(dialog().getByLabelText("Endpoint"))
    await submit(mandate)

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(sent(calls, "PATCH")?.url).toBe("")
  })

  it("renders a stored parameter into its typed field", () => {
    renderDialog({
      mandate: organizationGuardrail({
        profile: "house-policy",
        validate_kwargs: { policy: "Stored policy." },
      }),
    })

    expect(dialog().getByLabelText("Policy")).toHaveValue("Stored policy.")
  })

  it("round-trips a stored parameter the catalog does not describe", async () => {
    const calls = mockApi()
    const mandate = organizationGuardrail({
      profile: "house-policy",
      validate_kwargs: { policy: "Stored policy.", unmapped: [1, 2] },
    })
    const onSaved = renderDialog({ mandate })

    // The raw editor opens on its own when it holds something.
    expect(dialog().getByLabelText("Parameters (JSON)")).toHaveValue(
      JSON.stringify({ unmapped: [1, 2] }, null, 2),
    )
    await submit(mandate)

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(sent(calls, "PATCH")).toMatchObject({
      validate_kwargs: { policy: "Stored policy.", unmapped: [1, 2] },
    })
  })

  it("refuses a save whose raw parameters are not JSON", async () => {
    const calls = mockApi()
    const mandate = organizationGuardrail()
    renderDialog({ mandate })

    await userEvent.click(
      dialog().getByRole("button", {
        name: /Other parameters for prompt-injection/,
      }),
    )
    await userEvent.type(
      dialog().getByLabelText("Parameters (JSON)"),
      "not json",
    )
    await submit(mandate)

    expect(await screen.findByText("Not valid JSON.")).toBeInTheDocument()
    expect(calls).toEqual([])
  })
})
