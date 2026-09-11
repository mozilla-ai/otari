import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type {
  GuardrailCatalog,
  GuardrailParameterSpec,
  OrganizationGuardrail,
} from "@/client"
import { OrganizationGuardrailsCard } from "@/features/tools/OrganizationGuardrailsCard"
import { API_ROOT } from "@/shared/api/client"
import { organizationContext, organizationGuardrail } from "@/tests/fixtures"
import { pickOption, selectTrigger } from "@/tests/select"

const ALPHA = "11111111-1111-1111-1111-111111111111"
const BETA = "22222222-2222-2222-2222-222222222222"

// What the operator's guardrails service answered with, joined to the parameter
// schema of the any-guardrail class each profile is built from. Shaped as the
// gateway serves it, so these tests exercise the catalog contract rather than a
// convenient stand-in for it.
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
          description: "Natural-language policy to validate against.",
        },
        {
          name: "threshold",
          type: "number",
          required: false,
          secret: false,
          default: 0.5,
        },
        {
          name: "prompt_version",
          type: "enum",
          required: false,
          secret: false,
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

// Two profiles of one guardrail class differing only in the model they pin,
// which is how an operator's guardrails configuration is ordinarily written.
// They declare the same parameters, so nothing but the name separates them.
const TWIN_PARAMETERS: GuardrailParameterSpec[] = [
  {
    name: "policy",
    type: "string",
    required: true,
    secret: false,
    description: "Natural-language policy to validate against.",
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

function mockApi({
  guardrails = [] as OrganizationGuardrail[],
  role = "owner",
  catalog = CATALOG,
  catalogGate,
}: {
  guardrails?: OrganizationGuardrail[]
  role?: string
  catalog?: GuardrailCatalog
  /** Held open to keep the catalog read in flight while the card is asserted. */
  catalogGate?: Promise<void>
} = {}) {
  const calls: { url: string; method: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = init?.method ?? "GET"
    if (url.includes("/organizations/me/guardrails")) {
      calls.push({
        url,
        method,
        body:
          typeof init?.body === "string" ? JSON.parse(init.body) : init?.body,
      })
      if (method === "GET") {
        return Response.json({ data: guardrails, count: guardrails.length })
      }
      return Response.json(guardrails[0] ?? organizationGuardrail())
    }
    if (url.includes(`${API_ROOT}/tool-settings/guardrails/profiles`)) {
      calls.push({ url, method, body: undefined })
      if (catalogGate) await catalogGate
      return Response.json(catalog)
    }
    if (url.includes(`${API_ROOT}/workspaces`)) {
      return Response.json({
        data: [
          { id: ALPHA, name: "Alpha" },
          { id: BETA, name: "Beta" },
        ],
        count: 2,
      })
    }
    return Response.json(organizationContext({ role }))
  })
  return calls
}

/**
 * Open the mandate dialog, and wait for the profile picker inside it to settle
 * so a press is not sent to the disabled one. Idempotent on an open dialog.
 */
async function openDialog() {
  if (screen.queryByRole("dialog") === null) {
    await userEvent.click(
      await screen.findByRole("button", { name: "Mandate a guardrail" }),
    )
    await screen.findByRole("dialog")
  }
}

/** The dialog's own controls, since the heading's trigger shares its words. */
function inDialog() {
  return within(screen.getByRole("dialog"))
}

async function settledPicker() {
  await openDialog()
  // The profile control is a `forms/Select` now: react-aria names its trigger
  // with the label and the current value, so it is found by label rather than
  // by the placeholder it happens to be showing.
  return await waitFor(() => selectTrigger("Guardrail profile"))
}

function renderCard() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <OrganizationGuardrailsCard onSaved={() => {}} />
    </QueryClientProvider>,
  )
}

describe("OrganizationGuardrailsCard", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("says nothing is mandated when the organization has no entries", async () => {
    mockApi()
    renderCard()

    expect(
      await screen.findByText(/No organization guardrails/),
    ).toBeInTheDocument()
  })

  it("hides the whole surface from a member who cannot manage the organization", async () => {
    const calls = mockApi({ role: "member" })
    renderCard()

    expect(
      await screen.findByText(/set by an owner or admin of the organization/),
    ).toBeInTheDocument()
    // The read is gated too, so nothing is asked for: the entries name the
    // endpoints this gateway connects to.
    expect(calls).toEqual([])
  })

  it("names the workspaces an entry runs in, and says when it runs everywhere", async () => {
    mockApi({
      guardrails: [
        organizationGuardrail({ profile: "pii", workspace_ids: [ALPHA] }),
        organizationGuardrail({
          id: "66666666-6666-6666-6666-666666666666",
          profile: "prompt-injection",
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    // The badge, not any "Alpha" on the card: the row's scope picker carries
    // the same workspace name on a checkbox, so an unscoped query here passed
    // only while the workspace list had not answered yet.
    expect(
      await screen.findByText("Every workspace, including new ones"),
    ).toBeInTheDocument()
    expect(
      screen.getAllByText("Alpha").some((node) => node.tagName === "SPAN"),
    ).toBe(true)
  })

  it("marks a paused entry and one that carries its own endpoint and credential", async () => {
    mockApi({
      guardrails: [
        organizationGuardrail({
          enabled: false,
          url: "https://guardrails.example",
          has_credential: true,
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    expect(await screen.findByText("own endpoint")).toBeInTheDocument()
    expect(screen.getByText("credential set")).toBeInTheDocument()
    // "Paused" is also an option in the status picker, so the badge is asserted
    // through the picker's value rather than by matching the word twice.
    expect(selectTrigger("Status")).toHaveTextContent("Paused")
  })

  it("never renders a stored credential back, only offers to replace it", async () => {
    mockApi({
      guardrails: [
        organizationGuardrail({
          has_credential: true,
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    const field = await screen.findByLabelText(
      "New credential for prompt-injection",
    )
    expect(field).toHaveValue("")
    expect(field).toHaveAttribute("placeholder", "replace credential")
  })

  it("omits the credential from a save that did not touch it", async () => {
    const calls = mockApi({
      guardrails: [
        organizationGuardrail({
          has_credential: true,
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    await userEvent.click(
      await screen.findByRole("button", { name: "Save prompt-injection" }),
    )

    await waitFor(() =>
      expect(calls.some((call) => call.method === "PATCH")).toBe(true),
    )
    const patch = calls.find((call) => call.method === "PATCH")
    expect(patch?.body).not.toHaveProperty("credential")
    // And no workspace list either, since the entry applies to all of them and
    // the server refuses the pair.
    expect(patch?.body).not.toHaveProperty("workspace_ids")
  })

  it("sends the chosen workspaces when the entry does not apply to all of them", async () => {
    const calls = mockApi({
      guardrails: [organizationGuardrail({ workspace_ids: [ALPHA] })],
    })
    renderCard()

    // Scoped to the entry's own group rather than a per-box aria-label: the
    // box is labelled by the workspace name a reader sees, and the group says
    // which guardrail that name belongs to.
    await userEvent.click(
      within(
        await screen.findByRole("group", { name: "prompt-injection" }),
      ).getByLabelText("Beta"),
    )
    await userEvent.click(
      screen.getByRole("button", { name: "Save prompt-injection" }),
    )

    await waitFor(() =>
      expect(calls.some((call) => call.method === "PATCH")).toBe(true),
    )
    expect(calls.find((call) => call.method === "PATCH")?.body).toMatchObject({
      applies_to_all_workspaces: false,
      workspace_ids: [ALPHA, BETA],
    })
  })

  it("removes a guardrail only through the confirm dialog", async () => {
    // otari-ai#2110.
    const calls = mockApi({
      guardrails: [organizationGuardrail({ applies_to_all_workspaces: true })],
    })
    renderCard()

    await userEvent.click(
      await screen.findByRole("button", { name: "Remove prompt-injection" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    // The fixture monitors rather than blocks, so the consequence is that
    // requests go unchecked. Saying they would have been blocked would describe
    // a guardrail that never blocked one.
    expect(
      within(dialog).getByText(/prompt-injection stops running/),
    ).toBeVisible()
    expect(within(dialog).getByText(/go unchecked/)).toBeVisible()
    expect(calls.some((call) => call.method === "DELETE")).toBe(false)

    await userEvent.click(
      within(dialog).getByRole("button", { name: "Remove permanently" }),
    )

    await waitFor(() =>
      expect(calls.some((call) => call.method === "DELETE")).toBe(true),
    )
  })

  it("says what a blocking guardrail's removal serves, not what it records", async () => {
    mockApi({
      guardrails: [
        organizationGuardrail({
          mode: "block",
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    await userEvent.click(
      await screen.findByRole("button", { name: "Remove prompt-injection" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(
      within(dialog).getByText(/would have blocked are served/),
    ).toBeVisible()
  })

  it("rewrites the endpoint in place, so a typo is not a delete and recreate", async () => {
    const calls = mockApi({
      guardrails: [
        organizationGuardrail({
          url: "https://wrong.example",
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    const endpoint = await screen.findByLabelText(
      "Endpoint for prompt-injection",
    )
    await userEvent.clear(endpoint)
    await userEvent.type(endpoint, "https://right.example")
    await userEvent.click(
      screen.getByRole("button", { name: "Save prompt-injection" }),
    )

    await waitFor(() =>
      expect(calls.some((call) => call.method === "PATCH")).toBe(true),
    )
    expect(calls.find((call) => call.method === "PATCH")?.body).toMatchObject({
      url: "https://right.example",
    })
  })

  it("leaves a stored endpoint alone on a save that did not touch it", async () => {
    const calls = mockApi({
      guardrails: [
        organizationGuardrail({
          url: "https://guardrails.example",
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    await userEvent.click(
      await screen.findByRole("button", { name: "Save prompt-injection" }),
    )

    await waitFor(() =>
      expect(calls.some((call) => call.method === "PATCH")).toBe(true),
    )
    expect(
      calls.find((call) => call.method === "PATCH")?.body,
    ).not.toHaveProperty("url")
  })

  it("keeps one row's unsaved edits when another row is saved", async () => {
    // Passes with either dependency list: TanStack Query's structural sharing
    // hands the untouched row back its previous object, so the refetch a save
    // triggers does not re-run its effect. Kept as the property worth holding
    // rather than as a regression test for the dependency array.
    mockApi({
      guardrails: [
        organizationGuardrail({
          profile: "pii",
          applies_to_all_workspaces: true,
        }),
        organizationGuardrail({
          id: "66666666-6666-6666-6666-666666666666",
          profile: "prompt-injection",
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    const edited = await screen.findByLabelText("Endpoint for prompt-injection")
    await userEvent.type(edited, "https://half-typed.example")
    await userEvent.click(screen.getByRole("button", { name: "Save pii" }))

    await waitFor(() =>
      expect(
        screen.getByLabelText("Endpoint for prompt-injection"),
      ).toHaveValue("https://half-typed.example"),
    )
  })

  it("mandates a new guardrail from the add form", async () => {
    const calls = mockApi()
    renderCard()
    await openDialog()

    // A profile the catalog does not list, which is what the by-hand field is
    // for: this entry may be destined for an endpoint of its own.
    await userEvent.click(
      await screen.findByRole("button", { name: "Name a profile by hand" }),
    )
    await userEvent.type(screen.getByLabelText("Guardrail profile"), "pii")
    await userEvent.click(
      inDialog().getByRole("button", { name: "Mandate a guardrail" }),
    )

    await waitFor(() =>
      expect(calls.some((call) => call.method === "POST")).toBe(true),
    )
    expect(calls.find((call) => call.method === "POST")?.body).toMatchObject({
      profile: "pii",
      mode: "monitor",
      url: null,
      credential: null,
      applies_to_all_workspaces: false,
      workspace_ids: [],
    })
  })

  it("offers no profile to pick until the guardrails service has answered", async () => {
    let answer = () => {}
    mockApi({
      catalogGate: new Promise<void>((resolve) => {
        answer = resolve
      }),
    })
    renderCard()
    await openDialog()

    // The control does not start as a free-text box and turn into a picker
    // under the operator's cursor: it is the picker throughout, and says so
    // while it waits.
    // Named by its label now, and reading its waiting placeholder: the control
    // is the picker throughout rather than a box that becomes one.
    const waiting = await waitFor(() => selectTrigger("Guardrail profile"))
    expect(waiting).toBeDisabled()
    // The waiting sentence is the control's description, which is where it is
    // both rendered and announced; the trigger's own text is the selected
    // option's, and there is nothing to select yet.
    expect(waiting).toHaveAccessibleDescription(
      /Reading the guardrails service/,
    )
    expect(
      screen.queryByRole("button", { name: "Name a profile by hand" }),
    ).toBeNull()

    answer()
    // Waited on the state rather than on the element: the trigger is named by
    // its label throughout, so it exists before and after the catalog answers
    // and only its disabled state says which.
    await waitFor(() =>
      expect(selectTrigger("Guardrail profile")).toBeEnabled(),
    )
  })

  it("picks a profile from what the guardrails service has built", async () => {
    const calls = mockApi()
    renderCard()

    await settledPicker()
    await pickOption(userEvent.setup(), "Guardrail profile", "prompt-injection")
    await userEvent.click(
      inDialog().getByRole("button", { name: "Mandate a guardrail" }),
    )

    await waitFor(() =>
      expect(calls.some((call) => call.method === "POST")).toBe(true),
    )
    expect(calls.find((call) => call.method === "POST")?.body).toMatchObject({
      profile: "prompt-injection",
    })
  })

  it("writes the chosen profile's typed parameters into validate_kwargs", async () => {
    const calls = mockApi()
    renderCard()
    const user = userEvent.setup()

    await settledPicker()
    await pickOption(user, "Guardrail profile", "house-policy")
    await user.type(await screen.findByLabelText("Policy"), "No personal data.")
    await user.type(screen.getByLabelText("Threshold"), "0.8")
    await pickOption(user, "Prompt version", "v2")
    await user.click(
      inDialog().getByRole("button", { name: "Mandate a guardrail" }),
    )

    await waitFor(() =>
      expect(calls.some((call) => call.method === "POST")).toBe(true),
    )
    expect(calls.find((call) => call.method === "POST")?.body).toMatchObject({
      profile: "house-policy",
      validate_kwargs: {
        policy: "No personal data.",
        // Coerced back to its JSON-native type, not left as the typed string.
        threshold: 0.8,
        prompt_version: "v2",
      },
    })
  })

  it("refuses to add an entry whose guardrail needs a parameter it has not got", async () => {
    // One validation path, and it is the app's own: `parameters.check()` names
    // the field and says what it needs. The native `required` attribute would
    // refuse the submit first and silently, so inside a dialog that message
    // would never be reached.
    const calls = mockApi()
    const user = userEvent.setup()
    renderCard()

    await settledPicker()
    await pickOption(user, "Guardrail profile", "house-policy")
    await user.click(
      inDialog().getByRole("button", { name: "Mandate a guardrail" }),
    )

    expect(
      await screen.findByText("This guardrail needs a value here."),
    ).toBeInTheDocument()
    expect(calls.some((call) => call.method === "POST")).toBe(false)
  })

  it("omits a parameter left blank, so the profile's own default still applies", async () => {
    const calls = mockApi()
    renderCard()
    const user = userEvent.setup()

    await settledPicker()
    await pickOption(user, "Guardrail profile", "house-policy")
    await user.type(await screen.findByLabelText("Policy"), "No personal data.")
    await user.click(
      inDialog().getByRole("button", { name: "Mandate a guardrail" }),
    )

    await waitFor(() =>
      expect(calls.some((call) => call.method === "POST")).toBe(true),
    )
    const body = calls.find((call) => call.method === "POST")?.body as {
      validate_kwargs: Record<string, unknown>
    }
    expect(body.validate_kwargs).toEqual({ policy: "No personal data." })
  })

  it("renders a stored parameter into its typed field", async () => {
    mockApi({
      guardrails: [
        organizationGuardrail({
          profile: "house-policy",
          applies_to_all_workspaces: true,
          validate_kwargs: { policy: "Stored policy." },
        }),
      ],
    })
    renderCard()

    expect(await screen.findByLabelText("Policy")).toHaveValue("Stored policy.")
  })

  it("round-trips a stored parameter the catalog does not describe", async () => {
    const calls = mockApi({
      guardrails: [
        organizationGuardrail({
          profile: "house-policy",
          applies_to_all_workspaces: true,
          validate_kwargs: { policy: "Stored policy.", unmapped: [1, 2] },
        }),
      ],
    })
    renderCard()

    // The raw editor opens on its own when it holds something, so a value with
    // no typed field is not one the operator has to go looking for.
    expect(await screen.findByLabelText("Parameters (JSON)")).toHaveValue(
      JSON.stringify({ unmapped: [1, 2] }, null, 2),
    )

    await userEvent.click(
      screen.getByRole("button", { name: "Save house-policy" }),
    )
    await waitFor(() =>
      expect(calls.some((call) => call.method === "PATCH")).toBe(true),
    )
    expect(calls.find((call) => call.method === "PATCH")?.body).toMatchObject({
      validate_kwargs: { policy: "Stored policy.", unmapped: [1, 2] },
    })
  })

  it("refuses a save whose raw parameters are not JSON", async () => {
    const calls = mockApi({
      guardrails: [
        organizationGuardrail({
          profile: "prompt-injection",
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    await userEvent.click(
      await screen.findByRole("button", {
        name: /Other parameters for prompt-injection/,
      }),
    )
    await userEvent.type(screen.getByLabelText("Parameters (JSON)"), "not json")
    await userEvent.click(
      screen.getByRole("button", { name: "Save prompt-injection" }),
    )

    expect(await screen.findByText("Not valid JSON.")).toBeInTheDocument()
    expect(calls.some((call) => call.method === "PATCH")).toBe(false)
  })

  it("falls back to naming a profile by hand, and says why, when the service cannot be listed", async () => {
    const calls = mockApi({
      catalog: {
        available: false,
        reason: "The guardrails service could not be reached.",
        profiles: [],
      },
    })
    renderCard()
    await openDialog()

    expect(
      await screen.findByText("The guardrails service could not be reached."),
    ).toBeInTheDocument()
    await userEvent.type(screen.getByLabelText("Guardrail profile"), "pii")
    await userEvent.click(
      inDialog().getByRole("button", { name: "Mandate a guardrail" }),
    )

    await waitFor(() =>
      expect(calls.some((call) => call.method === "POST")).toBe(true),
    )
    expect(calls.find((call) => call.method === "POST")?.body).toMatchObject({
      profile: "pii",
    })
  })

  it("clears a typed parameter when the picker moves to a profile with the same schema", async () => {
    // otari-ai#2119. The two profiles declare identical parameters, so a form
    // that re-seeds on the schema alone keeps what was typed for the first and
    // sends it under the second one's name.
    mockApi({ catalog: TWIN_CATALOG })
    renderCard()
    const user = userEvent.setup()

    await settledPicker()
    await pickOption(user, "Guardrail profile", "house-policy-fast")
    await user.type(await screen.findByLabelText("Policy"), "No personal data.")
    await pickOption(user, "Guardrail profile", "house-policy-strict")

    await waitFor(() => expect(screen.getByLabelText("Policy")).toHaveValue(""))
  })

  it("keeps what is filled in while a profile the catalog does not describe is typed", async () => {
    // The other half of otari-ai#2119: a name typed by hand reaches the form one
    // character at a time, and none of those characters spell a profile the
    // catalog describes, so they have to share one identity or the form resets
    // on every keystroke.
    mockApi()
    renderCard()
    const user = userEvent.setup()

    await openDialog()
    await user.click(
      await screen.findByRole("button", { name: "Name a profile by hand" }),
    )
    // The raw editor is the only place a parameter can go for a profile with no
    // schema behind it.
    await user.click(
      inDialog().getByRole("button", {
        name: /Other parameters for the new guardrail/,
      }),
    )
    await user.type(
      screen.getByLabelText("Parameters (JSON)"),
      '{{"threshold": 0.8}',
    )
    await user.type(screen.getByLabelText("Guardrail profile"), "pii")

    expect(screen.getByLabelText("Parameters (JSON)")).toHaveValue(
      '{"threshold": 0.8}',
    )
  })

  it("asks for no catalog from a member who cannot manage the organization", async () => {
    const calls = mockApi({ role: "member" })
    renderCard()

    await screen.findByText(/set by an owner or admin of the organization/)
    expect(calls).toEqual([])
  })
})
