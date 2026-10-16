import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { CatalogModelSummary, PolicySpec } from "@/client"
import { CreatePolicyDialog } from "@/features/routing/CreatePolicyDialog"
import type { Intent } from "@/features/routing/policyDraft"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  catalogModelSummary,
  catalogResponse,
  organizationContext,
} from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"
import { pickOption, selectTrigger } from "@/tests/select"

const SONNET = catalogModelSummary({
  id: "anthropic/claude-sonnet-5.5",
  name: "Claude Sonnet 5.5",
  selectors: [
    "anthropic:claude-sonnet-5-5",
    "bedrock:anthropic.claude-sonnet-5-5-v1:0",
  ],
  providers: ["anthropic", "bedrock"],
  resolves_to: "anthropic:claude-sonnet-5-5",
})
const GPT = catalogModelSummary({
  id: "openai/gpt-5",
  name: "GPT-5",
  selectors: ["openai:gpt-5"],
  providers: ["openai"],
  resolves_to: "openai:gpt-5",
})
const HAIKU = catalogModelSummary({
  id: "anthropic/claude-haiku-4.5",
  name: "Claude Haiku 4.5",
  selectors: ["anthropic:claude-haiku-4-5"],
  providers: ["anthropic"],
  resolves_to: "anthropic:claude-haiku-4-5",
})
const CATALOG = [SONNET, GPT, HAIKU]

/** The server's own match: a term in the name, the id or any selector. */
function search(models: CatalogModelSummary[], term: string) {
  const wanted = term.trim().toLowerCase()
  return models.filter(
    (model) =>
      !wanted ||
      [model.name, model.id, ...model.selectors].some((value) =>
        value.toLowerCase().includes(wanted),
      ),
  )
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi() {
  const writes: { name: string; spec: PolicySpec; user_id?: string | null }[] =
    []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = new URL(String(input), "http://localhost")
    const method = (init?.method ?? "GET").toUpperCase()
    if (url.pathname === `${API_ROOT}/catalog/models`) {
      return jsonResponse(
        catalogResponse(search(CATALOG, url.searchParams.get("search") ?? "")),
      )
    }
    if (url.pathname === `${API_ROOT}/routing/policies` && method === "POST") {
      const body = JSON.parse(String(init?.body))
      writes.push(body)
      return jsonResponse({ ...body, source: "stored", is_dynamic: false })
    }
    if (url.pathname === `${API_ROOT}/tool-settings`) {
      return jsonResponse({
        fields: [{ key: "guardrails_url", value: "http://guardrails:8000" }],
      })
    }
    if (url.pathname === `${API_ROOT}/organizations/me`) {
      return jsonResponse(organizationContext())
    }
    if (url.pathname === `${API_ROOT}/organizations/me/members`) {
      return jsonResponse({ data: [], count: 0 })
    }
    return jsonResponse([])
  })
  return { writes }
}

async function renderDialog(startAt?: Intent) {
  const onClose = vi.fn()
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  await renderWithRouter(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>
        <CreatePolicyDialog
          isOpen
          startAt={startAt}
          isDeploymentWide
          workspaceId={null}
          onClose={onClose}
        />
      </QueryClientProvider>
    </DeploymentProvider>,
    { url: "/" },
  )
  await screen.findByRole("dialog")
  return { onClose }
}

type TestUser = ReturnType<typeof userEvent.setup>

/** Put a combo box's list away, only while it is open: Escape on a closed one
 *  reaches the dialog. */
async function dismiss(user: TestUser, box: HTMLElement) {
  if (box.getAttribute("aria-expanded") === "true") {
    await user.keyboard("{Escape}")
  }
}

async function typeInto(user: TestUser, name: RegExp, text: string) {
  const box = screen.getByRole("combobox", { name })
  await user.type(box, text)
  await dismiss(user, box)
}

const dialog = () => within(screen.getByRole("dialog"))

afterEach(() => {
  vi.restoreAllMocks()
})

describe("CreatePolicyDialog", () => {
  it("shows the model string callers will send, and refuses one with a space", async () => {
    mockApi()
    const user = userEvent.setup()
    await renderDialog("fallback")

    const name = screen.getByRole("textbox", { name: /^name$/i })
    await user.type(name, "fast")
    expect(dialog().getByText('model: "fast"')).toBeInTheDocument()

    await user.type(name, " lane")
    expect(
      dialog().getByText("A policy name cannot contain spaces."),
    ).toBeInTheDocument()
    expect(name).toHaveAttribute("aria-invalid", "true")
    await typeInto(user, /^main model$/i, "openai:gpt-5")
    expect(
      dialog().getByRole("button", { name: "Create policy" }),
    ).toBeDisabled()
  })

  it("lists models rather than selectors, and saves the provider chosen", async () => {
    const { writes } = mockApi()
    const user = userEvent.setup()
    await renderDialog("fallback")

    await user.type(screen.getByRole("textbox", { name: /^name$/i }), "smart")
    await user.type(
      screen.getByRole("combobox", { name: /^main model$/i }),
      "sonnet",
    )
    // One row for the model, naming every provider that serves it.
    const option = await screen.findByRole("option", {
      name: "Claude Sonnet 5.5 (Anthropic, Amazon Bedrock)",
    })
    await user.click(option)

    // A fresh pick takes the offering the catalog id resolves to.
    expect(selectTrigger("Provider")).toHaveTextContent("Anthropic")
    await pickOption(user, "Provider", "Amazon Bedrock")
    expect(screen.getByRole("combobox", { name: /^main model$/i })).toHaveValue(
      "Claude Sonnet 5.5",
    )
    await typeInto(user, /^backup 1$/i, "openai:gpt-5")

    await user.click(dialog().getByRole("button", { name: "Create policy" }))
    await waitFor(() => expect(writes).toHaveLength(1))
    expect(writes[0].spec.select).toEqual([
      { default: "bedrock:anthropic.claude-sonnet-5-5-v1:0" },
    ])
  })

  it("saves the backups as the failure chain, in the order they are arranged", async () => {
    const { writes } = mockApi()
    const user = userEvent.setup()
    await renderDialog("fallback")

    await user.type(screen.getByRole("textbox", { name: /^name$/i }), "fast")
    await typeInto(user, /^main model$/i, "anthropic:claude-sonnet-5-5")
    await typeInto(user, /^backup 1$/i, "openai:gpt-5")
    await user.click(dialog().getByRole("button", { name: "Add a backup" }))
    await typeInto(user, /^backup 2$/i, "anthropic:claude-haiku-4-5")

    await user.click(dialog().getByRole("button", { name: "Move backup 2 up" }))
    // Focus follows the backup to the top, onto its one arrow that still moves it.
    expect(
      dialog().getByRole("button", { name: "Move backup 1 down" }),
    ).toHaveFocus()
    expect(screen.getByRole("combobox", { name: /^backup 1$/i })).toHaveValue(
      "anthropic:claude-haiku-4-5",
    )

    await user.click(dialog().getByRole("button", { name: "Create policy" }))
    await waitFor(() => expect(writes).toHaveLength(1))
    expect(writes[0].spec).toEqual({
      select: [{ default: "anthropic:claude-sonnet-5-5" }],
      on_failure: ["anthropic:claude-haiku-4-5", "openai:gpt-5"],
    })
  })

  it("drops a removed backup from the chain", async () => {
    const { writes } = mockApi()
    const user = userEvent.setup()
    await renderDialog("fallback")

    await user.type(screen.getByRole("textbox", { name: /^name$/i }), "fast")
    await typeInto(user, /^main model$/i, "openai:gpt-5")
    await typeInto(user, /^backup 1$/i, "anthropic:claude-haiku-4-5")
    await user.click(dialog().getByRole("button", { name: "Add a backup" }))
    await typeInto(user, /^backup 2$/i, "anthropic:claude-sonnet-5-5")
    await user.click(dialog().getByRole("button", { name: "Remove backup 1" }))
    // The one left is the last, which a fallback cannot do without.
    expect(
      dialog().getByRole("button", { name: "Remove backup 1" }),
    ).toBeDisabled()

    await user.click(dialog().getByRole("button", { name: "Create policy" }))
    await waitFor(() => expect(writes).toHaveLength(1))
    expect(writes[0].spec.on_failure).toEqual(["anthropic:claude-sonnet-5-5"])
  })

  it("opens a fallback on one empty, required backup, and will not create it empty", async () => {
    const { writes } = mockApi()
    const user = userEvent.setup()
    await renderDialog("fallback")

    const backup = screen.getByRole("combobox", { name: /^backup 1$/i })
    expect(backup).toHaveValue("")
    expect(backup).toBeRequired()
    expect(dialog().queryByRole("combobox", { name: /^backup 2$/i })).toBeNull()
    expect(dialog().queryByText(/No backups yet/)).not.toBeInTheDocument()
    expect(
      dialog().getByRole("button", { name: "Remove backup 1" }),
    ).toBeDisabled()

    await user.type(screen.getByRole("textbox", { name: /^name$/i }), "fast")
    await typeInto(user, /^main model$/i, "openai:gpt-5")
    const create = dialog().getByRole("button", { name: "Create policy" })
    expect(create).toBeDisabled()

    await typeInto(user, /^backup 1$/i, "anthropic:claude-haiku-4-5")
    expect(create).toBeEnabled()
    await user.click(create)
    await waitFor(() => expect(writes).toHaveLength(1))
    expect(writes[0].spec).toEqual({
      select: [{ default: "openai:gpt-5" }],
      on_failure: ["anthropic:claude-haiku-4-5"],
    })
  })

  it("refuses a backup that repeats the main model or another backup", async () => {
    mockApi()
    const user = userEvent.setup()
    await renderDialog("fallback")

    await user.type(screen.getByRole("textbox", { name: /^name$/i }), "fast")
    await typeInto(user, /^main model$/i, "openai:gpt-5")
    await typeInto(user, /^backup 1$/i, "openai:gpt-5")
    const create = dialog().getByRole("button", { name: "Create policy" })
    expect(
      dialog().getByText("Same as the main model. Pick another one."),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("combobox", { name: /^backup 1$/i }),
    ).toHaveAttribute("aria-invalid", "true")
    expect(create).toBeDisabled()

    const first = screen.getByRole("combobox", { name: /^backup 1$/i })
    await user.clear(first)
    await typeInto(user, /^backup 1$/i, "anthropic:claude-haiku-4-5")
    await user.click(dialog().getByRole("button", { name: "Add a backup" }))
    await typeInto(user, /^backup 2$/i, "anthropic:claude-haiku-4-5")
    expect(
      dialog().getByText("Same as backup 1. Pick another one."),
    ).toBeInTheDocument()
    expect(create).toBeDisabled()
  })

  it("says which failures move on to a backup", async () => {
    mockApi()
    await renderDialog("fallback")

    expect(
      dialog().getByText(
        "Tried in order when the main model fails or is unavailable before it starts answering, including a rate limit, a missing or rejected provider key, or a request it refuses as invalid. Not once the answer has started streaming or a tool call is under way, or when the remaining budget cannot cover a pricier backup.",
      ),
    ).toBeInTheDocument()
  })

  it("previews where a request goes in one sentence", async () => {
    mockApi()
    const user = userEvent.setup()
    await renderDialog("fallback")

    const preview = screen.getByRole("region", { name: "Preview" })
    expect(preview).toHaveTextContent(
      "Pick a main model to see where requests go.",
    )

    await user.type(screen.getByRole("textbox", { name: /^name$/i }), "fast")
    await typeInto(user, /^main model$/i, "anthropic:claude-sonnet-5-5")
    await typeInto(user, /^backup 1$/i, "openai:gpt-5")
    await user.click(dialog().getByRole("button", { name: "Add a backup" }))
    await typeInto(user, /^backup 2$/i, "anthropic:claude-haiku-4-5")

    // Named from the catalog once it answers for each selector.
    await waitFor(() =>
      expect(preview).toHaveTextContent(
        "Requests to fast go to Claude Sonnet 5.5 on Anthropic. If that fails: GPT-5 on OpenAI, then Claude Haiku 4.5 on Anthropic.",
      ),
    )
  })

  it("previews a split in one sentence too", async () => {
    mockApi()
    const user = userEvent.setup()
    await renderDialog("weighted")

    await user.type(screen.getByRole("textbox", { name: /^name$/i }), "even")
    expect(screen.getByRole("region", { name: "Preview" })).toHaveTextContent(
      "Requests to even are split across 2 models by weight.",
    )
  })

  it("opens on the choice when no kind is given, and continues to it", async () => {
    mockApi()
    const user = userEvent.setup()
    await renderDialog()

    expect(screen.getByRole("dialog")).toHaveAccessibleName("New policy")
    await user.click(
      dialog().getByRole("radio", { name: "Tier down on budget" }),
    )
    await user.click(dialog().getByRole("button", { name: "Continue" }))
    expect(screen.getByRole("dialog")).toHaveAccessibleName(
      "New budget tier-down",
    )
    expect(
      screen.getByRole("textbox", { name: /budget used at least/i }),
    ).toHaveValue("80")
  })

  it("closes once the policy is created", async () => {
    const { writes } = mockApi()
    const user = userEvent.setup()
    const { onClose } = await renderDialog("fallback")

    await user.type(screen.getByRole("textbox", { name: /^name$/i }), "fast")
    await typeInto(user, /^main model$/i, "openai:gpt-5")
    await typeInto(user, /^backup 1$/i, "anthropic:claude-haiku-4-5")
    await user.click(dialog().getByRole("button", { name: "Create policy" }))
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce())
    expect(writes).toHaveLength(1)
  })
})
