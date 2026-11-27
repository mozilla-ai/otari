import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { ApiKey } from "@/client"
import { EditKeyForm } from "@/features/keys/EditKeyForm"
import { toDatetimeLocal } from "@/features/keys/ExpiryFields"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  apiKey,
  bootstrap,
  budget,
  organizationContext,
  organizationSpendCeiling,
  workspace,
  workspaceBudgetDefault,
} from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"
import { pickOption, selectTrigger } from "@/tests/select"

function json(body: unknown, status = 200, headers?: HeadersInit): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  })
}

const WORKSPACE_ID = "ws-data"
const KEY_ID = "key-1"

/**
 * An organization admin's view: two budgets, one of them the Data workspace's
 * per-person default, and the key capped by `capBudgetId` if given. Records
 * the key PATCH bodies and the spend-ceiling writes for assertions.
 */
function mockApi({
  capBudgetId,
  patchResponse,
  failCeilingWrites = false,
}: {
  capBudgetId?: string
  patchResponse?: Response
  failCeilingWrites?: boolean
} = {}) {
  const budgets = [
    budget({ budget_id: "b-default", name: "Each Data member, weekly" }),
    budget({ budget_id: "b-team", name: "Team standard" }),
  ]
  const ceilings = capBudgetId
    ? [
        organizationSpendCeiling({
          scope_type: "api_token",
          scope_id: KEY_ID,
          budget_id: capBudgetId,
          name: "Each Data member, weekly",
        }),
      ]
    : []
  const patchBodies: Record<string, unknown>[] = []
  const ceilingWrites: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    if (url === `${API_ROOT}/organizations/me`) {
      return json(
        organizationContext({ role: "admin", deployment_operator: false }),
      )
    }
    if (url.includes("/organizations/me/budgets")) {
      return json({ data: budgets, count: budgets.length })
    }
    if (url.includes("/organizations/me/spend-ceilings") && method !== "GET") {
      ceilingWrites.push(String(init?.body))
      if (failCeilingWrites) {
        return json({ detail: "budget ceiling refused" }, 500)
      }
      return json(
        organizationSpendCeiling({
          scope_type: "api_token",
          scope_id: KEY_ID,
          budget_id: "b-team",
        }),
        201,
      )
    }
    if (url.includes("/organizations/me/spend-ceilings")) {
      return json({ data: ceilings, count: ceilings.length })
    }
    if (url.includes(`/workspaces/${WORKSPACE_ID}/member-budget-policies`)) {
      const rows = [
        workspaceBudgetDefault({
          workspace_id: WORKSPACE_ID,
          budget_id: "b-default",
        }),
      ]
      return json({ data: rows, count: rows.length })
    }
    if (url.startsWith(`${API_ROOT}/workspaces?`)) {
      return json({
        data: [workspace({ id: WORKSPACE_ID, name: "Data" })],
        count: 1,
      })
    }
    if (url === `${API_ROOT}/organizations/me/keys/${KEY_ID}`) {
      patchBodies.push(JSON.parse(String(init?.body)))
      // Cloned per call: a saved Response can only be read once, and a
      // retry test reads every answer it hands out.
      return patchResponse
        ? patchResponse.clone()
        : json(apiKey({ id: KEY_ID, key_name: "ci-bot" }))
    }
    return json([])
  })
  return { patchBodies, ceilingWrites }
}

async function openBudgetPicker() {
  const user = userEvent.setup()
  await renderWithRouter(<div />, {
    shell: (
      <Provider>
        <DeploymentProvider value={bootstrap()}>
          <SelectedWorkspaceProvider>
            <EditKeyForm
              isOpen
              isDeploymentWide={false}
              apiKey={apiKey({ id: KEY_ID, key_name: "ci-bot" })}
              onClose={() => undefined}
            />
          </SelectedWorkspaceProvider>
        </DeploymentProvider>
      </Provider>
    ),
  })
  const dialog = await screen.findByRole("dialog")
  const label = "Budget (optional)"
  await waitFor(() => expect(selectTrigger(label, dialog)).toBeEnabled())
  await user.click(selectTrigger(label, dialog))
  return dialog
}

describe("EditKeyForm's budget picker", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("does not offer a budget that backs a workspace's per-person default", async () => {
    mockApi()
    await openBudgetPicker()

    expect(
      await screen.findByRole("option", { name: /^Team standard/ }),
    ).toBeVisible()
    expect(
      screen.queryByRole("option", { name: /^Each Data member/ }),
    ).not.toBeInTheDocument()
  })

  it("still names the default a key is already capped by", async () => {
    mockApi({ capBudgetId: "b-default" })
    await openBudgetPicker()

    expect(
      await screen.findAllByRole("option", { name: /Each Data member/ }),
    ).toHaveLength(1)
  })
})

describe("EditKeyForm's expiry cap", () => {
  const CAPPED_TO = "2026-03-01T00:00:00+00:00"

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("stays open over the expiry the gateway clamped to, and says so", async () => {
    // The caller left the expiry blank, which on a capped deployment is an
    // expiry at the cap, not no expiry. Closing over that would show a key
    // the caller never asked for with no word about why.
    const onClose = vi.fn()
    mockApi({
      patchResponse: json(
        apiKey({ id: KEY_ID, key_name: "ci-bot", expires_at: CAPPED_TO }),
        200,
        { "Otari-Key-Lifetime-Capped": CAPPED_TO },
      ),
    })
    const user = userEvent.setup()
    await renderWithRouter(<div />, {
      shell: (
        <Provider>
          <DeploymentProvider value={bootstrap()}>
            <SelectedWorkspaceProvider>
              <EditKeyForm
                isOpen
                isDeploymentWide={false}
                apiKey={apiKey({ id: KEY_ID, key_name: "ci-bot" })}
                onClose={onClose}
              />
            </SelectedWorkspaceProvider>
          </DeploymentProvider>
        </Provider>
      ),
    })
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    expect(
      await screen.findByText(/This deployment caps key lifetime/),
    ).toBeVisible()
    expect(onClose).not.toHaveBeenCalled()
    // The field shows the expiry that was stored, not the blank that was sent.
    expect(within(dialog).getByLabelText("Expires")).toHaveValue(
      toDatetimeLocal(CAPPED_TO).split("T")[0],
    )
  })

  it("closes on a save the gateway did not clamp", async () => {
    const onClose = vi.fn()
    mockApi()
    const user = userEvent.setup()
    await renderWithRouter(<div />, {
      shell: (
        <Provider>
          <DeploymentProvider value={bootstrap()}>
            <SelectedWorkspaceProvider>
              <EditKeyForm
                isOpen
                isDeploymentWide={false}
                apiKey={apiKey({ id: KEY_ID, key_name: "ci-bot" })}
                onClose={onClose}
              />
            </SelectedWorkspaceProvider>
          </DeploymentProvider>
        </Provider>
      ),
    })
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() => expect(onClose).toHaveBeenCalled())
    expect(screen.queryByText(/This deployment caps key lifetime/)).toBeNull()
  })
})

describe("EditKeyForm's save flow", () => {
  const CAPPED_TO = "2026-03-01T00:00:00+00:00"

  afterEach(() => {
    vi.restoreAllMocks()
  })

  // Mutations retry three times by default, which is seconds of backoff
  // between a refused ceiling write and the error the form shows.
  async function renderForm({
    key = apiKey({ id: KEY_ID, key_name: "ci-bot" }),
    onClose = () => undefined,
  }: {
    key?: ApiKey
    onClose?: () => void
  } = {}) {
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    })
    await renderWithRouter(<div />, {
      shell: (
        <Provider>
          <QueryClientProvider client={client}>
            <DeploymentProvider value={bootstrap()}>
              <SelectedWorkspaceProvider>
                <EditKeyForm
                  isOpen
                  isDeploymentWide={false}
                  apiKey={key}
                  onClose={onClose}
                />
              </SelectedWorkspaceProvider>
            </DeploymentProvider>
          </QueryClientProvider>
        </Provider>
      ),
    })
    return screen.findByRole("dialog")
  }

  it("does not resend expires_at on a save that left the Expires field alone", async () => {
    // The PATCH treats a present expires_at as a rewrite, which the cap can
    // clamp or refuse, so renaming a key must not put its expiry in the body.
    const { patchBodies } = mockApi()
    const user = userEvent.setup()
    const dialog = await renderForm({
      key: apiKey({ id: KEY_ID, key_name: "ci-bot", expires_at: CAPPED_TO }),
    })
    await user.type(within(dialog).getByLabelText("Name"), "!")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() => expect(patchBodies).toHaveLength(1))
    expect(patchBodies[0]).not.toHaveProperty("expires_at")
  })

  it("sends expires_at once the Expires field was edited", async () => {
    const { patchBodies } = mockApi()
    const user = userEvent.setup()
    const dialog = await renderForm()
    fireEvent.change(within(dialog).getByLabelText("Expires"), {
      target: { value: "2027-01-15" },
    })
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() => expect(patchBodies).toHaveLength(1))
    expect(patchBodies[0].expires_at).toBeTypeOf("string")
  })

  it("keeps a failed budget change dirty over the clamped expiry, and retries it on the next save", async () => {
    // The key PATCH landed, so the clamped expiry and its note show even
    // though the ceiling write that follows fails. The cap change stays
    // dirty for the next Save to retry, and the stored expiry is not sent
    // again.
    const { patchBodies, ceilingWrites } = mockApi({
      patchResponse: json(
        apiKey({ id: KEY_ID, key_name: "ci-bot", expires_at: CAPPED_TO }),
        200,
        { "Otari-Key-Lifetime-Capped": CAPPED_TO },
      ),
      failCeilingWrites: true,
    })
    const onClose = vi.fn()
    const user = userEvent.setup()
    const dialog = await renderForm({ onClose })
    await waitFor(() =>
      expect(selectTrigger("Budget (optional)", dialog)).toBeEnabled(),
    )
    await pickOption(user, "Budget (optional)", /^Team standard/, dialog)
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    expect(
      await screen.findByText(/This deployment caps key lifetime/),
    ).toBeVisible()
    expect(within(dialog).getByLabelText("Expires")).toHaveValue(
      toDatetimeLocal(CAPPED_TO).split("T")[0],
    )
    expect(await screen.findByText("budget ceiling refused")).toBeVisible()
    expect(onClose).not.toHaveBeenCalled()
    await waitFor(() => expect(ceilingWrites).toHaveLength(1))

    await user.click(within(dialog).getByRole("button", { name: "Save" }))
    await waitFor(() => expect(ceilingWrites).toHaveLength(2))
    expect(patchBodies).toHaveLength(2)
    expect(patchBodies[1]).not.toHaveProperty("expires_at")
  })
})
