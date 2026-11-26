import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
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
import { selectTrigger } from "@/tests/select"

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
 * per-person default, and the key capped by `capBudgetId` if given.
 */
function mockApi({
  capBudgetId,
  patchResponse,
}: {
  capBudgetId?: string
  patchResponse?: Response
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
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url === `${API_ROOT}/organizations/me`) {
      return json(
        organizationContext({ role: "admin", deployment_operator: false }),
      )
    }
    if (url.includes("/organizations/me/budgets")) {
      return json({ data: budgets, count: budgets.length })
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
      return patchResponse ?? json(apiKey({ id: KEY_ID, key_name: "ci-bot" }))
    }
    return json([])
  })
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
