import { Outlet, useNavigate } from "@tanstack/react-router"
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { Button } from "@/design-system/actions/Button"
import { MemberFirstRun } from "@/features/home/MemberFirstRun"
import { CreateKeyHost } from "@/features/keys/CreateKeyHost"
import { useOpenCreateKey } from "@/features/keys/openCreateKey"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { apiKey, bootstrap, organizationContext } from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"
import { recordEvent } from "@/tests/telemetry"

// The seam is the boundary: the host decides what to record, this hears it.
vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
  const { telemetrySpy } = await import("@/tests/telemetry")
  return { useTelemetry: vi.fn(() => telemetrySpy) }
})
const telemetry = { recordEvent }

describe("the Create API key dialog's entry points", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    telemetry.recordEvent.mockClear()
  })

  it("records the member first-run button under its own value", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const path = String(input)
      if (path === `${API_ROOT}/organizations/me`) {
        return Response.json(organizationContext({ role: "member" }))
      }
      if (path.includes("/aliases")) return Response.json([])
      return Response.json({ data: [], count: 0 })
    })
    const user = userEvent.setup()
    await renderWithRouter(
      <Provider>
        <DeploymentProvider value={bootstrap()}>
          <SelectedWorkspaceProvider>
            <CreateKeyHost>
              <MemberFirstRun workspace="Platform" canCreateKey />
            </CreateKeyHost>
          </SelectedWorkspaceProvider>
        </DeploymentProvider>
      </Provider>,
    )
    expect(telemetry.recordEvent).not.toHaveBeenCalled()
    await user.click(
      await screen.findByRole("button", { name: "Create API key" }),
    )
    // The first mount of the dialog in this file is cold: past findBy's one
    // second on a loaded runner.
    await screen.findByRole(
      "dialog",
      { name: "Create API key" },
      { timeout: 5_000 },
    )
    expect(telemetry.recordEvent).toHaveBeenCalledTimes(1)
    expect(telemetry.recordEvent).toHaveBeenCalledWith(
      "Create API Key Opened",
      {
        entry_point: "member_first_run",
      },
    )
  })
})

const SECRET = "gw-NEWSECRET0000000000000000000000000000000000000000000000"

/** A page with one opener and a way to leave, under the real host and dialog. */
function Opener({
  go,
}: {
  go: { current: ((to: string) => void) | undefined }
}) {
  const open = useOpenCreateKey()
  const navigate = useNavigate()
  go.current = (to) => void navigate({ to })
  return <Button onPress={() => open("sidebar")}>Make a key</Button>
}

async function renderOpener() {
  const creates: unknown[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    if (path === `${API_ROOT}/organizations/me`) {
      return Response.json(
        organizationContext({ role: "member", deployment_operator: false }),
      )
    }
    if (method === "POST" && /\/keys$/.test(path)) {
      creates.push(JSON.parse(String(init?.body ?? "{}")))
      // Slower than two presses in one task, so the second lands in flight.
      await new Promise((resolve) => setTimeout(resolve, 30))
      return Response.json({
        ...apiKey({ id: `key-${creates.length}`, key_name: "k" }),
        key: SECRET,
      })
    }
    if (path.includes("/aliases")) return Response.json([])
    return Response.json({ data: [], count: 0 })
  })
  const go: { current: ((to: string) => void) | undefined } = {
    current: undefined,
  }
  // The host lives in the shell, above the routes, so a navigation does not
  // unmount it.
  await renderWithRouter(<div />, {
    shell: (
      <Provider>
        <DeploymentProvider value={bootstrap()}>
          <SelectedWorkspaceProvider>
            <CreateKeyHost>
              <Opener go={go} />
              <Outlet />
            </CreateKeyHost>
          </SelectedWorkspaceProvider>
        </DeploymentProvider>
      </Provider>
    ),
  })
  return { creates, go }
}

async function openDialog(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("button", { name: "Make a key" }))
  const dialog = await screen.findByRole(
    "dialog",
    { name: "Create API key" },
    { timeout: 5_000 },
  )
  await waitFor(() =>
    expect(
      within(dialog).getByRole("button", { name: "Create API key" }),
    ).toBeEnabled(),
  )
  return dialog
}

describe("the Create API key dialog, once a key exists", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("gives focus back to the control that opened it after the secret is saved", async () => {
    const user = userEvent.setup()
    await renderOpener()
    const dialog = await openDialog(user)
    await user.click(
      within(dialog).getByRole("button", { name: "Create API key" }),
    )
    const created = await screen.findByRole("dialog", {
      name: "API key created",
    })
    await user.click(
      within(created).getByRole("button", { name: "I\u2019ve saved this key" }),
    )
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Make a key" })).toHaveFocus(),
    )
  })

  it("mints one key for two presses that land before the first settles", async () => {
    const user = userEvent.setup()
    const { creates } = await renderOpener()
    const dialog = await openDialog(user)
    const submit = within(dialog).getByRole("button", {
      name: "Create API key",
    })
    act(() => {
      fireEvent.click(submit)
      fireEvent.click(submit)
    })
    await screen.findByRole("dialog", { name: "API key created" })
    expect(creates).toHaveLength(1)
  })

  it("keeps the secret on screen when a navigation lands under it", async () => {
    const user = userEvent.setup()
    const { go } = await renderOpener()
    const dialog = await openDialog(user)
    await user.click(
      within(dialog).getByRole("button", { name: "Create API key" }),
    )
    await screen.findByRole("dialog", { name: "API key created" })
    act(() => go.current?.("/elsewhere"))
    await act(async () => {})
    expect(
      screen.getByRole("dialog", { name: "API key created" }),
    ).toBeVisible()
  })

  it("still closes the form, with nothing to lose, when a navigation lands under it", async () => {
    const user = userEvent.setup()
    const { go } = await renderOpener()
    await openDialog(user)
    act(() => go.current?.("/elsewhere"))
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Create API key" }),
      ).toBeNull(),
    )
  })
})
