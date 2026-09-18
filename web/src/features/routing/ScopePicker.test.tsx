import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { User } from "@/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { ScopePicker } from "./ScopePicker"

const USERS = [
  { user_id: "u1", alias: "ada" },
  { user_id: "u2", alias: "grace" },
] as unknown as User[]

// The scoped branch mounts `UserMultiSelect`, which runs its own roster query
// and reads the deployment. Mocked at the transport, per testing.md, so the
// picker's own query and surface gate still run.
function mockRoster() {
  return vi
    .spyOn(globalThis, "fetch")
    .mockResolvedValue(
      new Response(JSON.stringify({ data: [], count: 0 }), { status: 200 }),
    )
}

function renderPicker(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>{ui}</QueryClientProvider>
    </DeploymentProvider>,
  )
}

function setup(overrides: Partial<Parameters<typeof ScopePicker>[0]> = {}): {
  onChange: ReturnType<typeof vi.fn>
} {
  const onChange = vi.fn()
  renderPicker(
    <ScopePicker
      userIds={null}
      users={USERS}
      onChange={onChange}
      isSettled={false}
      {...overrides}
    />,
  )
  return { onChange }
}

describe("ScopePicker", () => {
  beforeEach(() => mockRoster())
  afterEach(() => vi.restoreAllMocks())

  it("reads as every caller when there is no scope", () => {
    setup({ userIds: null })
    expect(
      screen.getByRole("button", { name: "Every caller" }),
    ).toHaveAttribute("aria-pressed", "true")
  })

  it("reads as scoped for an empty list, which is not every caller", () => {
    // The three-state distinction the component exists for: `[]` is scoped with
    // nobody chosen yet, and showing it as "Every caller" would describe a
    // policy the form will not submit as the one policy it cannot be.
    setup({ userIds: [] })
    expect(
      screen.getByRole("button", { name: "Specific users" }),
    ).toHaveAttribute("aria-pressed", "true")
  })

  it("opens the user picker only once scoped", () => {
    const { unmount } = renderPicker(
      <ScopePicker
        userIds={null}
        users={USERS}
        onChange={vi.fn()}
        isSettled={false}
      />,
    )
    expect(screen.queryByLabelText("Users")).not.toBeInTheDocument()
    unmount()

    setup({ userIds: [] })
    expect(screen.getByLabelText("Users")).toBeInTheDocument()
  })

  it("starts a scope with nobody in it rather than everybody", async () => {
    const user = userEvent.setup()
    const { onChange } = setup({ userIds: null })
    await user.click(screen.getByRole("button", { name: "Specific users" }))
    expect(onChange).toHaveBeenCalledWith([])
  })

  it("ignores a press on the state it is already in", async () => {
    // Pressing the active tab would otherwise report a fresh `[]` and throw
    // away whoever had been chosen under it.
    const user = userEvent.setup()
    const { onChange } = setup({ userIds: ["u1"] })
    await user.click(screen.getByRole("button", { name: "Specific users" }))
    expect(onChange).not.toHaveBeenCalled()
  })

  it("withholds the controls once a write under this name has landed", () => {
    setup({ userIds: ["u1"], isSettled: true })
    expect(
      screen.queryByRole("button", { name: "Every caller" }),
    ).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Users")).not.toBeInTheDocument()
    expect(
      screen.getByText(/who it applies to is fixed now/),
    ).toBeInTheDocument()
  })
})
