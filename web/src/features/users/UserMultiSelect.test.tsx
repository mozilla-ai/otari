import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { type ReactElement, useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationMember, User } from "@/client"
import { UserMultiSelect } from "@/features/users/UserMultiSelect"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationMember } from "@/tests/fixtures"

// Only the fields the picker reads; the rest of User is irrelevant here.
function user(user_id: string, alias: string | null = null): User {
  return { user_id, alias } as User
}

function member(
  attributionUserId: string,
  fullName: string,
): OrganizationMember {
  return organizationMember({
    organization_member_id: attributionUserId,
    attribution_user_id: attributionUserId,
    full_name: fullName,
  })
}

// The transport, not the hook: the component's own query and surface gate run.
function mockRoster(members: OrganizationMember[]) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const isRoster = String(input).includes(
      `${API_ROOT}/organizations/me/members`,
    )
    return new Response(
      JSON.stringify(
        isRoster
          ? { data: members, count: members.length }
          : { detail: "not mocked" },
      ),
      { status: isRoster ? 200 : 501 },
    )
  })
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

const UUID = "33333333-3333-3333-3333-333333333333"

// The picker is controlled, so a test about what a pick reports needs something
// holding the value the way the budget form does.
function Controlled({ users }: { users: User[] }) {
  const [value, setValue] = useState<string[]>([])
  return (
    <>
      <UserMultiSelect
        label="Applies to"
        value={value}
        onChange={setValue}
        users={users}
      />
      <p>selected: {value.join(", ")}</p>
    </>
  )
}

describe("UserMultiSelect", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("names a member by the roster, with their owner id as the second line", async () => {
    mockRoster([member(UUID, "Alice Example")])
    renderPicker(
      <UserMultiSelect
        label="Applies to"
        value={[]}
        onChange={() => {}}
        users={[user(UUID, "alice@example.com"), user("ci-bot")]}
      />,
    )

    // `menuTrigger="input"`, so the list opens on typing or on ArrowDown rather
    // than on a click.
    await userEvent.click(screen.getByLabelText("Add a person"))
    await userEvent.keyboard("{ArrowDown}")

    // The same shape the owner picker uses, so a person reads the same way in
    // both: name first, billing id under it and in the row's name.
    const row = await screen.findByRole("option", {
      name: `Alice Example (${UUID})`,
    })
    expect(within(row).getByText("Alice Example")).toBeInTheDocument()
    expect(within(row).getByText(UUID)).toBeInTheDocument()
    // An id an operator named over the API is already its own name.
    expect(screen.getByRole("option", { name: "ci-bot" })).toBeInTheDocument()
  })

  it("finds a member by the id they are billed under, not only by name", async () => {
    mockRoster([member(UUID, "Alice Example")])
    renderPicker(
      <UserMultiSelect
        label="Applies to"
        value={[]}
        onChange={() => {}}
        users={[user(UUID), user("ci-bot")]}
      />,
    )

    await userEvent.type(screen.getByLabelText("Add a person"), UUID)

    expect(
      await screen.findByRole("option", { name: `Alice Example (${UUID})` }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("option", { name: "ci-bot" })).toBeNull()
  })

  it("reports the owner id of the person picked, and chips them by name", async () => {
    mockRoster([member(UUID, "Alice Example")])
    renderPicker(<Controlled users={[user(UUID)]} />)

    // `menuTrigger="input"`, so the list opens on typing or on ArrowDown rather
    // than on a click.
    await userEvent.click(screen.getByLabelText("Add a person"))
    await userEvent.keyboard("{ArrowDown}")
    await userEvent.click(
      await screen.findByRole("option", { name: `Alice Example (${UUID})` }),
    )

    // The id is what a budget assignment PATCHes; the chip is what the operator
    // reads back, and it is their name rather than that id.
    expect(await screen.findByText(`selected: ${UUID}`)).toBeInTheDocument()
    // The open popover aria-hides the rest of the form, chips included.
    await userEvent.keyboard("{Escape}")
    expect(
      screen.getByRole("button", { name: "Remove Alice Example" }),
    ).toBeInTheDocument()
  })
})
