import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { DeploymentBootstrap, OrganizationMember, User } from "@/client"
import { UserComboBox } from "@/features/users/UserComboBox"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationMember } from "@/tests/fixtures"

// Only the fields the picker reads; the rest of User is irrelevant here.
function user(user_id: string, alias: string | null = null): User {
  return { user_id, alias } as User
}

// A roster row for the owner id it bills through, which is the join the picker
// reads to put a name on that id.
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

// Mocked at the transport rather than at the hook, so the component's own query,
// its key and its surface gate all run.
function mockRoster(members: OrganizationMember[], status = 200) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const isRoster = String(input).includes(
      `${API_ROOT}/organizations/me/members`,
    )
    const body = isRoster
      ? { data: members, count: members.length }
      : { detail: "not mocked" }
    return new Response(JSON.stringify(body), {
      status: isRoster ? status : 501,
      headers: { "Content-Type": "application/json" },
    })
  })
}

function renderBox(
  ui: ReactElement,
  deployment: DeploymentBootstrap = bootstrap(),
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <DeploymentProvider value={deployment}>
      <QueryClientProvider client={client}>{ui}</QueryClientProvider>
    </DeploymentProvider>,
  )
}

const UUID = "33333333-3333-3333-3333-333333333333"

describe("UserComboBox", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("caps its width so the field and dropdown trigger stay within reach", () => {
    mockRoster([])
    renderBox(<UserComboBox value="" onChange={() => {}} users={[]} />)

    // The field is bounded rather than stretching across the whole form (#328).
    const field = screen.getByRole("combobox").closest(".max-w-md")
    expect(field).toBeInTheDocument()
  })

  it("says why the popover is empty rather than opening a silent box", async () => {
    mockRoster([])
    renderBox(<UserComboBox value="" onChange={() => {}} users={[]} />)

    await userEvent.click(screen.getByRole("combobox"))

    // Neither sentence promises what typing an id will do: that differs per
    // endpoint, and the caption line under the field is where it is answered.
    expect(
      await screen.findByText("No users to pick from yet."),
    ).toBeInTheDocument()
  })

  it("names a member by the roster, with their owner id as the second line", async () => {
    mockRoster([member(UUID, "Alice Example")])
    renderBox(
      <UserComboBox
        value=""
        onChange={() => {}}
        users={[user(UUID, "alice@example.com"), user("ci-bot")]}
      />,
    )

    await userEvent.click(screen.getByRole("combobox"))

    // The name leads and the id follows it as a muted line, which is what the
    // row's accessible name carries too: without the roster this option read as
    // "33333333-… (alice@example.com)", which nobody can pick out of a list.
    const row = await screen.findByRole("option", {
      name: `Alice Example (${UUID})`,
    })
    expect(within(row).getByText("Alice Example")).toBeInTheDocument()
    expect(within(row).getByText(UUID)).toBeInTheDocument()
    // A hand-made owner is already its own name, so it is left exactly as it was
    // rather than given a hint that repeats the label.
    expect(screen.getByRole("option", { name: "ci-bot" })).toBeInTheDocument()
  })

  it("finds a member by the id they are billed under, not only by name", async () => {
    mockRoster([member(UUID, "Alice Example")])
    renderBox(
      <UserComboBox
        value=""
        onChange={() => {}}
        users={[user(UUID), user("ci-bot")]}
      />,
    )

    await userEvent.type(screen.getByRole("combobox"), UUID)

    // The id an operator pastes out of a log or an API response still reaches
    // the person, even though the row no longer leads with it.
    expect(
      await screen.findByRole("option", { name: `Alice Example (${UUID})` }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("option", { name: "ci-bot" })).toBeNull()
  })

  it("falls back to the bare id when the roster cannot be read", async () => {
    mockRoster([], 500)
    renderBox(
      <UserComboBox value="" onChange={() => {}} users={[user(UUID)]} />,
    )

    await userEvent.click(screen.getByRole("combobox"))

    // A failed roster read leaves the owner as findable as it was before names
    // existed, rather than an empty row.
    expect(
      await screen.findByRole("option", { name: UUID }),
    ).toBeInTheDocument()
  })

  it("asks for no roster on a deployment that hosts none", async () => {
    const fetchMock = mockRoster([member(UUID, "Alice Example")])
    renderBox(
      <UserComboBox value="" onChange={() => {}} users={[user(UUID)]} />,
      bootstrap({ surfaces: [] }),
    )

    await userEvent.click(screen.getByRole("combobox"))

    expect(
      await screen.findByRole("option", { name: UUID }),
    ).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("sorts members ahead of the ids nobody named", async () => {
    mockRoster([member(UUID, "Zoe Example")])
    renderBox(
      <UserComboBox
        value=""
        onChange={() => {}}
        users={[user("aaa-bot"), user(UUID)]}
      />,
    )

    await userEvent.click(screen.getByRole("combobox"))
    await screen.findByRole("option", { name: `Zoe Example (${UUID})` })

    // Alphabetically "aaa-bot" wins; by relevance the member does, because a
    // member is who someone means when issuing a key. The member's row reads as
    // its two lines, name then id; the bot's is one line.
    const rows = screen.getAllByRole("option").map((o) => o.textContent)
    expect(rows).toEqual([`Zoe Example${UUID}`, "aaa-bot"])
  })

  it("submits the owner id, not the label shown for it", async () => {
    mockRoster([member(UUID, "Alice Example")])
    const changes: string[] = []
    renderBox(
      <UserComboBox
        value=""
        onChange={(id) => changes.push(id)}
        users={[user(UUID, "alice@example.com")]}
      />,
    )

    await userEvent.click(screen.getByRole("combobox"))
    await userEvent.click(
      await screen.findByRole("option", { name: `Alice Example (${UUID})` }),
    )

    // The value that reaches POST /v1/keys has to be the id. Submitting the
    // label would have the keys API create a second user called "Alice Example".
    expect(changes.at(-1)).toBe(UUID)
  })

  it("submits a typed id even when it is another owner's roster name", async () => {
    // The collision that used to need a tie-break rule: this member's roster
    // name is exactly another user's id, and a member sorts ahead of it. What
    // answers it now is that typed text is not looked up at all, so there is
    // nothing for the two rows to compete over.
    mockRoster([member(UUID, "ci-bot")])
    const changes: string[] = []
    renderBox(
      <UserComboBox
        value=""
        onChange={(id) => changes.push(id)}
        users={[user(UUID), user("ci-bot")]}
      />,
    )

    await userEvent.type(screen.getByRole("combobox"), "ci-bot")

    expect(changes.at(-1)).toBe("ci-bot")
  })

  it("submits the picked member, not the owner whose id is their roster name", async () => {
    // The same collision as above, reached from the other side: the row that was
    // picked is the member, and their label is the other owner's id.
    mockRoster([member(UUID, "ci-bot")])
    const changes: string[] = []
    renderBox(
      <UserComboBox
        value=""
        onChange={(id) => changes.push(id)}
        users={[user(UUID), user("ci-bot")]}
      />,
    )

    await userEvent.click(screen.getByRole("combobox"))
    await userEvent.click(
      await screen.findByRole("option", { name: `ci-bot (${UUID})` }),
    )

    // The member, not the `ci-bot` owner. A picked row reports its own id, so
    // the row that was clicked is the only answer this can have.
    expect(changes.at(-1)).toBe(UUID)
  })

  it("submits the member that was picked when two of them share a name", async () => {
    const first = "11111111-1111-1111-1111-111111111111"
    const second = "22222222-2222-2222-2222-222222222222"
    // Two people, one name. A roster carries no uniqueness rule over
    // `full_name`, so this is ordinary rather than a corner case.
    mockRoster([member(first, "Alex Smith"), member(second, "Alex Smith")])
    const changes: string[] = []
    renderBox(
      <UserComboBox
        value=""
        onChange={(id) => changes.push(id)}
        users={[user(first), user(second)]}
      />,
    )

    await userEvent.click(screen.getByRole("combobox"))
    // One name on screen twice, told apart by the id under it, which is the
    // whole reason the hint joins the accessible name.
    expect(await screen.findAllByText("Alex Smith")).toHaveLength(2)
    await userEvent.click(
      screen.getByRole("option", { name: `Alex Smith (${second})` }),
    )

    // Resolving the label back to a row would always answer with the first one,
    // so the second person could never be picked.
    expect(changes.at(-1)).toBe(second)
  })

  it("does not turn a typed display name into the id it names", async () => {
    mockRoster([member(UUID, "Alice Example")])
    const changes: string[] = []
    renderBox(
      <UserComboBox
        value=""
        onChange={(id) => changes.push(id)}
        users={[user(UUID)]}
      />,
    )

    await userEvent.type(screen.getByRole("combobox"), "Alice Example")

    // Deliberate, and the honest version of the two collisions above: picking
    // the row is how an existing person is chosen, and a typed name is an id of
    // its own. Resolving it to Alice's UUID would be a guess, since a roster
    // puts no uniqueness rule on a name.
    expect(changes.at(-1)).toBe("Alice Example")
    expect(changes).not.toContain(UUID)
  })
})
