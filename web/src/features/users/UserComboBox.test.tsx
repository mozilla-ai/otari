import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it } from "vitest"

import type { User } from "@/client"
import { UserComboBox } from "@/features/users/UserComboBox"

// Only the fields the picker reads; the rest of User is irrelevant here.
function user(user_id: string, alias: string | null = null): User {
  return { user_id, alias } as User
}

describe("UserComboBox", () => {
  it("caps its width so the field and dropdown trigger stay within reach", () => {
    render(<UserComboBox value="" onChange={() => {}} users={[]} />)

    // The field is bounded rather than stretching across the whole form (#328).
    const field = screen.getByRole("combobox").closest(".max-w-md")
    expect(field).toBeInTheDocument()
  })

  it("says why the popover is empty rather than opening a silent box", async () => {
    render(<UserComboBox value="" onChange={() => {}} users={[]} />)

    await userEvent.click(screen.getByRole("combobox"))

    // Neither sentence promises what typing an id will do: that differs per
    // endpoint, and the caption line under the field is where it is answered.
    expect(
      await screen.findByText("No users to pick from yet."),
    ).toBeInTheDocument()
  })

  it("names a member by the roster instead of the UUID they were minted under", async () => {
    const uuid = "33333333-3333-3333-3333-333333333333"
    render(
      <UserComboBox
        value=""
        onChange={() => {}}
        users={[user(uuid, "alice@example.com"), user("ci-bot")]}
        memberLabels={new Map([[uuid, "Alice Example"]])}
      />,
    )

    await userEvent.click(screen.getByRole("combobox"))

    // The member reads as a person. Without the map this option would render as
    // "33333333-… (alice@example.com)", which nobody can pick out of a list.
    expect(
      screen.getByRole("option", { name: "Alice Example" }),
    ).toBeInTheDocument()
    // A hand-made owner is already readable, so it is left exactly as it was.
    expect(screen.getByRole("option", { name: "ci-bot" })).toBeInTheDocument()
  })

  it("sorts members ahead of the ids nobody named", async () => {
    const uuid = "33333333-3333-3333-3333-333333333333"
    render(
      <UserComboBox
        value=""
        onChange={() => {}}
        users={[user("aaa-bot"), user(uuid, "zoe@example.com")]}
        memberLabels={new Map([[uuid, "Zoe Example"]])}
      />,
    )

    await userEvent.click(screen.getByRole("combobox"))

    // Alphabetically "aaa-bot" wins; by relevance the member does, because a
    // member is who someone means when issuing a key.
    const options = screen.getAllByRole("option").map((o) => o.textContent)
    expect(options).toEqual(["Zoe Example", "aaa-bot"])
  })

  it("submits the owner id, not the label shown for it", async () => {
    const uuid = "33333333-3333-3333-3333-333333333333"
    const changes: string[] = []
    render(
      <UserComboBox
        value=""
        onChange={(id) => changes.push(id)}
        users={[user(uuid, "alice@example.com")]}
        memberLabels={new Map([[uuid, "Alice Example"]])}
      />,
    )

    await userEvent.click(screen.getByRole("combobox"))
    await userEvent.click(screen.getByRole("option", { name: "Alice Example" }))

    // The value that reaches POST /v1/keys has to be the id. Submitting the
    // label would have the keys API create a second user called "Alice Example".
    expect(changes.at(-1)).toBe(uuid)
  })

  it("submits a typed id even when it is another owner's roster name", async () => {
    const uuid = "33333333-3333-3333-3333-333333333333"
    const changes: string[] = []
    render(
      <UserComboBox
        value=""
        onChange={(id) => changes.push(id)}
        users={[user(uuid), user("ci-bot")]}
        // The collision: this member's roster name is exactly another user's id,
        // and a member sorts ahead of it. Matching either field in one scan would
        // resolve the typed id to the member's UUID and bill the key to them.
        memberLabels={new Map([[uuid, "ci-bot"]])}
      />,
    )

    await userEvent.type(screen.getByRole("combobox"), "ci-bot")

    expect(changes.at(-1)).toBe("ci-bot")
  })

  it("submits the picked member, not the owner whose id is their roster name", async () => {
    const uuid = "33333333-3333-3333-3333-333333333333"
    const changes: string[] = []
    render(
      <UserComboBox
        value=""
        onChange={(id) => changes.push(id)}
        users={[user(uuid), user("ci-bot")]}
        // The same collision as above, reached from the other side: the row that
        // was picked is the member, and their label is the other owner's id.
        memberLabels={new Map([[uuid, "ci-bot"]])}
      />,
    )

    await userEvent.click(screen.getByRole("combobox"))
    await userEvent.click(screen.getAllByRole("option", { name: "ci-bot" })[0])

    // The member, not the `ci-bot` owner. This is what the display-text echo
    // would get wrong: the label reported after the id resolves by id first, and
    // an id match on "ci-bot" is a different row than the one that was clicked.
    expect(changes.at(-1)).toBe(uuid)
  })

  it("submits the member that was picked when two of them share a name", async () => {
    const first = "11111111-1111-1111-1111-111111111111"
    const second = "22222222-2222-2222-2222-222222222222"
    const changes: string[] = []
    render(
      <UserComboBox
        value=""
        onChange={(id) => changes.push(id)}
        users={[user(first), user(second)]}
        // Two people, one name. A roster carries no uniqueness rule over
        // `full_name`, so this is ordinary rather than a corner case.
        memberLabels={
          new Map([
            [first, "Alex Smith"],
            [second, "Alex Smith"],
          ])
        }
      />,
    )

    await userEvent.click(screen.getByRole("combobox"))
    const rows = screen.getAllByRole("option", { name: "Alex Smith" })
    expect(rows).toHaveLength(2)
    await userEvent.click(rows[1])

    // Resolving the label back to a row would always answer with the first one,
    // so the second person could never be picked.
    expect(changes.at(-1)).toBe(second)
  })
})
