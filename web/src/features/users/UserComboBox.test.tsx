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
})
