import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { RoutingPage } from "@/features/routing/RoutingPage"
import {
  CHAIN,
  createTrigger,
  MEMBERS,
  mockApi,
  nameAndServe,
  pickUsers,
  policy,
  policyWrites,
  renderPage,
  submitDialog,
} from "@/tests/routing"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("RoutingPage policy scopes", () => {
  it("writes exactly one unscoped policy for every caller", async () => {
    // The default tab, asserted rather than assumed: "every caller" and "scoped
    // but nobody chosen" are two states, and only this one is one write.
    const { calls } = mockApi([], null, [], { members: MEMBERS })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await nameAndServe(user, "cheap")
    await submitDialog(user)

    const writes = policyWrites(calls)
    expect(writes).toHaveLength(1)
    expect(writes[0].user_id ?? null).toBeNull()
  })

  it("writes one policy per chosen user, each carrying its own scope", async () => {
    // A policy's key is its name plus its user, so N people are N rows of the
    // same name and spec. There is no batch endpoint; this is the whole design.
    const { calls } = mockApi([], null, [], { members: MEMBERS })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await nameAndServe(user, "cheap")
    await pickUsers(user, ["bob", "carol"])
    await submitDialog(user)

    const writes = policyWrites(calls)
    expect(writes).toHaveLength(2)
    expect(writes.map((write) => write.user_id)).toEqual(["u-bob", "u-carol"])
    // Same name and same spec on each: only the scope differs.
    expect(new Set(writes.map((write) => write.name))).toEqual(
      new Set(["cheap"]),
    )
    for (const write of writes) {
      expect(write.spec.select).toEqual([{ default: "openai:gpt-5-nano" }])
    }
  })

  it("shows the chosen people by name above the input, not as bare owner ids", async () => {
    mockApi([], null, [], { members: MEMBERS })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await nameAndServe(user, "cheap")
    await pickUsers(user, ["bob"])

    const dialog = within(screen.getByRole("dialog"))
    // The roster names this one, so the chip says the person rather than the id
    // the request plane bills to.
    expect(
      dialog.getByRole("button", { name: "Remove Bob Builder" }),
    ).toBeInTheDocument()
  })

  it("will not submit a scoped policy with nobody chosen", async () => {
    const { calls } = mockApi([], null, [], { members: MEMBERS })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await nameAndServe(user, "cheap")
    const dialog = within(screen.getByRole("dialog"))
    await user.click(dialog.getByRole("button", { name: "Specific users" }))

    expect(dialog.getByRole("button", { name: "Create policy" })).toBeDisabled()
    await submitDialog(user)
    expect(policyWrites(calls)).toHaveLength(0)
  })

  it("keeps the dialog open on a part-written save, naming who landed and who did not", async () => {
    // Three writes with no transaction over them, so a refusal partway leaves
    // the earlier rows in place. Saying "it failed" would leave the operator to
    // work out which of the three exist by reading the table. The refusal is in
    // the middle, so this also pins that the writes after it are still
    // attempted rather than one conflict standing in for everyone behind it.
    const { calls } = mockApi([], null, [], {
      members: MEMBERS,
      refuseFirstWriteFor: ["u-carol"],
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await nameAndServe(user, "cheap")
    await pickUsers(user, ["bob", "carol", "alice"])
    await submitDialog(user)

    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("Created for Bob Builder, alice (alice).")
    // By its id, since nobody named this one, and with the refusal's own reason
    // rather than a swallowed "something went wrong".
    expect(alert).toHaveTextContent("Not created for u-carol (carol)")
    expect(alert).toHaveTextContent("No room for u-carol")
    expect(screen.getByRole("dialog")).toBeInTheDocument()

    // Pressing again rewrites nothing that already landed.
    await submitDialog(user)
    const scopes = policyWrites(calls).map((write) => write.user_id)
    expect(scopes).toEqual(["u-bob", "u-carol", "alice", "u-carol"])
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("fixes who a policy applies to once a write for somebody has landed", async () => {
    // Nothing here can take a policy back, so a scope that can still be edited
    // after a partial write is a way to end up with rows the operator did not
    // ask for: drop a written person from the selection and their policy lives
    // on unmentioned, or switch to every caller and it lives on ALSO outranking
    // the global one for exactly them, which is the precedence the field's own
    // description promises. The controls are withheld instead, and say why.
    mockApi([], null, [], {
      members: MEMBERS,
      refuseFirstWriteFor: ["u-carol"],
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await nameAndServe(user, "cheap")
    await pickUsers(user, ["bob", "carol"])
    await submitDialog(user)
    await screen.findByRole("alert")

    const dialog = within(screen.getByRole("dialog"))
    // Neither tab, and no picker: the selection is settled.
    expect(
      dialog.queryByRole("button", { name: "Specific users" }),
    ).not.toBeInTheDocument()
    expect(
      dialog.queryByRole("button", { name: "Every caller" }),
    ).not.toBeInTheDocument()
    expect(dialog.queryByLabelText("Users")).not.toBeInTheDocument()
    // And the reason, since a control that vanishes without one teaches
    // nothing. The route back out is naming the list, not this form.
    expect(
      dialog.getByText(/already been created/, { exact: false }),
    ).toHaveTextContent("delete it from the list")
  })

  it("writes every scope again when the payload changed after a part-written save", async () => {
    // The ids that landed are remembered against the payload they landed under,
    // so correcting the name first is not the same policy: skipping them then
    // would leave the corrected one unwritten for exactly the people the old
    // one already reached.
    const { calls } = mockApi([], null, [], {
      members: MEMBERS,
      refuseFirstWriteFor: ["u-carol"],
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await nameAndServe(user, "cheap")
    await pickUsers(user, ["bob", "carol"])
    await submitDialog(user)
    expect(await screen.findByRole("alert")).toBeInTheDocument()

    await user.type(screen.getByRole("textbox", { name: /policy name/i }), "er")
    await submitDialog(user)

    const writes = policyWrites(calls)
    expect(writes.map((write) => [write.name, write.user_id])).toEqual([
      ["cheap", "u-bob"],
      ["cheap", "u-carol"],
      ["cheaper", "u-bob"],
      ["cheaper", "u-carol"],
    ])
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("lists a row per user for a policy written to several of them", async () => {
    mockApi(
      [
        policy("cheap", CHAIN, { user_id: "u-bob" }),
        policy("cheap", CHAIN, { user_id: "u-carol" }),
      ],
      null,
      [],
      { members: MEMBERS },
    )
    renderPage(<RoutingPage />)

    // Two rows under one name: the scope is half the row's identity, so they do
    // not collapse into one.
    expect(
      await screen.findAllByRole("rowheader", { name: "cheap" }),
    ).toHaveLength(2)
  })
})
