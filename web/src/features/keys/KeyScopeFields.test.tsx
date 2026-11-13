import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it } from "vitest"

import type { KeyScope } from "@/client"
import { heldScopes, KeyScopeFields } from "@/features/keys/KeyScopeFields"

function renderFields(initial: KeyScope[], isServiceKey = false) {
  const seen: KeyScope[][] = []
  function Harness() {
    const [value, setValue] = useState(initial)
    return (
      <KeyScopeFields
        value={value}
        onChange={(next) => {
          seen.push(next)
          setValue(next)
        }}
        isServiceKey={isServiceKey}
      />
    )
  }
  render(<Harness />)
  return seen
}

describe("heldScopes", () => {
  it("orders the scopes as the gateway does, once each", () => {
    expect(
      heldScopes(["usage:read", "inference", "usage:read"], false),
    ).toEqual(["inference", "usage:read"])
  })

  it("leaves the end-user scopes off a key that is not a service key", () => {
    expect(heldScopes(["inference", "end_users:read"], false)).toEqual([
      "inference",
    ])
    expect(heldScopes(["inference", "end_users:read"], true)).toEqual([
      "inference",
      "end_users:read",
    ])
  })
})

describe("KeyScopeFields", () => {
  it("checks the scopes the key holds and adds one in the gateway's order", async () => {
    const user = userEvent.setup()
    const seen = renderFields(["inference"])

    expect(screen.getByRole("checkbox", { name: /Call models/ })).toBeChecked()
    expect(screen.getByRole("checkbox", { name: /Run code/ })).not.toBeChecked()
    await user.click(screen.getByRole("checkbox", { name: /Run code/ }))
    expect(seen.at(-1)).toEqual(["inference", "code:execute"])
  })

  it("checks or clears a whole group from its own box", async () => {
    const user = userEvent.setup()
    const seen = renderFields(["inference", "files:read"])
    const files = screen.getByRole("checkbox", { name: "All files scopes" })

    expect(files).toBePartiallyChecked()
    await user.click(files)
    expect(seen.at(-1)).toEqual(["inference", "files:read", "files:write"])
    expect(files).toBeChecked()
    await user.click(files)
    expect(seen.at(-1)).toEqual(["inference"])
  })

  it("flags the scopes a leaked key could do lasting harm with", () => {
    renderFields(["inference"])

    expect(screen.getByText("Runs arbitrary code")).toBeInTheDocument()
    expect(
      screen.getByText("Can queue spend that runs after the key is revoked"),
    ).toBeInTheDocument()
  })

  it("offers the end-user scopes only to a service key", () => {
    renderFields(["inference"])
    expect(
      screen.queryByRole("checkbox", { name: /end users/ }),
    ).not.toBeInTheDocument()
  })

  it("offers the end-user group on a service key", async () => {
    const user = userEvent.setup()
    const seen = renderFields(["inference"], true)

    await user.click(
      screen.getByRole("checkbox", { name: "All end users scopes" }),
    )

    expect(seen.at(-1)).toEqual([
      "inference",
      "end_users:read",
      "end_users:write",
      "end_users:delete",
    ])
  })

  it("says a key needs a scope when none is checked", async () => {
    const user = userEvent.setup()
    renderFields(["web"])

    await user.click(screen.getByRole("checkbox", { name: /Search and fetch/ }))

    expect(screen.getByText("Choose at least one")).toBeInTheDocument()
  })
})
