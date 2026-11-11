import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it } from "vitest"

import type { KeyScope } from "@/client"
import { heldScopes, KeyScopeFields } from "@/features/keys/KeyScopeFields"
import { pickOption, selectTrigger } from "@/tests/select"

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
  it("shows a preset rather than the list when the scopes match one", () => {
    renderFields(["inference"])

    expect(screen.getByText(/Model calls alone/)).toBeInTheDocument()
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument()
  })

  it("opens the list on Custom and keeps the scopes it had", async () => {
    const user = userEvent.setup()
    const seen = renderFields(["inference"])

    await pickOption(user, "Access", "Custom")

    expect(screen.getByRole("checkbox", { name: /Call models/ })).toBeChecked()
    expect(seen).toEqual([])
    await user.click(screen.getByRole("checkbox", { name: /Run code/ }))
    expect(seen.at(-1)).toEqual(["inference", "code:execute"])
  })

  it("flags the scopes a leaked key could do lasting harm with", async () => {
    const user = userEvent.setup()
    renderFields(["inference"])

    await pickOption(user, "Access", "Custom")

    expect(screen.getByText("Runs arbitrary code")).toBeInTheDocument()
    expect(
      screen.getByText("Can queue spend that runs after the key is revoked"),
    ).toBeInTheDocument()
  })

  it("offers the end-user scopes and preset only to a service key", async () => {
    const user = userEvent.setup()
    renderFields(["inference", "usage:read"])

    expect(
      screen.queryByRole("checkbox", { name: /end users/ }),
    ).not.toBeInTheDocument()
    await user.click(selectTrigger("Access"))
    expect(await screen.findByRole("option", { name: "Custom" })).toBeVisible()
    expect(
      screen.queryByRole("option", { name: "Service with end users" }),
    ).not.toBeInTheDocument()
  })

  it("picks the service preset on a service key", async () => {
    const user = userEvent.setup()
    const seen = renderFields(["inference"], true)

    await pickOption(user, "Access", "Service with end users")

    expect(seen.at(-1)).toEqual([
      "inference",
      "end_users:read",
      "end_users:write",
    ])
  })

  it("says a key needs a scope when none is checked", async () => {
    const user = userEvent.setup()
    renderFields(["web"])

    await user.click(screen.getByRole("checkbox", { name: /Search and fetch/ }))

    expect(screen.getByText("Choose at least one")).toBeInTheDocument()
  })
})
