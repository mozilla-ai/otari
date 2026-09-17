import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"

import {
  enforcementFields,
  type GuardrailEnforcement,
  GuardrailEnforcementFields,
} from "@/features/tools/GuardrailEnforcementFields"
import { pickOption, selectTrigger } from "@/tests/select"

const WORKSPACES = [
  { id: "ws-1", name: "Platform" },
  { id: "ws-2", name: "Research" },
]

const DEFAULTS: GuardrailEnforcement = {
  mode: "block",
  onUnavailable: "block",
  everywhere: true,
  workspaceIds: [],
}

function Live({
  initial = DEFAULTS,
  workspaces = WORKSPACES,
  isLoadingWorkspaces = false,
  workspacesError,
  onChange,
}: {
  initial?: GuardrailEnforcement
  workspaces?: { id: string; name: string }[]
  isLoadingWorkspaces?: boolean
  workspacesError?: unknown
  onChange?: (next: GuardrailEnforcement) => void
}) {
  const [value, setValue] = useState(initial)
  return (
    <GuardrailEnforcementFields
      value={value}
      onChange={(next) => {
        setValue(next)
        onChange?.(next)
      }}
      workspaces={workspaces}
      isLoadingWorkspaces={isLoadingWorkspaces}
      workspacesError={workspacesError}
      isDisabled={false}
    />
  )
}

const workspacesField = () =>
  screen.getByRole("combobox", { name: /Workspaces/ })

describe("GuardrailEnforcementFields", () => {
  it("has nothing to decide about a missing verdict while it only reports", async () => {
    // The request path consults the fallback only for a blocking definition,
    // so offering it under "Report only" would promise a refusal that never
    // comes.
    const user = userEvent.setup()
    render(<Live />)
    expect(selectTrigger("When it cannot answer")).toBeEnabled()

    await pickOption(
      user,
      "When it flags a request",
      "Report only, let it through",
    )

    expect(selectTrigger("When it cannot answer")).toBeDisabled()
    expect(screen.getByText(/the request is served/)).toBeInTheDocument()
  })

  it("keeps the fallback the operator chose for when blocking resumes", async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(
      <Live
        initial={{ ...DEFAULTS, onUnavailable: "allow" }}
        onChange={onChange}
      />,
    )

    await pickOption(
      user,
      "When it flags a request",
      "Report only, let it through",
    )

    expect(onChange).toHaveBeenLastCalledWith({
      ...DEFAULTS,
      mode: "monitor",
      onUnavailable: "allow",
    })
  })

  it("says the workspaces are loading rather than that there are none", async () => {
    const user = userEvent.setup()
    render(
      <Live
        initial={{ ...DEFAULTS, everywhere: false }}
        workspaces={[]}
        isLoadingWorkspaces
      />,
    )

    await user.click(workspacesField())

    expect(screen.getByText("Loading workspaces…")).toBeInTheDocument()
    expect(screen.queryByText(/no workspaces yet/)).not.toBeInTheDocument()
  })

  it("says the workspaces could not be read rather than that there are none", async () => {
    const user = userEvent.setup()
    render(
      <Live
        initial={{ ...DEFAULTS, everywhere: false }}
        workspaces={[]}
        workspacesError={new Error("workspaces unavailable")}
      />,
    )

    expect(screen.getByRole("alert")).toHaveTextContent(
      "workspaces unavailable",
    )
    await user.click(workspacesField())
    expect(screen.queryByText(/no workspaces yet/)).not.toBeInTheDocument()
  })

  it("narrows the scope to the workspaces picked", async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Live onChange={onChange} />)

    await pickOption(user, "Where it runs", "Chosen workspaces")
    await user.click(workspacesField())
    await user.click(await screen.findByRole("option", { name: /Research/ }))

    expect(onChange).toHaveBeenLastCalledWith({
      ...DEFAULTS,
      everywhere: false,
      workspaceIds: ["ws-2"],
    })
  })

  it("sends no workspace list for a definition that covers every workspace", () => {
    // A list left over from a narrowed draft must not travel with "all": the
    // server ignores it, but a reader of the row would not.
    expect(enforcementFields({ ...DEFAULTS, workspaceIds: ["ws-1"] })).toEqual({
      mode: "block",
      on_unavailable: "block",
      applies_to_all_workspaces: true,
      workspace_ids: [],
    })
  })
})
