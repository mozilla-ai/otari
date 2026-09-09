import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it } from "vitest"

import { useAutosave } from "@/shared/hooks/useAutosave"

function Harness({ commit }: { commit: () => Promise<unknown> }) {
  const save = useAutosave()
  return (
    <>
      <button
        type="button"
        disabled={save.isSaving}
        onClick={() => void save.run(commit)}
      >
        Commit
      </button>
      <span data-testid="error">{save.error}</span>
    </>
  )
}

const commitButton = () => screen.getByRole("button", { name: "Commit" })
const error = () => screen.getByTestId("error")

describe("useAutosave", () => {
  it("says nothing when a save works", async () => {
    // No confirmation, no timer to clear: the page saves as you leave a field,
    // and a mark on every row is one the reader learns to ignore.
    const user = userEvent.setup()
    render(<Harness commit={() => Promise.resolve()} />)

    await user.click(commitButton())

    await waitFor(() => expect(commitButton()).toBeEnabled())
    expect(error()).toBeEmptyDOMElement()
    expect(screen.queryByRole("status")).toBeNull()
  })

  it("holds the control while a save is in flight", async () => {
    const user = userEvent.setup()
    let release: () => void = () => {}
    const pending = new Promise<void>((resolve) => {
      release = resolve
    })
    render(<Harness commit={() => pending} />)

    await user.click(commitButton())
    expect(commitButton()).toBeDisabled()

    release()
    await waitFor(() => expect(commitButton()).toBeEnabled())
  })

  it("keeps a refused save's message", async () => {
    // The value that caused it is still in the field, so the message stays
    // until the next attempt.
    const user = userEvent.setup()
    render(
      <Harness commit={() => Promise.reject(new Error("Must be a URL."))} />,
    )

    await user.click(commitButton())

    await waitFor(() => expect(error()).toHaveTextContent("Must be a URL."))
    expect(commitButton()).toBeEnabled()
  })

  it("drops the previous message when a new attempt starts", async () => {
    const user = userEvent.setup()
    let fail = true
    render(
      <Harness
        commit={() =>
          fail ? Promise.reject(new Error("Must be a URL.")) : Promise.resolve()
        }
      />,
    )

    await user.click(commitButton())
    await waitFor(() => expect(error()).toHaveTextContent("Must be a URL."))

    fail = false
    await user.click(commitButton())

    await waitFor(() => expect(error()).toBeEmptyDOMElement())
  })
})
