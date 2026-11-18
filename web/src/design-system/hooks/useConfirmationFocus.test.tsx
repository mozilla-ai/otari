import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it } from "vitest"
import { ConfirmRowAction } from "@/design-system/actions/ConfirmRowAction"

function Table({ names }: { names: string[] }) {
  const [rows, setRows] = useState(names)
  return (
    <section>
      <h2>Keys</h2>
      <table>
        <tbody>
          {rows.map((name) => (
            <tr key={name}>
              <td>{name}</td>
              <td>
                <ConfirmRowAction
                  confirmLabel={`Delete ${name}`}
                  onConfirm={() => setRows((r) => r.filter((n) => n !== name))}
                >
                  {`Remove ${name}`}
                </ConfirmRowAction>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

const remove = async (name: string) => {
  const user = userEvent.setup()
  await user.click(screen.getByRole("button", { name: `Remove ${name}` }))
  await user.click(screen.getByRole("button", { name: `Delete ${name}` }))
}

describe("focus after a confirmed delete", () => {
  it("lands on the next row's same control", async () => {
    render(<Table names={["a", "b", "c"]} />)
    await remove("b")
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Remove c" })).toHaveFocus(),
    )
  })

  it("lands on the previous row's control after the last row", async () => {
    render(<Table names={["a", "b"]} />)
    await remove("b")
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Remove a" })).toHaveFocus(),
    )
  })

  it("lands on the heading after the only row, which is not left focusable", async () => {
    render(<Table names={["a"]} />)
    await remove("a")
    const heading = screen.getByRole("heading", { name: "Keys" })
    await waitFor(() => expect(heading).toHaveFocus())
    await userEvent.setup().tab()
    expect(heading).not.toHaveAttribute("tabindex")
  })

  it("leaves focus alone when the action is cancelled", async () => {
    render(<Table names={["a", "b", "c"]} />)
    const user = userEvent.setup()
    await user.click(screen.getByRole("button", { name: "Remove b" }))
    await user.click(screen.getAllByRole("button", { name: "Cancel" })[0])
    expect(screen.getByRole("button", { name: "Remove b" })).toHaveFocus()
  })
})
