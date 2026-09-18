import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { pickOption } from "@/tests/select"

import { TablePagination } from "./TablePagination"

function setup(overrides: Partial<Parameters<typeof TablePagination>[0]> = {}) {
  const onPageChange = vi.fn()
  const onPageSizeChange = vi.fn()
  render(
    <TablePagination
      page={0}
      pageSize={100}
      total={4231}
      rowsOnPage={100}
      onPageChange={onPageChange}
      onPageSizeChange={onPageSizeChange}
      {...overrides}
    />,
  )
  return { onPageChange, onPageSizeChange }
}

describe("TablePagination", () => {
  it("shows a truthful range-of-total summary", () => {
    setup()
    expect(screen.getByText("1–100 of 4,231")).toBeInTheDocument()
    expect(screen.getByText("/ 43")).toBeInTheDocument()
  })

  it("disables first/prev on the first page", () => {
    setup()
    expect(screen.getByRole("button", { name: "First page" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Previous page" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Next page" })).toBeEnabled()
    expect(screen.getByRole("button", { name: "Last page" })).toBeEnabled()
  })

  it("jumps to the last page", async () => {
    const user = userEvent.setup()
    const { onPageChange } = setup()
    await user.click(screen.getByRole("button", { name: "Last page" }))
    expect(onPageChange).toHaveBeenCalledWith(42) // ceil(4231/100) - 1
  })

  it("follows the page it is given, discarding anything half-typed", async () => {
    const user = userEvent.setup()
    const props = {
      pageSize: 100,
      total: 4231,
      rowsOnPage: 100,
      onPageChange: vi.fn(),
      onPageSizeChange: vi.fn(),
    }
    const { rerender } = render(<TablePagination page={0} {...props} />)
    const box = screen.getByRole("textbox", { name: "Page number" })
    expect(box).toHaveValue("1")

    // Uncommitted: the box holds it, nothing else has been told.
    await user.clear(box)
    await user.type(box, "7")
    expect(box).toHaveValue("7")

    // The page moves from outside, which is the operator having pressed
    // something else. The box reads the page it labels, not the abandoned
    // keystrokes.
    rerender(<TablePagination page={3} {...props} />)
    expect(screen.getByRole("textbox", { name: "Page number" })).toHaveValue(
      "4",
    )
  })

  it("commits a typed page, clamped to the page count", async () => {
    const user = userEvent.setup()
    const { onPageChange } = setup()
    const box = screen.getByRole("textbox", { name: "Page number" })
    await user.clear(box)
    await user.type(box, "999{enter}")
    expect(onPageChange).toHaveBeenCalledWith(42) // clamped to last page
  })

  it("changes rows per page", async () => {
    const user = userEvent.setup()
    const { onPageSizeChange } = setup()
    // 50 rather than the 100 this fixture already shows: react-aria reports a
    // selection that changed, so re-picking the current size is correctly
    // silent, where the native select fired `change` either way.
    await pickOption(user, "Rows per page", "50")
    expect(onPageSizeChange).toHaveBeenCalledWith(50)
  })

  it("with an unknown total, hides last and uses the next fallback", () => {
    setup({ total: null, hasNextFallback: true, rowsOnPage: 100 })
    expect(screen.getByText("1–100")).toBeInTheDocument()
    expect(screen.queryByText(/\/ /)).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Last page" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Next page" })).toBeEnabled()
  })

  it("announces the range, which changes with no focus move to carry it", () => {
    setup()
    // `role="status"` rather than a plain span: paging, resizing and filtering
    // all rewrite this text in place, and the controls that did it keep focus.
    expect(screen.getByRole("status")).toHaveTextContent("1–100 of 4,231")
  })

  it("with an unknown total and a short page, disables next", () => {
    setup({ total: null, hasNextFallback: false, rowsOnPage: 40, page: 2 })
    expect(screen.getByRole("button", { name: "Next page" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Previous page" })).toBeEnabled()
  })
})
