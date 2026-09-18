import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { RefreshButton } from "@/design-system/actions/RefreshButton"

describe("RefreshButton", () => {
  it("fires onRefresh and shows a freshness label", async () => {
    const user = userEvent.setup()
    const onRefresh = vi.fn()
    render(
      <RefreshButton onRefresh={onRefresh} updatedAt={Date.now() - 5_000} />,
    )
    expect(screen.getByText(/Updated/)).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Refresh" }))
    expect(onRefresh).toHaveBeenCalledOnce()
  })

  it("hides the timestamp before the first load and disables while fetching", () => {
    const onRefresh = vi.fn()
    render(<RefreshButton onRefresh={onRefresh} isFetching updatedAt={0} />)
    expect(screen.queryByText(/Updated/)).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Refresh" })).toBeDisabled()
  })

  // The layout fact being pinned: the button's box is 44x44 below `md` and
  // drops back to 32x32 from `md` up. jsdom performs no layout and this control
  // has no accessible expression of its own size, so the classes that cause the
  // box are the only thing a unit test can see. It is worth a test at all
  // because this is a design-system export: it carries its size onto four
  // pages at once, and it shipped under the floor on all four (#1336).
  it("keeps a 44px target on a phone and the toolbar's density on a pointer", () => {
    render(<RefreshButton onRefresh={vi.fn()} updatedAt={Date.now()} />)
    const button = screen.getByRole("button", { name: "Refresh" })
    expect(button).toHaveClass("min-h-11", "min-w-11")
    expect(button).toHaveClass("md:min-h-8", "md:min-w-8")
  })
})
