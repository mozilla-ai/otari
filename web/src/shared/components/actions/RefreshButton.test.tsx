import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { RefreshButton } from "@/shared/components/actions/RefreshButton"

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
})
