import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { PageLoading } from "@/shared/components/feedback/PageLoading"

describe("PageLoading", () => {
  it("exposes a status role so the wait is announced", () => {
    render(<PageLoading />)
    const status = screen.getByRole("status")
    expect(status).toHaveTextContent("Loading…")
  })
})
