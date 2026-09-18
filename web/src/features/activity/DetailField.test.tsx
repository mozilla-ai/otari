import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { DetailField } from "./DetailField"

describe("DetailField", () => {
  it("labels its value", () => {
    render(<DetailField label="Endpoint">/v1/chat/completions</DetailField>)

    expect(screen.getByText("Endpoint")).toBeInTheDocument()
    expect(screen.getByText("/v1/chat/completions")).toBeInTheDocument()
  })

  it("offers a copy control only for a field that holds an identifier", () => {
    const { rerender } = render(
      <DetailField label="Provider">OpenAI</DetailField>,
    )
    expect(screen.queryByRole("button")).not.toBeInTheDocument()

    rerender(
      <DetailField label="Request ID" copyValue="req-1">
        req-1
      </DetailField>,
    )
    expect(
      screen.getByRole("button", { name: /request id/i }),
    ).toBeInTheDocument()
  })

  it("names the copy control for the value, not for the column heading", () => {
    // "API key" holds a key id, never key material, so the heading would
    // misname what the button puts on the clipboard.
    render(
      <DetailField label="API key" copyValue="key-1" copyLabel="api key id">
        key-1
      </DetailField>,
    )

    expect(
      screen.getByRole("button", { name: /api key id/i }),
    ).toBeInTheDocument()
  })
})
