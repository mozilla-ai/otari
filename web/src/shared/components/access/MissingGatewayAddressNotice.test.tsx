import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { API_ROOT } from "@/shared/api/client"
import { MissingGatewayAddressNotice } from "@/shared/components/access/MissingGatewayAddressNotice"

describe("MissingGatewayAddressNotice", () => {
  it("says why there is no snippet and what to ask for instead", () => {
    // The two pages that hand out a key render this one component, so the wording
    // is pinned here rather than twice over in their own suites.
    render(<MissingGatewayAddressNotice />)

    expect(
      screen.getByText(/has not published the gateway address/),
    ).toBeInTheDocument()
    expect(screen.getByText(`${API_ROOT}/chat/completions`)).toBeInTheDocument()
    expect(screen.getByText("Otari-Key")).toBeInTheDocument()
  })
})
