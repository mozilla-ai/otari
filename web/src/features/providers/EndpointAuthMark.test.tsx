import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import type { EndpointAuth } from "@/client"
import { EndpointAuthMark } from "@/features/providers/EndpointAuthMark"

function auth(overrides: Partial<EndpointAuth> = {}): EndpointAuth {
  return {
    method: "static_header",
    header_name: "Fastly-Key",
    last4: "abcd",
    expires_at: "2027-03-01T00:00:00+00:00",
    state: "valid",
    ...overrides,
  }
}

describe("EndpointAuthMark", () => {
  it("draws nothing for a key that carries no endpoint credential", () => {
    const { container } = render(
      <EndpointAuthMark auth={null} isUnreadable={false} />,
    )

    expect(container).toBeEmptyDOMElement()
  })

  it("draws nothing while the credential is inside its window", () => {
    const { container } = render(
      <EndpointAuthMark auth={auth()} isUnreadable={false} />,
    )

    expect(container).toBeEmptyDOMElement()
  })

  it("names the rotation date as due when it is near", () => {
    render(
      <EndpointAuthMark
        auth={auth({ state: "expiring_soon" })}
        isUnreadable={false}
      />,
    )

    expect(screen.getByText(/Endpoint credential due/)).toHaveClass(
      "text-warning",
    )
  })

  it("names the rotation date as passed once it has", () => {
    render(
      <EndpointAuthMark
        auth={auth({ state: "expired" })}
        isUnreadable={false}
      />,
    )

    expect(screen.getByText(/Endpoint credential expired/)).toHaveClass(
      "text-danger",
    )
  })

  // A date that has passed is a declaration about the proxy, not a fact about
  // the credential, so the mark reports it and nothing else acts on it.
  it("reports a passed date without hiding the date itself", () => {
    render(
      <EndpointAuthMark
        auth={auth({ state: "expired", expires_at: "2020-01-01T00:00:00Z" })}
        isUnreadable={false}
      />,
    )

    expect(screen.getByText(/2020/)).toBeInTheDocument()
  })

  it("names a stored credential this deployment cannot read", () => {
    render(<EndpointAuthMark auth={null} isUnreadable={true} />)

    expect(screen.getByText(/Endpoint credential unreadable/)).toHaveClass(
      "text-danger",
    )
  })

  // The row carries one and nobody can send it, so there is no window left to
  // report. Reporting the date would say the credential is fine until March.
  it("reports the unreadable credential rather than its date", () => {
    render(<EndpointAuthMark auth={auth()} isUnreadable={true} />)

    expect(
      screen.getByText(/Endpoint credential unreadable/),
    ).toBeInTheDocument()
    expect(screen.queryByText(/due/)).not.toBeInTheDocument()
  })
})
