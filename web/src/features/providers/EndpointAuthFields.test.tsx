import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it } from "vitest"

import type { EndpointAuth } from "@/client"
import {
  EMPTY_ENDPOINT_AUTH,
  type EndpointAuthDraft,
  EndpointAuthFields,
  endpointAuthDraftFrom,
  isEndpointAuthComplete,
} from "@/features/providers/EndpointAuthFields"

function stored(overrides: Partial<EndpointAuth> = {}): EndpointAuth {
  return {
    method: "static_header",
    header_name: "Fastly-Key",
    last4: "abcd",
    expires_at: "2027-03-01T09:15:00+00:00",
    state: "valid",
    ...overrides,
  }
}

function Harness({
  initial = EMPTY_ENDPOINT_AUTH,
  held = null,
}: {
  initial?: EndpointAuthDraft
  held?: EndpointAuth | null
}) {
  const [value, setValue] = useState(initial)
  return (
    <>
      <EndpointAuthFields value={value} onChange={setValue} stored={held} />
      <output data-testid="draft">{JSON.stringify(value)}</output>
    </>
  )
}

const headerField = () => screen.getByLabelText("Endpoint credential header")

describe("endpointAuthDraftFrom", () => {
  it("is empty for a key that carries no endpoint credential", () => {
    expect(endpointAuthDraftFrom(null)).toEqual(EMPTY_ENDPOINT_AUTH)
  })

  it("never seeds the value, which the gateway does not return", () => {
    expect(endpointAuthDraftFrom(stored()).secret).toBe("")
  })

  it("seeds the rotation date in the shape the inputs read", () => {
    const seeded = endpointAuthDraftFrom(stored()).expiresAt

    expect(seeded).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/)
    // The input reads local time, so the calendar day follows the host. Read
    // back as local, it is still the instant the gateway reported.
    expect(new Date(seeded).toISOString()).toBe("2027-03-01T09:15:00.000Z")
  })
})

describe("isEndpointAuthComplete", () => {
  const complete: EndpointAuthDraft = {
    headerName: "Fastly-Key",
    secret: "edge-s3cret",
    expiresAt: "2027-03-01T00:00",
  }

  it("accepts a key that declares no endpoint credential", () => {
    expect(
      isEndpointAuthComplete(EMPTY_ENDPOINT_AUTH, {
        hasStoredSecret: false,
        addressMoved: false,
      }),
    ).toBe(true)
  })

  it("accepts a complete credential", () => {
    expect(
      isEndpointAuthComplete(complete, {
        hasStoredSecret: false,
        addressMoved: false,
      }),
    ).toBe(true)
  })

  it("refuses a header named with no value behind it", () => {
    expect(
      isEndpointAuthComplete(
        { ...complete, secret: "" },
        {
          hasStoredSecret: false,
          addressMoved: false,
        },
      ),
    ).toBe(false)
  })

  // The value is never returned, so a form reopened on a stored credential
  // shows a blank field that means "keep it" rather than "there is none".
  it("accepts a blank value when one is already stored", () => {
    expect(
      isEndpointAuthComplete(
        { ...complete, secret: "" },
        {
          hasStoredSecret: true,
          addressMoved: false,
        },
      ),
    ).toBe(true)
  })

  // The gateway will not send a stored value to a host the person saving it
  // never proved they hold, so the dialog asks rather than offering the save.
  it("refuses a stored value kept while the key moves to another address", () => {
    expect(
      isEndpointAuthComplete(
        { ...complete, secret: "" },
        { hasStoredSecret: true, addressMoved: true },
      ),
    ).toBe(false)
  })

  it("accepts a retyped value while the key moves to another address", () => {
    expect(
      isEndpointAuthComplete(complete, {
        hasStoredSecret: true,
        addressMoved: true,
      }),
    ).toBe(true)
  })

  it("refuses a credential with no rotation date", () => {
    expect(
      isEndpointAuthComplete(
        { ...complete, expiresAt: "" },
        {
          hasStoredSecret: true,
          addressMoved: false,
        },
      ),
    ).toBe(false)
  })
})

describe("EndpointAuthFields", () => {
  it("asks only for the header until one is named", () => {
    render(<Harness />)

    expect(headerField()).toBeInTheDocument()
    expect(
      screen.queryByLabelText("Endpoint credential"),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByLabelText("Credential expires"),
    ).not.toBeInTheDocument()
  })

  it("asks for the value and the rotation date once a header is named", async () => {
    render(<Harness />)

    await userEvent.type(headerField(), "Fastly-Key")

    expect(screen.getByLabelText("Endpoint credential")).toBeInTheDocument()
    expect(screen.getByLabelText("Credential expires")).toBeInTheDocument()
  })

  it("says a stored value is kept when the field is left blank", () => {
    render(
      <Harness initial={endpointAuthDraftFrom(stored())} held={stored()} />,
    )

    expect(screen.getByText(/Ending abcd/)).toBeInTheDocument()
    expect(screen.getByText(/Leave blank to keep it/)).toBeInTheDocument()
  })

  it("says nothing about a stored value on a credential being set for the first time", async () => {
    render(<Harness />)

    await userEvent.type(headerField(), "Fastly-Key")

    expect(screen.queryByText(/Leave blank to keep it/)).not.toBeInTheDocument()
  })
})
