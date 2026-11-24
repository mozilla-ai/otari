import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it } from "vitest"

import type { EndpointAuth } from "@/client"
import {
  EMPTY_ENDPOINT_AUTH,
  type EndpointAuthDraft,
  EndpointAuthFields,
  endpointAuthAddressError,
  endpointAuthDraftFrom,
  endpointAuthSecretError,
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
  heldIsUnreadable = false,
  moved = false,
}: {
  initial?: EndpointAuthDraft
  held?: EndpointAuth | null
  heldIsUnreadable?: boolean
  moved?: boolean
}) {
  const [value, setValue] = useState(initial)
  return (
    <>
      <EndpointAuthFields
        value={value}
        onChange={setValue}
        stored={held}
        storedIsUnreadable={heldIsUnreadable}
        addressMoved={moved}
      />
      <output data-testid="draft">{JSON.stringify(value)}</output>
    </>
  )
}

const secretField = () => screen.getByLabelText("Endpoint credential")

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

describe("endpointAuthAddressError", () => {
  const named: EndpointAuthDraft = {
    headerName: "Fastly-Key",
    secret: "edge-s3cret",
    expiresAt: "2027-03-01T00:00",
  }

  it("finds nothing wrong with an address while no header is named", () => {
    expect(
      endpointAuthAddressError(
        "http://edge.example.com/v1",
        EMPTY_ENDPOINT_AUTH,
      ),
    ).toBeUndefined()
  })

  it("accepts an HTTPS address", () => {
    expect(
      endpointAuthAddressError("https://edge.example.com/v1", named),
    ).toBeUndefined()
  })

  it("accepts an address whose scheme is spelled in capitals", () => {
    expect(
      endpointAuthAddressError("HTTPS://edge.example.com/v1", named),
    ).toBeUndefined()
  })

  // The refusal the gateway answers with, said before the save is offered.
  it.each([
    "http://edge.example.com/v1",
    "HTTP://edge.example.com/v1",
    "//edge.example.com/v1",
    "edge.example.com/v1",
  ])("refuses %s, which would send the credential in clear", (apiBase) => {
    expect(endpointAuthAddressError(apiBase, named)).toBe(
      "Must be HTTPS, since the endpoint credential travels with it.",
    )
  })

  // The other half of the gateway's rule: a credential needs an address of its
  // own, so clearing the address is refused while a header is named.
  it("refuses a credential with no address behind it", () => {
    expect(endpointAuthAddressError("   ", named)).toBe(
      "An endpoint credential needs an address of its own. Enter one, or clear the credential header below.",
    )
  })
})

describe("endpointAuthSecretError", () => {
  const draft = (secret: string): EndpointAuthDraft => ({
    headerName: "Fastly-Key",
    secret,
    expiresAt: "2027-03-01T00:00",
  })

  // A blank field means "keep what is stored", so it carries no complaint.
  it("finds nothing wrong with a blank value", () => {
    expect(endpointAuthSecretError(draft(""))).toBeUndefined()
  })

  it("names the floor for a value under it", () => {
    expect(endpointAuthSecretError(draft("123"))).toBe("At least 8 characters.")
  })

  it("finds nothing wrong with a value at the floor", () => {
    expect(endpointAuthSecretError(draft("12345678"))).toBeUndefined()
  })

  // Eight UTF-16 units and four code points. The gateway counts code points,
  // so passing this would offer a save it refuses.
  it("counts what the gateway counts, not UTF-16 units", () => {
    expect(
      endpointAuthSecretError(draft("\u{1F510}\u{1F510}\u{1F510}\u{1F510}")),
    ).toBe("At least 8 characters.")
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

  // Without this the dialog offers a save the gateway answers with its own
  // validator's error, which names the request field rather than the label.
  it("refuses a value under the floor", () => {
    expect(
      isEndpointAuthComplete(
        { ...complete, secret: "123" },
        { hasStoredSecret: false, addressMoved: false },
      ),
    ).toBe(false)
  })

  it("refuses a value under the floor even where one is already stored", () => {
    expect(
      isEndpointAuthComplete(
        { ...complete, secret: "123" },
        { hasStoredSecret: true, addressMoved: false },
      ),
    ).toBe(false)
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

  it("says on the field itself that a value is too short", async () => {
    render(<Harness />)

    await userEvent.type(headerField(), "Fastly-Key")
    await userEvent.type(screen.getByLabelText("Endpoint credential"), "123")

    expect(screen.getByText("At least 8 characters.")).toBeInTheDocument()
  })

  it("drops the complaint once the value reaches the floor", async () => {
    render(<Harness />)

    await userEvent.type(headerField(), "Fastly-Key")
    await userEvent.type(
      screen.getByLabelText("Endpoint credential"),
      "12345678",
    )

    expect(screen.queryByText("At least 8 characters.")).not.toBeInTheDocument()
  })

  it("says a stored value is kept when the field is left blank", () => {
    render(
      <Harness initial={endpointAuthDraftFrom(stored())} held={stored()} />,
    )

    expect(screen.getByText(/Leave blank to keep it/)).toBeInTheDocument()
  })

  // A bare tail reads as a broken word, which is how "value" showed as "alue".
  it("shows a stored tail masked, in the field rather than the caption", () => {
    render(
      <Harness initial={endpointAuthDraftFrom(stored())} held={stored()} />,
    )

    expect(secretField()).toHaveAttribute("placeholder", "Keep ••••abcd")
    expect(screen.queryByText(/Ending abcd/)).not.toBeInTheDocument()
  })

  // The state the paragraph got wrong: a blank is refused here, so offering to
  // keep one contradicts the disabled save.
  it("asks for the value again while the key moves to another address", () => {
    render(
      <Harness
        initial={endpointAuthDraftFrom(stored())}
        held={stored()}
        moved={true}
      />,
    )

    expect(
      screen.getByText(
        "Enter the credential again: it goes to the new address.",
      ),
    ).toBeInTheDocument()
    expect(screen.queryByText(/Leave blank to keep it/)).not.toBeInTheDocument()
  })

  it("marks the value required on a credential being set for the first time", async () => {
    render(<Harness />)

    await userEvent.type(headerField(), "Fastly-Key")

    expect(secretField()).toBeRequired()
  })

  it("marks the value required while the key moves to another address", () => {
    render(
      <Harness
        initial={endpointAuthDraftFrom(stored())}
        held={stored()}
        moved={true}
      />,
    )

    expect(secretField()).toBeRequired()
  })

  it("leaves the value optional where a blank keeps what is stored", () => {
    render(
      <Harness initial={endpointAuthDraftFrom(stored())} held={stored()} />,
    )

    expect(secretField()).not.toBeRequired()
  })

  // A blank removes an unreadable credential, so asking for one would mark a
  // field whose whole purpose is to be left empty.
  it("leaves the value optional where a blank removes an unreadable credential", async () => {
    render(<Harness heldIsUnreadable={true} />)

    await userEvent.type(headerField(), "Fastly-Key")

    expect(secretField()).not.toBeRequired()
  })

  it("marks the rotation date required", async () => {
    render(<Harness />)

    await userEvent.type(headerField(), "Fastly-Key")

    expect(screen.getByLabelText("Credential expires")).toBeRequired()
  })

  it("says nothing about a stored value on a credential being set for the first time", async () => {
    render(<Harness />)

    await userEvent.type(headerField(), "Fastly-Key")

    expect(screen.queryByText(/Leave blank to keep it/)).not.toBeInTheDocument()
  })

  // The fields come up empty either way, so without this the form says the key
  // carries no credential while it carries one nobody can send.
  it("says a stored credential cannot be read, and how to remove it", () => {
    render(<Harness heldIsUnreadable={true} />)

    expect(screen.getByText(/cannot read/)).toBeInTheDocument()
    expect(
      screen.getByText(/Save with this blank to remove it/),
    ).toBeInTheDocument()
  })
})
