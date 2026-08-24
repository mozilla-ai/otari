import { afterEach, describe, expect, it, vi } from "vitest"

import {
  base64UrlToBuffer,
  bufferToBase64Url,
  createPasskey,
  getPasskeyAssertion,
  PasskeyCancelledError,
  supportsPasskeys,
} from "@/shared/helpers/webauthn"

function bytes(...values: number[]): ArrayBuffer {
  return new Uint8Array(values).buffer
}

function toArray(buffer: ArrayBuffer): number[] {
  return [...new Uint8Array(buffer)]
}

describe("base64url", () => {
  it("round-trips bytes", () => {
    const original = bytes(0, 1, 250, 251, 252, 253, 254, 255)
    expect(toArray(base64UrlToBuffer(bufferToBase64Url(original)))).toEqual(
      toArray(original),
    )
  })

  it("emits the URL-safe alphabet and no padding", () => {
    // 0xfb 0xff 0xbf is the byte triple whose plain base64 is "+/+/", so it
    // exercises both substitutions at once.
    const encoded = bufferToBase64Url(bytes(0xfb, 0xff, 0xbf))
    expect(encoded).not.toMatch(/[+/=]/)
    expect(toArray(base64UrlToBuffer(encoded))).toEqual([0xfb, 0xff, 0xbf])
  })

  it("decodes a value whose padding was stripped", () => {
    // One byte encodes to two characters plus two "=" a browser omits; `atob`
    // rejects that, so the decoder has to put them back.
    expect(toArray(base64UrlToBuffer("AQ"))).toEqual([1])
    expect(toArray(base64UrlToBuffer("AQI"))).toEqual([1, 2])
  })

  it("decodes an empty value", () => {
    expect(toArray(base64UrlToBuffer(""))).toEqual([])
  })
})

describe("supportsPasskeys", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("is false without PublicKeyCredential", () => {
    vi.stubGlobal("PublicKeyCredential", undefined)
    expect(supportsPasskeys()).toBe(false)
  })

  it("is true when the API is present", () => {
    vi.stubGlobal("PublicKeyCredential", function PublicKeyCredential() {})
    vi.stubGlobal("navigator", { credentials: {} })
    expect(supportsPasskeys()).toBe(true)
  })
})

// The minimum a `PublicKeyCredential` has to look like for the serializers.
function fakeCredential(response: object) {
  return {
    id: "Y3JlZA",
    rawId: bytes(1, 2, 3),
    type: "public-key",
    response,
    getClientExtensionResults: () => ({}),
  }
}

const CREATION_OPTIONS = {
  challenge: "Y2hhbGxlbmdl",
  rp: { id: "otari.example.com", name: "otari" },
  user: { id: "dXNlcg", name: "operator", displayName: "Operator" },
  excludeCredentials: [{ id: "b2xk", type: "public-key" }],
}

const REQUEST_OPTIONS = { challenge: "Y2hhbGxlbmdl", allowCredentials: [] }

describe("createPasskey", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("decodes the binary fields before handing them to the browser", async () => {
    const create = vi.fn().mockResolvedValue(
      fakeCredential({
        clientDataJSON: bytes(4, 5),
        attestationObject: bytes(6, 7),
        getTransports: () => ["internal"],
      }),
    )
    vi.stubGlobal("navigator", { credentials: { create } })

    await createPasskey(CREATION_OPTIONS)

    const passed = create.mock.calls[0][0].publicKey
    // The two fields the gateway sends as base64url reach the browser as bytes;
    // anything still a string here would throw inside the real API.
    expect(toArray(passed.challenge)).toEqual(
      toArray(base64UrlToBuffer("Y2hhbGxlbmdl")),
    )
    expect(toArray(passed.user.id)).toEqual(
      toArray(base64UrlToBuffer("dXNlcg")),
    )
    expect(toArray(passed.excludeCredentials[0].id)).toEqual(
      toArray(base64UrlToBuffer("b2xk")),
    )
    // Everything else is passed through untouched.
    expect(passed.rp).toEqual(CREATION_OPTIONS.rp)
  })

  it("serializes the response back to base64url", async () => {
    vi.stubGlobal("navigator", {
      credentials: {
        create: vi.fn().mockResolvedValue(
          fakeCredential({
            clientDataJSON: bytes(4, 5),
            attestationObject: bytes(6, 7),
            getTransports: () => ["internal", "hybrid"],
          }),
        ),
      },
    })

    const result = (await createPasskey(CREATION_OPTIONS)) as {
      rawId: string
      response: Record<string, unknown>
    }

    expect(result.rawId).toBe(bufferToBase64Url(bytes(1, 2, 3)))
    expect(result.response.clientDataJSON).toBe(bufferToBase64Url(bytes(4, 5)))
    expect(result.response.attestationObject).toBe(
      bufferToBase64Url(bytes(6, 7)),
    )
    expect(result.response.transports).toEqual(["internal", "hybrid"])
  })

  it("reports no transports when the browser does not implement them", async () => {
    vi.stubGlobal("navigator", {
      credentials: {
        create: vi.fn().mockResolvedValue(
          fakeCredential({
            clientDataJSON: bytes(4, 5),
            attestationObject: bytes(6, 7),
          }),
        ),
      },
    })

    const result = (await createPasskey(CREATION_OPTIONS)) as {
      response: Record<string, unknown>
    }
    expect(result.response.transports).toEqual([])
  })

  it("turns a dismissed prompt into PasskeyCancelledError", async () => {
    vi.stubGlobal("navigator", {
      credentials: {
        create: vi
          .fn()
          .mockRejectedValue(new DOMException("denied", "NotAllowedError")),
      },
    })

    await expect(createPasskey(CREATION_OPTIONS)).rejects.toBeInstanceOf(
      PasskeyCancelledError,
    )
  })

  it("lets any other failure through as itself", async () => {
    vi.stubGlobal("navigator", {
      credentials: {
        create: vi
          .fn()
          .mockRejectedValue(new DOMException("bad rp", "SecurityError")),
      },
    })

    // A misconfigured relying party is a real failure and must not be reported
    // as somebody cancelling.
    await expect(createPasskey(CREATION_OPTIONS)).rejects.toBeInstanceOf(
      DOMException,
    )
  })
})

describe("getPasskeyAssertion", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("serializes an assertion, carrying a null user handle through", async () => {
    vi.stubGlobal("navigator", {
      credentials: {
        get: vi.fn().mockResolvedValue(
          fakeCredential({
            clientDataJSON: bytes(4, 5),
            authenticatorData: bytes(6, 7),
            signature: bytes(8, 9),
            userHandle: null,
          }),
        ),
      },
    })

    const result = (await getPasskeyAssertion(REQUEST_OPTIONS)) as {
      response: Record<string, unknown>
    }

    expect(result.response.authenticatorData).toBe(
      bufferToBase64Url(bytes(6, 7)),
    )
    expect(result.response.signature).toBe(bufferToBase64Url(bytes(8, 9)))
    expect(result.response.userHandle).toBeNull()
  })

  it("turns a dismissed prompt into PasskeyCancelledError", async () => {
    vi.stubGlobal("navigator", {
      credentials: {
        get: vi
          .fn()
          .mockRejectedValue(new DOMException("denied", "NotAllowedError")),
      },
    })

    await expect(getPasskeyAssertion(REQUEST_OPTIONS)).rejects.toBeInstanceOf(
      PasskeyCancelledError,
    )
  })

  it("treats a null credential as a cancellation", async () => {
    // Specified to resolve null rather than reject in some paths, and a null
    // dereference below would be a crash instead of a dismissed prompt.
    vi.stubGlobal("navigator", {
      credentials: { get: vi.fn().mockResolvedValue(null) },
    })

    await expect(getPasskeyAssertion(REQUEST_OPTIONS)).rejects.toBeInstanceOf(
      PasskeyCancelledError,
    )
  })
})
