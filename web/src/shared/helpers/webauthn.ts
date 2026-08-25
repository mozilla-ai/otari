/**
 * The browser half of a passkey ceremony: base64url, and the two `navigator`
 * calls.
 *
 * The gateway speaks base64url for every binary field (a challenge, a
 * credential id, a public key) because that is what the WebAuthn payloads carry
 * on the wire. `navigator.credentials` speaks `ArrayBuffer`. This module is the
 * one place that translates, so no component ever holds a half-decoded options
 * object.
 *
 * Deliberately not a library. `@simplewebauthn/browser` does exactly this and
 * little more, and the conversion is a few lines that the DOM already types;
 * adding a dependency to the dashboard bundle for it would cost more than it
 * saves.
 */

/**
 * The longest label a passkey may be given.
 *
 * Mirrors `MAX_WEBAUTHN_CREDENTIAL_NAME` in `models/tenancy.py`. Stated here so
 * the field stops accepting characters the gateway would refuse, rather than
 * letting somebody type a name and be told no on submit.
 */
export const MAX_PASSKEY_NAME_LENGTH = 255

/**
 * Whether this browser can do a passkey ceremony at all.
 *
 * Both halves are needed, and the second is the one that is easy to leave out.
 * `PublicKeyCredential` and `navigator.credentials` are *present* in an
 * insecure context and throw a `SecurityError` on use, so checking only for
 * the API offers a button whose one possible outcome is a failure. A gateway
 * served over plain HTTP on a LAN address is exactly that case, and it is a
 * deployment this project supports. `isSecureContext` is what the browser
 * itself gates the API on: HTTPS, plus localhost and the loopback addresses,
 * which is what keeps this working in local development.
 */
export function supportsPasskeys(): boolean {
  return (
    typeof window !== "undefined" &&
    window.isSecureContext === true &&
    typeof window.PublicKeyCredential === "function" &&
    typeof navigator !== "undefined" &&
    navigator.credentials != null
  )
}

/**
 * Decode base64url to bytes.
 *
 * base64url is base64 with two characters swapped and the padding dropped, and
 * `atob` accepts neither difference, so both are put back before decoding.
 */
export function base64UrlToBuffer(value: string): ArrayBuffer {
  const base64 = value.replace(/-/g, "+").replace(/_/g, "/")
  const padded = base64.padEnd(
    base64.length + ((4 - (base64.length % 4)) % 4),
    "=",
  )
  const binary = atob(padded)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }
  return bytes.buffer
}

/** Encode bytes as base64url, which is what every field the gateway reads uses. */
export function bufferToBase64Url(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ""
  for (const byte of bytes) {
    binary += String.fromCharCode(byte)
  }
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "")
}

/**
 * What the gateway's options endpoints return, narrowed to the fields decoded here.
 *
 * Typed as an index signature plus the two binary fields rather than as the
 * full W3C shape: the gateway passes the options through verbatim, everything
 * except these is handed to the browser untouched, and restating the spec would
 * mean dropping any field a later version adds.
 */
interface CreationOptionsJson {
  challenge: string
  user: { id: string; name: string; displayName: string }
  excludeCredentials?: { id: string; type: string; transports?: string[] }[]
  [key: string]: unknown
}

interface RequestOptionsJson {
  challenge: string
  allowCredentials?: { id: string; type: string; transports?: string[] }[]
  [key: string]: unknown
}

/**
 * The error thrown when a person dismisses the passkey prompt.
 *
 * Distinguished from every other failure because it is not one: closing the
 * system dialog is a decision, and reporting "that passkey could not be
 * verified" over it would be the dashboard telling somebody their hardware
 * failed when they pressed Escape.
 */
export class PasskeyCancelledError extends Error {
  constructor() {
    super("Passkey prompt dismissed")
    this.name = "PasskeyCancelledError"
  }
}

function asCancellation(error: unknown): never {
  // `NotAllowedError` is what a browser reports both for a dismissed prompt and
  // for a timed-out one, and it does not distinguish them on purpose (telling a
  // site which would leak whether a credential exists). Both mean "no ceremony
  // happened", which is the same thing to a caller.
  if (error instanceof DOMException && error.name === "NotAllowedError") {
    throw new PasskeyCancelledError()
  }
  throw error
}

/**
 * Run `navigator.credentials.create` and serialize the result for the gateway.
 *
 * The returned object is the JSON the gateway's `/register` endpoint verifies;
 * nothing else has to know its shape.
 */
export async function createPasskey(
  options: CreationOptionsJson,
): Promise<Record<string, unknown>> {
  let credential: Credential | null
  try {
    credential = await navigator.credentials.create({
      publicKey: {
        ...(options as unknown as PublicKeyCredentialCreationOptions),
        challenge: base64UrlToBuffer(options.challenge),
        user: {
          ...options.user,
          id: base64UrlToBuffer(options.user.id),
        },
        excludeCredentials: (options.excludeCredentials ?? []).map(
          (descriptor) => ({
            ...descriptor,
            id: base64UrlToBuffer(descriptor.id),
            type: "public-key" as const,
            transports: descriptor.transports as AuthenticatorTransport[],
          }),
        ),
      },
    })
  } catch (error) {
    asCancellation(error)
  }
  const created = credential as PublicKeyCredential | null
  if (created == null) {
    throw new PasskeyCancelledError()
  }
  const response = created.response as AuthenticatorAttestationResponse
  return {
    id: created.id,
    rawId: bufferToBase64Url(created.rawId),
    type: created.type,
    response: {
      clientDataJSON: bufferToBase64Url(response.clientDataJSON),
      attestationObject: bufferToBase64Url(response.attestationObject),
      // Advisory, and not every browser implements it, so an absent one is an
      // empty list rather than a failure.
      transports: response.getTransports?.() ?? [],
    },
    clientExtensionResults: created.getClientExtensionResults(),
  }
}

/**
 * Run `navigator.credentials.get` and serialize the assertion for the gateway.
 *
 * `allowCredentials` is empty for this deployment's usernameless sign-in, so
 * the browser offers whichever passkey it holds; the mapping below is kept
 * because the field is part of the shape and a future caller may pass one.
 */
export async function getPasskeyAssertion(
  options: RequestOptionsJson,
): Promise<Record<string, unknown>> {
  let credential: Credential | null
  try {
    credential = await navigator.credentials.get({
      publicKey: {
        ...(options as unknown as PublicKeyCredentialRequestOptions),
        challenge: base64UrlToBuffer(options.challenge),
        allowCredentials: (options.allowCredentials ?? []).map(
          (descriptor) => ({
            ...descriptor,
            id: base64UrlToBuffer(descriptor.id),
            type: "public-key" as const,
            transports: descriptor.transports as AuthenticatorTransport[],
          }),
        ),
      },
    })
  } catch (error) {
    asCancellation(error)
  }
  const asserted = credential as PublicKeyCredential | null
  if (asserted == null) {
    throw new PasskeyCancelledError()
  }
  const response = asserted.response as AuthenticatorAssertionResponse
  return {
    id: asserted.id,
    rawId: bufferToBase64Url(asserted.rawId),
    type: asserted.type,
    response: {
      clientDataJSON: bufferToBase64Url(response.clientDataJSON),
      authenticatorData: bufferToBase64Url(response.authenticatorData),
      signature: bufferToBase64Url(response.signature),
      userHandle:
        response.userHandle == null
          ? null
          : bufferToBase64Url(response.userHandle),
    },
    clientExtensionResults: asserted.getClientExtensionResults(),
  }
}
