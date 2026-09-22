import { describe, expect, it } from "vitest"

import type { GuardrailParameterSpec } from "@/client"
import {
  buildCreateKwargs,
  definitionFieldSpecs,
  heldSecrets,
  seedableArguments,
} from "@/features/guardrails/definitionForm"
import {
  parameterErrors,
  seedParameters,
} from "@/features/guardrails/guardrailParameters"
import { REDACTED_SECRET } from "@/shared/helpers/redaction"

function spec(
  overrides: Partial<GuardrailParameterSpec> &
    Pick<GuardrailParameterSpec, "name">,
): GuardrailParameterSpec {
  return {
    type: "string",
    required: false,
    secret: false,
    storable: true,
    ...overrides,
  }
}

const apiKey = spec({ name: "api_key", secret: true, required: true })
const endpoint = spec({ name: "endpoint" })
const specs = [apiKey, endpoint]

describe("heldSecrets", () => {
  it("names the secrets the definition holds", () => {
    expect(
      heldSecrets({ create_secrets: { api_key: REDACTED_SECRET } }),
    ).toEqual(new Set(["api_key"]))
  })

  it("holds nothing for a new definition", () => {
    expect(heldSecrets(undefined)).toEqual(new Set())
  })
})

describe("seedableArguments", () => {
  it("puts a stored plain argument in its field", () => {
    const seeded = seedParameters(
      specs,
      seedableArguments(specs, {
        create_kwargs: { endpoint: "https://eu.api.lakera.ai" },
      }),
    )
    expect(seeded.values.endpoint).toBe("https://eu.api.lakera.ai")
  })

  it("never prefills a secret, even with the mask", () => {
    // The generic seeder would write `***` into the box, and a password box
    // shows that as a real three-character value, which a user cannot tell
    // from a cleared one.
    const seeded = seedParameters(
      specs,
      seedableArguments(specs, { create_kwargs: { api_key: REDACTED_SECRET } }),
    )
    expect(seeded.values.api_key).toBe("")
  })
})

describe("definitionFieldSpecs", () => {
  it("stops asking the user for a required secret the definition holds", () => {
    // Otherwise every later edit of another field is refused over a box that
    // is never prefilled.
    const relaxed = definitionFieldSpecs(specs, new Set(["api_key"]))
    expect(parameterErrors(relaxed, { api_key: "", endpoint: "" })).toEqual({})
  })

  it("still asks for a required secret the definition does not hold", () => {
    const strict = definitionFieldSpecs(specs, new Set())
    expect(
      parameterErrors(strict, { api_key: "", endpoint: "" }).api_key,
    ).toBeDefined()
  })

  it("says a held secret is kept when left blank", () => {
    const [key] = definitionFieldSpecs([apiKey], new Set(["api_key"]))
    expect(key?.description).toMatch(/blank to keep/i)
  })
})

describe("buildCreateKwargs", () => {
  it("sends the mask for a held secret left blank, which keeps it", () => {
    // Leaving the key out would clear the stored secret.
    expect(
      buildCreateKwargs(
        specs,
        { api_key: "", endpoint: "https://x" },
        new Set(["api_key"]),
      ),
    ).toEqual({ api_key: REDACTED_SECRET, endpoint: "https://x" })
  })

  it("sends a typed secret, which replaces it", () => {
    expect(
      buildCreateKwargs(
        specs,
        { api_key: "new-key", endpoint: "" },
        new Set(["api_key"]),
      ),
    ).toEqual({ api_key: "new-key" })
  })

  it("leaves out a blank secret the definition does not hold", () => {
    expect(
      buildCreateKwargs(
        [spec({ name: "token", secret: true })],
        { token: "" },
        new Set(),
      ),
    ).toEqual({})
  })

  it("answers an empty map, never null, for nothing set", () => {
    // `null` would read as "leave the stored arguments alone" on an update.
    expect(buildCreateKwargs([endpoint], { endpoint: "" }, new Set())).toEqual(
      {},
    )
  })
})
