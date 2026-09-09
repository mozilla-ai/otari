import { describe, expect, it } from "vitest"

import {
  BYO_UNSUPPORTED_PROVIDERS,
  credentialFieldsFor,
  credentialSpecFor,
  mergeCredentialFields,
  REDACTED_CLIENT_ARG,
  splitClientArgs,
  validateCredentialFields,
} from "./providerCredentialFields"

const bedrock = () => credentialFieldsFor("bedrock")

describe("the registry", () => {
  // boto3's own constructor keyword arguments, spread into AnyLLM.create by
  // gateway/services/provider_kwargs.py. Pinned so an edit here is a deliberate
  // one; nothing checks them against the SDK that consumes them.
  it("names Bedrock's fields with the kwargs boto3 takes", () => {
    expect(bedrock().map((field) => field.key)).toEqual([
      "region_name",
      "aws_access_key_id",
      "aws_secret_access_key",
    ])
  })

  it("requires only the region, which every Bedrock credential shape needs", () => {
    expect(
      bedrock()
        .filter((field) => field.isRequired)
        .map((field) => field.key),
    ).toEqual(["region_name"])
  })

  it("masks the input for the IAM secret and not for the access key id", () => {
    // `isSecret` decides how the control renders. What the gateway masks on
    // read is a wider, key-name rule that catches the id as well.
    const byKey = new Map(bedrock().map((field) => [field.key, field]))
    expect(byKey.get("aws_secret_access_key")?.isSecret).toBe(true)
    expect(byKey.get("aws_access_key_id")?.isSecret).toBeUndefined()
  })

  it("pairs the two IAM fields from both sides", () => {
    const byKey = new Map(bedrock().map((field) => [field.key, field]))
    expect(byKey.get("aws_access_key_id")?.pairedWith).toBe(
      "aws_secret_access_key",
    )
    expect(byKey.get("aws_secret_access_key")?.pairedWith).toBe(
      "aws_access_key_id",
    )
    expect(byKey.get("region_name")?.pairedWith).toBeUndefined()
  })

  it("renames the API key field where the provider does not call it one", () => {
    expect(credentialSpecFor("bedrock")?.apiKeyLabel).toBe("Bedrock API key")
    expect(credentialSpecFor("openai")).toBeUndefined()
  })

  it("leaves the providers that need nothing beyond an API key with no fields", () => {
    expect(credentialFieldsFor("openai")).toEqual([])
    expect(credentialFieldsFor("")).toEqual([])
  })

  it("withholds only the provider whose stored credential can never authenticate", () => {
    expect(BYO_UNSUPPORTED_PROVIDERS).toEqual(["sagemaker"])
  })
})

describe("splitClientArgs", () => {
  it("routes registered keys to the typed fields and the rest to the JSON box", () => {
    expect(
      splitClientArgs(bedrock(), { region_name: "us-east-1", timeout: 1800 }),
    ).toEqual({
      typed: { region_name: "us-east-1" },
      rest: { timeout: 1800 },
      redacted: [],
    })
  })

  it("records a masked value as set instead of putting the mask in a control", () => {
    // The gateway masks by key name, so both halves of the IAM pair come back
    // as the mask: `aws_access_key_id` contains "key". Neither may be prefilled
    // with it, or the operator reads a three-character placeholder as the value
    // they stored.
    const split = splitClientArgs(bedrock(), {
      region_name: "eu-west-1",
      aws_access_key_id: REDACTED_CLIENT_ARG,
      aws_secret_access_key: REDACTED_CLIENT_ARG,
    })
    expect(split.typed).toEqual({ region_name: "eu-west-1" })
    expect(split.redacted).toEqual([
      "aws_access_key_id",
      "aws_secret_access_key",
    ])
  })

  it("leaves a non-string under a registered name in the JSON box", () => {
    // A text control would write it back as a string and change its meaning.
    expect(splitClientArgs(bedrock(), { region_name: 42 }).rest).toEqual({
      region_name: 42,
    })
  })

  it("reads a key with no stored options as empty", () => {
    expect(splitClientArgs(bedrock(), null)).toEqual({
      typed: {},
      rest: {},
      redacted: [],
    })
  })
})

describe("mergeCredentialFields", () => {
  it("puts the typed values back beside whatever the JSON box still holds", () => {
    expect(
      mergeCredentialFields(
        { region_name: "us-east-1" },
        {
          timeout: 1800,
        },
      ),
    ).toEqual({ region_name: "us-east-1", timeout: 1800 })
  })

  it("trims a value rather than storing the whitespace around it", () => {
    expect(mergeCredentialFields({ region_name: " us-east-1 " }, null)).toEqual(
      { region_name: "us-east-1" },
    )
  })

  it("sends the mask back for a stored value left blank, so the gateway keeps it", () => {
    expect(
      mergeCredentialFields(
        { region_name: "us-east-1", aws_secret_access_key: "" },
        null,
        ["aws_access_key_id", "aws_secret_access_key"],
      ),
    ).toEqual({
      region_name: "us-east-1",
      aws_access_key_id: REDACTED_CLIENT_ARG,
      aws_secret_access_key: REDACTED_CLIENT_ARG,
    })
  })

  it("keeps a value whose field has stopped rendering", () => {
    // `/providers` lets an instance's provider type be retyped mid-edit, which
    // changes the field list under values that are already out of client_args.
    // Dropping them here would delete the stored credential on save.
    expect(
      mergeCredentialFields({ region_name: "us-east-1" }, { timeout: 1800 }, [
        "aws_secret_access_key",
      ]),
    ).toEqual({
      region_name: "us-east-1",
      aws_secret_access_key: REDACTED_CLIENT_ARG,
      timeout: 1800,
    })
  })

  it("lets a filled-in field win over the same name in the JSON box", () => {
    expect(
      mergeCredentialFields(
        { region_name: "us-east-1" },
        {
          region_name: "eu-west-1",
        },
      ),
    ).toEqual({ region_name: "us-east-1" })
  })

  it("leaves the JSON box alone where its field is blank", () => {
    // The box only ever holds what a text control cannot edit, so a blank field
    // is not an instruction to delete what is sitting there.
    expect(
      mergeCredentialFields(
        { region_name: "" },
        {
          region_name: 42,
        },
      ),
    ).toEqual({ region_name: 42 })
  })

  it("reads an empty result as null, which is how the API clears the column", () => {
    expect(mergeCredentialFields({}, null)).toBeNull()
  })
})

describe("validateCredentialFields", () => {
  it("refuses a Bedrock credential with no region", () => {
    expect(validateCredentialFields(bedrock(), {})).toEqual({
      region_name: "AWS region is required for this provider.",
    })
  })

  it("treats whitespace as absent", () => {
    expect(validateCredentialFields(bedrock(), { region_name: "  " })).toEqual({
      region_name: "AWS region is required for this provider.",
    })
  })

  it("rejects a region that is not shaped like one", () => {
    // boto3 raises InvalidRegionError while building the client, well past
    // anything that could tell an operator which field was wrong.
    expect(
      validateCredentialFields(bedrock(), { region_name: "US East 1" }),
    ).toEqual({
      region_name:
        "A region is lowercase letters, digits and hyphens, like us-east-1.",
    })
  })

  it("accepts a filled-in classic IAM pair", () => {
    expect(
      validateCredentialFields(bedrock(), {
        region_name: "ap-southeast-2",
        aws_access_key_id: "AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key: "secret",
      }),
    ).toEqual({})
  })

  it("accepts the bearer-token shape, which fills in neither IAM field", () => {
    // The API key field carries the whole credential there, so an empty pair is
    // a complete answer rather than a half-filled one.
    expect(
      validateCredentialFields(bedrock(), { region_name: "us-east-1" }),
    ).toEqual({})
  })

  it("asks for the secret when only the access key id is given", () => {
    // An id with no secret reaches boto3 as a credential that cannot sign, and
    // the gateway reads the id alone as "this is the classic IAM shape".
    expect(
      validateCredentialFields(bedrock(), {
        region_name: "us-east-1",
        aws_access_key_id: "AKIAIOSFODNN7EXAMPLE",
      }),
    ).toEqual({
      aws_secret_access_key:
        "AWS secret access key is required alongside AWS access key ID.",
    })
  })

  it("asks for the access key id when only the secret is given", () => {
    expect(
      validateCredentialFields(bedrock(), {
        region_name: "us-east-1",
        aws_secret_access_key: "secret",
      }),
    ).toEqual({
      aws_access_key_id:
        "AWS access key ID is required alongside AWS secret access key.",
    })
  })

  it("counts a stored, masked half of the pair as filled in", () => {
    expect(
      validateCredentialFields(
        bedrock(),
        { region_name: "us-east-1", aws_access_key_id: "AKIA…" },
        ["aws_secret_access_key"],
      ),
    ).toEqual({})
  })

  it("counts an already-stored required secret as filled in", () => {
    const fields = [
      {
        key: "token",
        label: "Token",
        isRequired: true,
        isSecret: true,
        helpText: "",
      },
    ]
    expect(validateCredentialFields(fields, {}, ["token"])).toEqual({})
    expect(validateCredentialFields(fields, {})).toEqual({
      token: "Token is required for this provider.",
    })
  })

  it("asks nothing of a provider with no registered fields", () => {
    expect(validateCredentialFields(credentialFieldsFor("openai"), {})).toEqual(
      {},
    )
  })
})
