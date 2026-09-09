// What one provider needs in `client_args` beyond an API key, described well
// enough for a form to ask for it.
//
// `client_args` is forwarded verbatim to the provider's client constructor
// (`AnyLLM.create(provider, api_key=…, api_base=…, **client_args)`), so its
// contents are whichever keyword arguments that SDK takes. For almost every
// provider that is optional transport tuning and a JSON textarea is the honest
// control. Bedrock is the exception: boto3 builds no client without
// `region_name`, and a credential omitted from a free-text JSON box surfaces
// as a provider error on the first request rather than as a validation error
// on the form.
//
// The registry lives here rather than being published by the gateway on
// purpose. `client_args` is a passthrough on every otari code path that reads
// an organization key, so there is no server-side call built from these specs
// for the form to drift away from; the authority for the names below is
// any-llm and boto3, which a server-side copy would be paraphrasing just as
// this one does. The endpoint that would carry it, `/v1/providers/catalog`, is
// `require_deployment_operator`-gated, and the page that most needs these
// fields (`/organization/provider-keys`) is used by organization owners and
// admins who hold no deployment authority. Publishing it there would mean
// either a 403 on the page that needs it or a new public route for what is
// static, per-release data with no deployment variance.
//
// The names themselves are the wire contract in `docs/hybrid-mode-protocol.md`
// and are read gateway-side by `src/gateway/services/bedrock_gateway_auth.py`;
// `providerCredentialFields.test.ts` pins them so a rename here has to be a
// deliberate edit in both places.

/** One `client_args` entry a provider expects, and how to ask for it. */
export interface ProviderCredentialFieldSpec {
  /** Literal SDK keyword argument, and the `client_args` key it is stored under. */
  key: string
  label: string
  isRequired: boolean
  /**
   * Holds a credential. Rendered masked, and never prefilled with what came
   * back from the API: the gateway returns `REDACTED_CLIENT_ARG` in place of
   * the stored value (`redact_secret_like_values`), and re-sending that mask
   * is what keeps the stored value (`restore_redacted_values`).
   */
  isSecret?: boolean
  placeholder?: string
  helpText: string
  /** A non-empty value must match this, checked before the form can submit. */
  pattern?: RegExp
  patternMessage?: string
  /**
   * Another field of the same provider that has to be filled in with this one.
   * Neither is required on its own, but half a pair is worse than none of it:
   * it reaches the SDK as a credential that cannot authenticate, and for
   * Bedrock it also decides which credential shape the gateway thinks is in
   * play (`bedrock_uses_bearer_token` keys on `aws_access_key_id` alone).
   */
  pairedWith?: string
}

/** A provider whose credential fields differ from the plain api_key + api_base pair. */
export interface ProviderCredentialSpec {
  fields: ProviderCredentialFieldSpec[]
  /** Overrides the "API key" label where the credential is not an opaque key. */
  apiKeyLabel?: string
  apiKeyHelpText?: string
}

// The value the gateway substitutes for a credential-shaped `client_args`
// entry when it serializes a key, and the value that means "keep what is
// stored" when it is sent back. Kept in step with `REDACTED_VALUE` in
// `src/gateway/models/secret_fields.py`.
export const REDACTED_CLIENT_ARG = "***"

const AWS_REGION_PATTERN = /^[a-z0-9-]+$/

const BEDROCK: ProviderCredentialSpec = {
  // Bedrock takes either of two credential shapes, and the region under both. A
  // bearer token ("Bedrock API key") goes in the API key field on its own; a
  // classic IAM pair puts its id and secret in the fields below and leaves the
  // API key field blank. Filling in both is ambiguous, which is what each
  // field's help text is for.
  apiKeyLabel: "Bedrock API key",
  apiKeyHelpText:
    "AWS's bearer token for Bedrock. Leave blank if you are using a classic IAM key pair below instead.",
  fields: [
    {
      key: "region_name",
      label: "AWS region",
      isRequired: true,
      placeholder: "us-east-1",
      helpText:
        "The region your Bedrock models are enabled in. Required for every credential shape: boto3 builds no client without one.",
      pattern: AWS_REGION_PATTERN,
      patternMessage:
        "A region is lowercase letters, digits and hyphens, like us-east-1.",
    },
    {
      key: "aws_access_key_id",
      label: "AWS access key ID",
      isRequired: false,
      pairedWith: "aws_secret_access_key",
      placeholder: "AKIAIOSFODNN7EXAMPLE",
      helpText:
        "Only for a classic IAM key pair, and then both halves are needed. Leave blank when the API key above is a Bedrock bearer token.",
    },
    {
      key: "aws_secret_access_key",
      label: "AWS secret access key",
      isRequired: false,
      isSecret: true,
      pairedWith: "aws_access_key_id",
      helpText:
        "The other half of the IAM key pair. Masked when read back, but stored unencrypted, unlike the API key above.",
    },
  ],
}

const PROVIDER_CREDENTIAL_SPECS: Record<string, ProviderCredentialSpec> = {
  bedrock: BEDROCK,
}

/** The credential spec for a provider, or undefined where api_key is the whole of it. */
export function credentialSpecFor(
  provider: string,
): ProviderCredentialSpec | undefined {
  return PROVIDER_CREDENTIAL_SPECS[provider]
}

/** The typed fields for a provider; empty for the vast majority of them. */
export function credentialFieldsFor(
  provider: string,
): ProviderCredentialFieldSpec[] {
  return PROVIDER_CREDENTIAL_SPECS[provider]?.fields ?? []
}

/**
 * Providers a stored credential can never authenticate, so a form that
 * collects one must not offer them.
 *
 * any-llm's SageMaker provider verifies against a bare `boto3.Session()`
 * before its client is built, so it authenticates from the gateway's own
 * ambient AWS chain and refuses outright when there is none, whatever was
 * supplied. otari already treats it that way on the dispatch side
 * (`_AMBIENT_CREDENTIAL_PROVIDERS` in `services/provider_kwargs.py` lets a
 * SageMaker selector resolve with nothing configured), and it reports no model
 * listing, so a connection test cannot tell an operator any of this either.
 *
 * Scoped to the call sites that collect a bring-your-own credential rather
 * than applied to every picker: a deployment operator adding a SageMaker
 * *instance* on `/providers` to give it an endpoint and a model list, backed
 * by the host's own credentials, is a configuration that does work.
 *
 * The old platform form excluded `platform` here too. otari has no such
 * pseudo-provider: `AnyLLM.get_supported_providers()` carries no `platform`
 * entry, so there is nothing to exclude.
 */
export const BYO_UNSUPPORTED_PROVIDERS: readonly string[] = ["sagemaker"]

/** The typed fields' current values, keyed by their SDK keyword argument. */
export type CredentialFieldValues = Record<string, string>

/**
 * Split stored `client_args` into the values the typed fields own and the rest,
 * which stays in the JSON escape hatch.
 *
 * A secret whose stored value came back masked is dropped from `typed` rather
 * than shown: a masked value in a password box reads as a real one three
 * characters long. `redacted` records that it was set, so
 * {@link mergeCredentialFields} can send the mask back and keep it.
 */
export function splitClientArgs(
  fields: ProviderCredentialFieldSpec[],
  stored: Record<string, unknown> | null | undefined,
): {
  typed: CredentialFieldValues
  rest: Record<string, unknown>
  redacted: string[]
} {
  const typed: CredentialFieldValues = {}
  const rest: Record<string, unknown> = {}
  const redacted: string[] = []
  const byKey = new Map(fields.map((field) => [field.key, field]))

  for (const [key, value] of Object.entries(stored ?? {})) {
    const field = byKey.get(key)
    if (!field) {
      rest[key] = value
      continue
    }
    if (field.isSecret && value === REDACTED_CLIENT_ARG) {
      redacted.push(key)
      continue
    }
    // Only a string can be edited in a text field. Anything else an operator
    // wrote under a registered name (a number, an object) keeps its meaning
    // only as JSON, so it stays in the escape hatch rather than being
    // stringified into a control that would write it back as text.
    if (typeof value === "string") typed[key] = value
    else rest[key] = value
  }
  return { typed, rest, redacted }
}

/**
 * Rebuild the `client_args` payload from the typed fields and whatever the JSON
 * textarea still holds. Null when nothing is left, which is how the API reads
 * "clear it".
 *
 * A blank secret whose stored value was masked is sent as the mask, so the
 * gateway keeps what it has: the same bargain the API key field makes by
 * omitting itself when it was not retyped.
 *
 * A filled-in typed field overwrites the same name in the JSON box, so the two
 * views cannot disagree about one option. A blank one writes nothing rather
 * than deleting: `splitClientArgs` hands the box only what a text control
 * cannot edit, and dropping that would silently discard a value the operator
 * can see.
 */
export function mergeCredentialFields(
  fields: ProviderCredentialFieldSpec[],
  values: CredentialFieldValues,
  rest: Record<string, unknown> | null,
  redacted: readonly string[] = [],
): Record<string, unknown> | null {
  const merged: Record<string, unknown> = { ...(rest ?? {}) }
  for (const field of fields) {
    const value = (values[field.key] ?? "").trim()
    if (value !== "") {
      merged[field.key] = value
    } else if (field.isSecret && redacted.includes(field.key)) {
      merged[field.key] = REDACTED_CLIENT_ARG
    }
  }
  return Object.keys(merged).length > 0 ? merged : null
}

/**
 * Per-field messages for what the form cannot submit, keyed by field. A secret
 * already stored (and therefore masked) counts as filled in, here as everywhere
 * else: the form is not shown it and is not asking for it again.
 */
export function validateCredentialFields(
  fields: ProviderCredentialFieldSpec[],
  values: CredentialFieldValues,
  redacted: readonly string[] = [],
): Record<string, string> {
  const errors: Record<string, string> = {}
  const byKey = new Map(fields.map((field) => [field.key, field]))
  const isFilled = (field: ProviderCredentialFieldSpec) =>
    (values[field.key] ?? "").trim() !== "" || redacted.includes(field.key)

  for (const field of fields) {
    const value = (values[field.key] ?? "").trim()
    if (value === "") {
      if (field.isRequired && !redacted.includes(field.key)) {
        errors[field.key] = `${field.label} is required for this provider.`
      }
      continue
    }
    if (field.pattern && !field.pattern.test(value)) {
      errors[field.key] =
        field.patternMessage ?? `${field.label} is not in the expected format.`
    }
  }

  // Half a pair, reported on the half that is missing. Declared from both
  // sides, so filling in either one alone asks for the other.
  for (const field of fields) {
    const partner = field.pairedWith ? byKey.get(field.pairedWith) : undefined
    if (!partner || !isFilled(field) || isFilled(partner)) continue
    errors[partner.key] ??=
      `${partner.label} is required alongside ${field.label}.`
  }
  return errors
}
