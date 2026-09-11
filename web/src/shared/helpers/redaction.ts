/**
 * The value the gateway substitutes for a credential-shaped entry in a
 * free-form settings dict, and the value that means "keep what is stored" when
 * it is sent back.
 *
 * Kept in step with `REDACTED_VALUE` in `src/gateway/models/secret_fields.py`,
 * which is the one place that decides it. Shared because more than one feature
 * edits such a dict: provider `client_args` and an organization guardrail's
 * `validate_kwargs` round-trip through the same rule.
 */
export const REDACTED_SECRET = "***"
