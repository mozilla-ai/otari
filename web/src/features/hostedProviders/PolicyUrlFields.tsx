import { Field } from "@/design-system/forms/Field"

import { policyUrlError } from "./policyUrls"

/**
 * The provider's Terms of service and Privacy policy, linked from its page on
 * Providers. Both optional; an address the server would refuse is named here
 * before the save.
 */
export function PolicyUrlFields({
  termsUrl,
  privacyUrl,
  onTermsUrlChange,
  onPrivacyUrlChange,
}: {
  termsUrl: string
  privacyUrl: string
  onTermsUrlChange: (value: string) => void
  onPrivacyUrlChange: (value: string) => void
}) {
  const termsError = policyUrlError(termsUrl)
  const privacyError = policyUrlError(privacyUrl)
  return (
    <>
      <Field
        label="Terms of service URL"
        value={termsUrl}
        onChange={onTermsUrlChange}
        placeholder="https://example.com/terms"
        description="Optional. Linked from the provider's page on Providers."
        isInvalid={termsError !== undefined}
        errorMessage={termsError}
      />
      <Field
        label="Privacy policy URL"
        value={privacyUrl}
        onChange={onPrivacyUrlChange}
        placeholder="https://example.com/privacy"
        description="Optional. Linked from the provider's page on Providers."
        isInvalid={privacyError !== undefined}
        errorMessage={privacyError}
      />
    </>
  )
}
