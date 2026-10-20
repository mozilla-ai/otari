import { Button } from "@heroui/react"
import { useState } from "react"
import type { InvitationPreview } from "@/client"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { AuthPasswordField, AuthTextField } from "@/features/auth/AuthFields"
import { TermsAcceptanceField } from "@/features/auth/TermsAcceptanceField"
import type { useAcceptInvitation } from "@/shared/api/organizations"
import {
  MAX_PASSWORD_BYTES,
  MIN_PASSWORD_LENGTH,
  newPasswordProblem,
} from "@/shared/helpers/password"
import { useDeployment } from "@/shared/hooks/useDeployment"

// Accepts and sets the first password in one call, for an address that has
// never signed in. Kept apart from `SignupPage`: that form ends in a
// verification email, and this one ends ready to sign in. The published terms
// gate the provider way in as well: that sign-in is the address's first, which
// the gateway refuses without them, so the accept carries the acceptance on to
// the page that offers the providers.
export function ClaimForm({
  token,
  preview,
  accept,
  offersProviderSignIn,
}: {
  token: string
  preview: InvitationPreview
  accept: ReturnType<typeof useAcceptInvitation>
  offersProviderSignIn: boolean
}) {
  const { terms_url } = useDeployment()
  const [fullName, setFullName] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [isTermsAccepted, setIsTermsAccepted] = useState(false)

  const problem = newPasswordProblem(password, confirmPassword)
  const awaitsTerms = terms_url !== null && !isTermsAccepted
  const canSubmit =
    password !== "" &&
    confirmPassword !== "" &&
    problem === null &&
    !awaitsTerms

  const submit = () => {
    if (!canSubmit || accept.isPending) return
    accept.mutate({
      token,
      password,
      full_name: fullName.trim() || null,
      // Present only where a document was actually shown, as on signup.
      ...(terms_url !== null ? { terms_accepted: true } : {}),
    })
  }

  return (
    <form
      className="flex flex-col gap-3"
      onSubmit={(event) => {
        event.preventDefault()
        submit()
      }}
    >
      <p className="text-caption">
        Set a password to sign in as {preview.email}.
      </p>
      <AuthTextField
        label="Full name (optional)"
        value={fullName}
        onChange={setFullName}
        autoComplete="name"
      />
      <AuthPasswordField
        label="Password"
        value={password}
        onChange={setPassword}
        autoComplete="new-password"
        description={`At least ${MIN_PASSWORD_LENGTH} characters, and at most ${MAX_PASSWORD_BYTES} bytes.`}
        errorMessage={problem ?? undefined}
      />
      <AuthPasswordField
        label="Confirm password"
        value={confirmPassword}
        onChange={setConfirmPassword}
        autoComplete="new-password"
      />
      {terms_url !== null ? (
        <TermsAcceptanceField
          termsUrl={terms_url}
          isAccepted={isTermsAccepted}
          onChange={setIsTermsAccepted}
        />
      ) : null}

      <ErrorBanner error={accept.error} />

      <Button
        type="submit"
        variant="primary"
        fullWidth
        isPending={accept.isPending}
        isDisabled={!canSubmit}
      >
        Accept and set password
      </Button>
      {offersProviderSignIn ? (
        <Button
          variant="ghost"
          fullWidth
          isDisabled={accept.isPending || awaitsTerms}
          onPress={() =>
            accept.mutate({
              token,
              ...(terms_url !== null ? { terms_accepted: true } : {}),
            })
          }
        >
          Accept and sign in with a provider instead
        </Button>
      ) : null}
    </form>
  )
}
