import { Button } from "@heroui/react"
import { useState } from "react"
import type { InvitationPreview } from "@/client"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { AuthPasswordField, AuthTextField } from "@/features/auth/AuthFields"
import type { useAcceptInvitation } from "@/shared/api/organizations"
import {
  MAX_PASSWORD_BYTES,
  MIN_PASSWORD_LENGTH,
  newPasswordProblem,
} from "@/shared/helpers/password"
import { useDeployment } from "@/shared/hooks/useDeployment"

// Accepts and sets the first password in one call, for an address that has
// never signed in. Kept apart from `SignupPage`: that form ends in a
// verification email, and this one ends ready to sign in.
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
  const canSubmit =
    password !== "" &&
    confirmPassword !== "" &&
    problem === null &&
    (terms_url === null || isTermsAccepted)

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
      {/* Beside the checkbox rather than inside its label, for the reason
          `SignupPage` gives: nested, the link only ticked the box. */}
      {terms_url !== null ? (
        <div className="flex flex-wrap items-center gap-x-1 text-caption">
          <Checkbox
            isSelected={isTermsAccepted}
            onChange={setIsTermsAccepted}
            ariaLabel="I accept the terms of service"
          >
            <span className="text-caption">I accept the</span>
          </Checkbox>
          <span>
            <a
              href={terms_url}
              target="_blank"
              rel="noreferrer"
              className="font-medium text-link hover:text-link-hover"
            >
              terms of service
            </a>
            .
          </span>
        </div>
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
          isDisabled={accept.isPending}
          onPress={() => accept.mutate({ token })}
        >
          Accept and sign in with a provider instead
        </Button>
      ) : null}
    </form>
  )
}
