import type { DeploymentUser } from "@/client"
import { CopyableValue } from "@/design-system/actions/CopyField"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { useIssuePasswordLink } from "@/shared/api/deployment"
import { absoluteDashboardLink } from "@/shared/helpers/dashboardLink"
import { formatDateTime } from "@/shared/helpers/format"
import { useDeployment } from "@/shared/hooks/useDeployment"

import { accountLabel } from "./accounts"

/**
 * Give one account a single-use link that sets its password.
 *
 * The way back for somebody who forgot their password where "Forgot your
 * password?" cannot help, which is every deployment that sends no mail. Where
 * mail goes out the link goes to the account's own address, the same as that
 * flow; otherwise, or when the send failed, the link is shown here to pass on,
 * the way an invitation's accept link is.
 */
export function PasswordLinkDialog({
  account,
  onClose,
}: {
  /** The account to issue the link for; the dialog is closed while unset. */
  account: DeploymentUser | undefined
  onClose: () => void
}) {
  const { mail_ready } = useDeployment()
  const issue = useIssuePasswordLink()
  const result = issue.data
  const name = account ? accountLabel(account) : ""
  const close = () => {
    issue.reset()
    onClose()
  }

  if (account && result) {
    const link = result.link ? absoluteDashboardLink(result.link) : ""
    return (
      <FormDialog
        isOpen
        onOpenChange={(open) => {
          if (!open) close()
        }}
        title="Password link"
        // A link shown here is shown nowhere else, so acknowledging it is the
        // way out rather than one of two.
        isDismissable={result.mail_sent}
        submitLabel="Done"
        hideCancel
        onSubmit={close}
        isPending={false}
      >
        <InfoBanner>
          {result.mail_sent ? (
            <>
              An email with a link to set a new password was sent to{" "}
              <strong>{account.email}</strong>.
            </>
          ) : (
            // Not "mail isn't configured": the send can also have failed.
            <>
              Otari did not send the email. Share this link with{" "}
              <strong>{name}</strong> yourself.
            </>
          )}
        </InfoBanner>
        {link ? (
          <>
            <CopyableValue value={link} label="Password link">
              <span className="break-all text-xs">{link}</span>
            </CopyableValue>
            <p className="text-caption">
              Whoever opens it can set this account's password, so send it only
              to {account.email}. It works once, until{" "}
              {formatDateTime(result.expires_at)}.
            </p>
          </>
        ) : null}
      </FormDialog>
    )
  }

  return (
    <FormDialog
      isOpen={account !== undefined}
      onOpenChange={(open) => {
        if (!open) close()
      }}
      title="Send a password link"
      submitLabel={mail_ready ? "Email the link" : "Create the link"}
      onSubmit={() => {
        if (account) issue.mutate(account.id)
      }}
      isPending={issue.isPending}
      error={issue.error}
    >
      <p className="text-body">
        Give <strong>{name}</strong> a single-use link that sets a new password,
        for when they cannot sign in.{" "}
        {mail_ready
          ? `It is emailed to ${account?.email ?? ""}.`
          : "This deployment sends no mail, so you get the link to share with them yourself."}{" "}
        Any earlier reset link for this account stops working.
      </p>
    </FormDialog>
  )
}
