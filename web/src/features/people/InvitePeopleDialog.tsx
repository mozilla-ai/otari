import { useEffect, useRef, useState } from "react"
import type {
  BulkInviteOrganizationMembersResult,
  MembershipRole,
} from "@/client"
import { CopyButton } from "@/design-system/actions/CopyButton"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { RadioGroup } from "@/design-system/forms/RadioGroup"
import { Select } from "@/design-system/forms/Select"
import { TextArea } from "@/design-system/forms/TextArea"
import {
  asMembershipRole,
  grantableRoleOptions,
} from "@/features/organization/roles"
import {
  addressProblem,
  inviteLabel,
  needsOutcome,
  outcomeCopy,
  teamHelp,
  teamRoleFor,
} from "@/features/people/invite"
import {
  useBulkInviteOrganizationMembers,
  useOrganizationContext,
} from "@/shared/api/organizations"
import { useWorkspaces } from "@/shared/api/workspaces"
import { parseAddresses } from "@/shared/helpers/addresses"
import { absoluteDashboardLink } from "@/shared/helpers/dashboardLink"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useDeployment } from "@/shared/hooks/useDeployment"

/**
 * Invite people: addresses, a role, and the workspace they join once
 * they accept. Every address goes through the organization's invitation
 * endpoint, one call for all of them, once each one looks like an address.
 * Where every email went out the dialog just closes, and the invitations are
 * on the list; otherwise it says which addresses failed and hands over the
 * links nobody was emailed.
 */
export function InvitePeopleDialog({
  isOpen,
  onOpenChange,
}: {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
}) {
  const invite = useBulkInviteOrganizationMembers()
  const { mail_ready } = useDeployment()
  const context = useOrganizationContext().data
  const organization = context?.organization.name
  const roles = grantableRoleOptions(context?.role)
  const workspaces = useWorkspaces().data ?? []
  const { selected } = useSelectedWorkspace()
  const [emails, setEmails] = useState("")
  const [role, setRole] = useState<MembershipRole>("member")
  const [team, setTeam] = useState("")
  // Checked on blur and on submit, never while typing; once shown it follows
  // the text, so fixing the address clears it.
  const [isChecked, setChecked] = useState(false)
  const [outcome, setOutcome] = useState<BulkInviteOrganizationMembersResult>()
  const addressesRef = useRef<HTMLDivElement>(null)
  // The field does not pass its blur on, so it is heard on the textarea
  // itself. Leaving it empty is not a mistake yet; submitting it empty is.
  // Bound again whenever the form view mounts its textarea.
  useEffect(() => {
    const field = addressesRef.current?.querySelector("textarea")
    if (!isOpen || outcome || !field) return
    const check = () => {
      if (field.value.trim() !== "") setChecked(true)
    }
    field.addEventListener("blur", check)
    return () => field.removeEventListener("blur", check)
  }, [isOpen, outcome])
  const addresses = parseAddresses(emails)
  const problem = addressProblem(emails)
  const shownProblem = isChecked ? problem : undefined
  const teamId =
    team ||
    workspaces.find((one) => one.id === selected?.workspace_id)?.id ||
    workspaces[0]?.id ||
    ""

  const close = () => {
    onOpenChange(false)
    setEmails("")
    setRole("member")
    setTeam("")
    setChecked(false)
    setOutcome(undefined)
    invite.reset()
  }

  const submit = () => {
    setChecked(true)
    if (problem) {
      addressesRef.current?.querySelector("textarea")?.focus()
      return
    }
    invite.mutate(
      {
        emails: addresses,
        role,
        workspace_assignments: teamId
          ? [{ workspace_id: teamId, role: teamRoleFor(role) }]
          : null,
      },
      {
        onSuccess: (result) => {
          if (needsOutcome(result)) setOutcome(result)
          else close()
        },
      },
    )
  }

  if (outcome) {
    const unsent = outcome.invited.filter(
      (one) => !one.mail_sent && one.accept_link,
    )
    const copy = outcomeCopy(outcome)
    return (
      <FormDialog
        isOpen={isOpen}
        onOpenChange={(open) => {
          if (!open) close()
        }}
        title={copy.title}
        description={copy.description}
        isDismissable={unsent.length === 0}
        submitLabel="Done"
        hideCancel
        isPending={false}
        onSubmit={close}
      >
        <ul className="flex flex-col border-y border-border-subtle">
          {outcome.failed.map((one) => (
            <li
              key={one.email}
              className="flex flex-col gap-0.5 border-b border-border-subtle py-3 last:border-b-0"
            >
              <span className="break-all text-sm text-foreground">
                {one.email}
              </span>
              <span className="text-caption text-danger">{one.detail}</span>
            </li>
          ))}
          {unsent.map((one) => {
            const link = absoluteDashboardLink(one.accept_link ?? "")
            return (
              <li
                key={one.invitation_id}
                className="flex flex-col gap-0.5 border-b border-border-subtle py-3 last:border-b-0"
              >
                <span className="break-all text-sm text-foreground">
                  {one.email}
                </span>
                <span className="flex items-start gap-2">
                  <span className="min-w-0 flex-1 break-all text-caption">
                    {link}
                  </span>
                  <CopyButton value={link} label={`Copy ${one.email}’s link`} />
                </span>
              </li>
            )
          })}
        </ul>
      </FormDialog>
    )
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) close()
        else onOpenChange(true)
      }}
      title="Invite people"
      description={
        mail_ready
          ? `They get an email with an invitation link to join ${organization ?? "the organization"}.`
          : "This deployment sends no email, so you get a link to share with each of them."
      }
      submitLabel={inviteLabel(addresses.length)}
      isPending={invite.isPending}
      error={invite.error}
      isDirty={emails.length > 0}
      onSubmit={submit}
    >
      <div className="flex flex-col gap-6">
        <div ref={addressesRef}>
          <TextArea
            label="Email addresses"
            value={emails}
            onChange={setEmails}
            placeholder="ana@example.com, li@example.com"
            rows={3}
            description={
              shownProblem
                ? undefined
                : "Separate them with commas, spaces or new lines."
            }
            isInvalid={shownProblem !== undefined}
            errorMessage={shownProblem}
            shouldReserveMessage={false}
          />
        </div>
        <RadioGroup
          label="Role"
          value={role}
          onChange={(value) => setRole(asMembershipRole(value) ?? "member")}
          options={roles}
        />
        {workspaces.length > 0 ? (
          <Select
            label="Workspace"
            value={teamId}
            onChange={setTeam}
            options={workspaces.map((one) => ({
              value: one.id,
              label: one.name,
            }))}
            description={teamHelp(role)}
            shouldReserveMessage={false}
          />
        ) : null}
      </div>
    </FormDialog>
  )
}
