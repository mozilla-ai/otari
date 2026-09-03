/**
 * The invitee's inbox: which organizations are waiting on the signed-in
 * identity, and the two things they can do about each (otari-ai#1999).
 *
 * The counterpart to `AcceptInvitationPage`, and deliberately a separate page
 * rather than a mode of it. That one is reached before the auth gate by
 * somebody holding nothing but a token, and everything about it follows from
 * that: it takes no identity, mints no session, and hands off to the claim
 * flow. This one is behind the gate, addresses an invitation by membership id
 * rather than by token, and exists because losing the email used to leave an
 * invitation reachable by nothing at all.
 *
 * Not in a sidebar rail. It is a chrome destination like `/account` and
 * `/docs`: a permanent row for something that is empty almost always would
 * read as a section of the product. The organization switcher surfaces it
 * instead, and only while something is actually waiting.
 */

import { Button, Card } from "@heroui/react"
import { useState } from "react"
import { FiMail } from "react-icons/fi"

import type { PendingOrganizationInvitation } from "@/client"
import { membershipLabel } from "@/features/organization/roles"
import {
  useAcceptPendingMembership,
  useDeclinePendingMembership,
  usePendingOrganizationInvitations,
} from "@/shared/api/hooks"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import {
  EmptyState,
  ErrorBanner,
  PageHeader,
  PageLoading,
} from "@/shared/components/ui"
import { formatDateTime } from "@/shared/helpers/format"

export function PendingInvitationsPage() {
  const invitations = usePendingOrganizationInvitations()
  const accept = useAcceptPendingMembership()
  const decline = useDeclinePendingMembership()
  const [declining, setDeclining] = useState<
    PendingOrganizationInvitation | undefined
  >(undefined)
  // Which row's accept is in flight. `accept.isPending` alone would put every
  // row's button in a pending state, because one mutation backs the whole list.
  const [accepting, setAccepting] = useState<string | undefined>(undefined)

  const waiting = invitations.data ?? []
  // Whether the server has actually answered, as against the query having
  // settled. A failed read settles too, and reading `isFetched` alone would
  // put "nothing is waiting" on screen next to the error banner saying the
  // list could not be read, which is a claim this page has no basis for.
  const answered = invitations.data !== undefined

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Invitations"
        description="Organizations that have invited you. Accepting adds the membership to your organization switcher; declining ends the invitation and stops its emailed link from working."
      />

      {/* `&& !data` on both guards, so a refetch behind an accept keeps the
          list on screen instead of collapsing to a spinner and back. */}
      {invitations.isError && !invitations.data ? (
        <ErrorBanner error={invitations.error} />
      ) : null}
      {invitations.isPending && !invitations.data ? (
        <PageLoading label="Loading invitations…" />
      ) : null}

      {answered && waiting.length === 0 ? (
        <EmptyState
          title="No invitations waiting"
          description="When an organization invites you, it appears here as well as in the email it sends, so a link you never received is not the only way in."
        />
      ) : null}

      {waiting.length > 0 ? (
        <ul className="flex flex-col gap-4">
          {waiting.map((invitation) => (
            <li key={invitation.organization_member_id}>
              <Card>
                <Card.Content className="flex flex-col gap-4 p-6 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-start gap-3">
                    <FiMail
                      aria-hidden="true"
                      className="mt-0.5 size-5 shrink-0 text-muted"
                    />
                    <div className="flex flex-col gap-1">
                      <h2 className="text-title">
                        {invitation.organization_name}
                      </h2>
                      <p className="text-sm text-muted">
                        Invited as {membershipLabel(invitation.role)} to{" "}
                        {invitation.email}
                      </p>
                      <p className="text-caption">
                        Expires {formatDateTime(invitation.expires_at)}
                      </p>
                    </div>
                  </div>
                  <div className="flex shrink-0 flex-wrap gap-2">
                    <Button
                      variant="primary"
                      isPending={
                        accepting === invitation.organization_member_id &&
                        accept.isPending
                      }
                      onPress={() => {
                        setAccepting(invitation.organization_member_id)
                        accept.mutate(invitation.organization_member_id, {
                          onSettled: () => setAccepting(undefined),
                        })
                      }}
                    >
                      Accept
                    </Button>
                    <Button
                      variant="outline"
                      onPress={() => setDeclining(invitation)}
                    >
                      Decline
                    </Button>
                  </div>
                </Card.Content>
              </Card>
            </li>
          ))}
        </ul>
      ) : null}

      {/* The accept error belongs under the list rather than in the row: the
          row is gone on success, and a refusal here is nearly always "this
          invitation is no longer valid", which is about the list. */}
      <ErrorBanner error={accept.error} />

      <ConfirmDialog
        isOpen={declining !== undefined}
        onOpenChange={(open) => {
          if (!open) setDeclining(undefined)
        }}
        heading="Decline invitation"
        body={
          <>
            Decline the invitation to{" "}
            <strong>{declining?.organization_name ?? ""}</strong>? The emailed
            link stops working, so this cannot be undone from your side.
            Somebody in that organization can invite you again.
          </>
        }
        confirmLabel="Decline invitation"
        isPending={decline.isPending}
        error={decline.error}
        onConfirm={() => {
          if (declining) {
            decline.mutate(declining.organization_member_id, {
              onSuccess: () => setDeclining(undefined),
            })
          }
        }}
      />
    </div>
  )
}
