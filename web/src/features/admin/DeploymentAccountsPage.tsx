import { useMemo, useState } from "react"

import type { DeploymentUser } from "@/client"
import {
  useDeploymentAdminAccess,
  useDeploymentUsers,
  useUpdateDeploymentUser,
} from "@/shared/api/hooks"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import {
  Dot,
  PageIntro,
  RowAction,
  RowActionRow,
  TableScrollFrame,
} from "@/shared/components/surface"
import {
  EmptyState,
  ErrorBanner,
  InfoBanner,
  PageLoading,
} from "@/shared/components/ui"
import { formatRelative } from "@/shared/helpers/format"

import {
  accountLabel,
  accountLockoutReason,
  organizationSummary,
} from "./accounts"

// Every account on the deployment, which is the one identity list that is not
// scoped to an organization. The workspace roster and Members & roles both read
// through a membership and so cannot show an account whose memberships are all
// suspended, which is exactly the account an operator comes looking for; before
// this page the recourse was SQL.
//
// Three controls, matching the three the API serves: deactivate (which also ends
// that account's dashboard sessions), reactivate, and grant or remove operator
// access. Creating an account is not here, because an account without a
// membership can do nothing and memberships are the organization surface's; nor
// is deleting one, because historical attribution resolves through rows that
// hang off it.

/**
 * The access mark, affirmative: an account that reaches more than the ordinary
 * carries the accent, and one that does not carries the quiet dot rather than a
 * second kind of badge. Member is the unmarked state and reads as one.
 */
function AccessMark({ account }: { account: DeploymentUser }) {
  const { dot, ink, word } = account.is_bootstrap_operator
    ? { dot: "bg-accent", ink: "text-muted", word: "Bootstrap operator" }
    : account.is_superuser
      ? { dot: "bg-accent", ink: "text-muted", word: "Operator" }
      : { dot: "bg-text-subtle", ink: "text-subtle", word: "Member" }
  return (
    <span className={`flex items-center gap-2 font-mono text-[13px] ${ink}`}>
      <Dot className={dot} />
      {word.toUpperCase()}
    </span>
  )
}

export function DeploymentAccountsPage() {
  const access = useDeploymentAdminAccess()
  // Withheld until the gate answers: fetching the list first would put a 404 in
  // the console on every non-operator load to learn what `access` is about to
  // say, and the query would be discarded either way.
  const granted = access.data === true
  const accounts = useDeploymentUsers(granted)
  const update = useUpdateDeploymentUser()
  const [deactivating, setDeactivating] = useState<DeploymentUser | null>(null)

  const rows = accounts.data ?? []

  const columns = useMemo<DataTableColumn<DeploymentUser>[]>(
    () => [
      {
        id: "account",
        header: "Account",
        isRowHeader: true,
        cell: (account) => (
          <div className="flex flex-col gap-0.5">
            <span className="text-sm text-foreground">
              {accountLabel(account)}
            </span>
            {/* `full_name?.trim()`, matching `accountLabel`: a name that is
                only whitespace falls back to the email up there, so testing the
                untrimmed value here would print the same address twice. */}
            {account.full_name?.trim() && account.email ? (
              <span
                className="truncate text-xs text-muted"
                title={account.email}
              >
                {account.email}
              </span>
            ) : null}
          </div>
        ),
      },
      {
        id: "organizations",
        header: "Organizations",
        cell: (account) => (
          <span className="text-sm text-muted">
            {organizationSummary(account)}
          </span>
        ),
      },
      {
        id: "last-sign-in",
        header: "Last sign-in",
        // "never" rather than a dash: the column records dashboard sign-ins, and
        // an account that has never had one is a finding rather than missing
        // data. It stays "never" after the sessions expire, which is why the
        // gateway stores the stamp instead of deriving it from live sessions.
        cell: (account) => (
          <span className="text-sm text-muted">
            {formatRelative(account.last_sign_in_at)}
          </span>
        ),
      },
      {
        id: "status",
        header: "Status",
        // A deactivated account takes the danger dot with muted words: its
        // sessions are ended and it cannot sign in, which is worth noticing,
        // and it is also a state an operator chose rather than a fault.
        cell: (account) => (
          <span className="flex items-center gap-2 font-mono text-[13px] text-muted">
            <Dot className={account.is_active ? "bg-success" : "bg-danger"} />
            {account.is_active ? "ACTIVE" : "DEACTIVATED"}
          </span>
        ),
      },
      {
        id: "access",
        header: "Access",
        cell: (account) => <AccessMark account={account} />,
      },
      {
        id: "actions",
        header: "Actions",
        align: "end",
        cell: (account) => {
          const blocked = accountLockoutReason(account)
          return (
            <RowActionRow>
              {/* `title` reaches a mouse; the reason is folded into the
                  control's own name so it reaches everyone else too, as the
                  organization roster does it. A disabled control is not
                  focusable, so an `aria-describedby` would never be announced. */}
              <span title={account.is_superuser ? blocked : undefined}>
                <RowAction
                  isDisabled={
                    (account.is_superuser && blocked !== undefined) ||
                    update.isPending
                  }
                  ariaLabel={
                    account.is_superuser
                      ? `Remove operator access from ${accountLabel(account)}${blocked ? ` (${blocked})` : ""}`
                      : `Grant operator access to ${accountLabel(account)}`
                  }
                  onPress={() =>
                    update.mutate({
                      id: account.id,
                      body: { is_superuser: !account.is_superuser },
                    })
                  }
                >
                  {account.is_superuser ? "Remove operator" : "Make operator"}
                </RowAction>
              </span>
              <span title={account.is_active ? blocked : undefined}>
                <RowAction
                  isDisabled={
                    (account.is_active && blocked !== undefined) ||
                    update.isPending
                  }
                  ariaLabel={
                    account.is_active
                      ? `Deactivate ${accountLabel(account)}${blocked ? ` (${blocked})` : ""}`
                      : `Reactivate ${accountLabel(account)}`
                  }
                  onPress={() => {
                    if (account.is_active) {
                      setDeactivating(account)
                      return
                    }
                    update.mutate({
                      id: account.id,
                      body: { is_active: true },
                    })
                  }}
                >
                  {account.is_active ? "Deactivate" : "Reactivate"}
                </RowAction>
              </span>
            </RowActionRow>
          )
        },
      },
    ],
    [update],
  )

  if (access.isLoading) {
    return <PageLoading label="Loading accounts…" />
  }

  // A gate that failed to answer is not a gate that said no, so a failed request
  // is reported as one rather than falling into the refusal below, which would
  // tell an operator the page is not theirs because asking for it 500ed. Every
  // status reads that way, a 404 included: whether this deployment serves the
  // surface at all is the `surface` axis, settled before the shell routed here,
  // so inferring it from a status code would be a second answer to a question
  // the registry already answers.
  if (access.isError) {
    return (
      <div className="flex flex-col">
        <PageIntro title="Accounts" />
        <ErrorBanner error={access.error} />
      </div>
    )
  }

  // The API refuses a non-operator with 404 rather than 403, so a caller who is
  // not one lands here rather than being signed out. The sidebar drops the row
  // on the same answer, which makes this the state of somebody who arrived by
  // URL or whose access was taken away while the page was open.
  if (!granted) {
    return (
      <div className="flex flex-col">
        <PageIntro title="Accounts" />
        <EmptyState
          title="Accounts is not available to you"
          description="Managing the deployment's accounts is for its operators. Ask one of them if you need access here."
        />
      </div>
    )
  }

  return (
    <div className="flex flex-col">
      <PageIntro title="Accounts">
        Every account on this deployment, across all its organizations.
        Deactivating one ends its dashboard sessions immediately; operator
        access is what reaches this page.
      </PageIntro>

      <ErrorBanner error={accounts.error ?? update.error} />

      <InfoBanner>
        Your own account, and the bootstrap operator that master-key sign-in
        reaches this deployment through, cannot be deactivated or lose operator
        access here. Everything else on this page applies to them normally.
      </InfoBanner>

      <TableScrollFrame className="otari-accounts-table">
        <DataTable
          ariaLabel="Deployment accounts"
          columns={columns}
          rows={rows}
          getRowKey={(account) => account.id}
          isLoading={accounts.isLoading}
          emptyContent="No accounts yet."
        />
      </TableScrollFrame>

      <ConfirmDialog
        isOpen={deactivating !== null}
        onOpenChange={(open) => {
          if (!open) setDeactivating(null)
        }}
        heading="Deactivate account"
        body={
          <>
            Deactivate{" "}
            <strong>{deactivating ? accountLabel(deactivating) : ""}</strong>?
            Their dashboard sessions end straight away and they cannot sign in
            again until the account is reactivated. Their memberships and usage
            history are left exactly as they are. This closes the dashboard and
            not the API: any key they created belongs to its workspace and goes
            on working until it is revoked on Keys.
          </>
        }
        confirmLabel="Deactivate account"
        isPending={update.isPending}
        error={update.error}
        onConfirm={() => {
          if (deactivating) {
            update.mutate(
              { id: deactivating.id, body: { is_active: false } },
              { onSuccess: () => setDeactivating(null) },
            )
          }
        }}
      />
    </div>
  )
}
