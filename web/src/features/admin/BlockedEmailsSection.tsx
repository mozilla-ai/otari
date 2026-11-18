import { useState } from "react"
import { FiUnlock } from "react-icons/fi"

import type { BlockedEmail, BlockedEmailCreated } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmRowAction } from "@/design-system/actions/ConfirmRowAction"
import { RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import {
  PAGE_SIZE_OPTIONS,
  TablePagination,
} from "@/design-system/data/TablePagination"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Field } from "@/design-system/forms/Field"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import {
  useBlockEmail,
  useBlockedEmails,
  useUnblockEmail,
} from "@/shared/api/deployment"
import { formatRelative } from "@/shared/helpers/format"

// The addresses and domains that may not hold an account here. The server is
// the whole enforcement: it refuses signup, adding, inviting and every sign-in
// for what an entry matches, and ends the matching accounts' sessions when the
// entry is added. This section only manages the list.

function blockedSummary(created: BlockedEmailCreated): string {
  const count = created.blocked_account_count
  if (count === 0)
    return `Blocked ${created.pattern}. No existing account matched.`
  const accounts =
    count === 1 ? "1 existing account" : `${count} existing accounts`
  return `Blocked ${created.pattern}. ${accounts} can no longer sign in, and their sessions were ended.`
}

export function BlockedEmailsSection() {
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState<number>(PAGE_SIZE_OPTIONS[0])
  const entries = useBlockedEmails({ page, pageSize })
  const block = useBlockEmail()
  const unblock = useUnblockEmail()
  const [value, setValue] = useState("")
  const [reason, setReason] = useState("")
  const [lastBlocked, setLastBlocked] = useState<BlockedEmailCreated>()

  const rows = entries.data?.data ?? []
  // Clamped to the refreshed total, so a page emptied by an unblock, here or by another operator, is left for
  // the last one that still has rows.
  const lastPage =
    entries.data === undefined
      ? page
      : Math.max(0, Math.ceil(entries.data.count / pageSize) - 1)
  if (page > lastPage) setPage(lastPage)

  const submit = () => {
    const entry = value.trim()
    if (entry === "") return
    block.mutate(
      { value: entry, reason: reason.trim() || null },
      {
        onSuccess: (created) => {
          setLastBlocked(created)
          setValue("")
          setReason("")
        },
      },
    )
  }

  const columns: DataTableColumn<BlockedEmail>[] = [
    {
      id: "entry",
      header: "Address or domain",
      isRowHeader: true,
      cell: (entry) => (
        <div className="flex flex-col gap-0.5">
          <span className="text-body">{entry.pattern}</span>
          <span className="text-mono-caption text-subtle">
            {entry.kind === "domain" ? "DOMAIN AND SUBDOMAINS" : "ADDRESS"}
          </span>
        </div>
      ),
    },
    {
      id: "reason",
      header: "Reason",
      cell: (entry) => (
        <span className="text-sm text-muted">{entry.reason ?? "—"}</span>
      ),
    },
    {
      id: "added",
      header: "Added",
      cell: (entry) => (
        <span className="text-sm text-muted">
          {formatRelative(entry.created_at)}
        </span>
      ),
    },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (entry) => (
        <RowActionRow>
          <ConfirmRowAction
            icon={FiUnlock}
            label={`Unblock ${entry.pattern}`}
            confirmLabel="Unblock"
            isPending={unblock.isPending}
            onConfirm={() =>
              unblock.mutate(entry.id, {
                // The banner describes a block that no longer stands.
                onSuccess: () => setLastBlocked(undefined),
              })
            }
          />
        </RowActionRow>
      ),
    },
  ]

  return (
    <section
      aria-labelledby="blocked-emails-heading"
      className="mt-10 flex flex-col gap-3"
    >
      <div className="flex flex-col gap-1">
        <h2 id="blocked-emails-heading" className="text-title">
          Blocked emails
        </h2>
        <p className="text-caption">
          A blocked address cannot sign up, be added to or invited into an
          organization, or sign in, and is sent no signup or reset mail. A
          domain also blocks its subdomains. Blocking ends the matching
          accounts' sessions; unblocking lets them sign in again. API keys are
          not affected.
        </p>
      </div>

      <ErrorBanner error={entries.error ?? block.error ?? unblock.error} />
      {lastBlocked ? (
        <InfoBanner>{blockedSummary(lastBlocked)}</InfoBanner>
      ) : null}

      <form
        className="flex flex-col gap-3 md:flex-row md:items-start"
        onSubmit={(event) => {
          event.preventDefault()
          submit()
        }}
      >
        <div className="md:w-80">
          <Field
            label="Address or domain"
            value={value}
            onChange={setValue}
            placeholder="spam.example or person@example.com"
          />
        </div>
        <div className="md:w-80">
          <Field
            label="Reason (optional)"
            value={reason}
            onChange={setReason}
            placeholder="Only operators see this"
          />
        </div>
        <Button
          type="submit"
          variant="primary"
          className="shrink-0 md:mt-6"
          isPending={block.isPending}
          isDisabled={value.trim() === ""}
        >
          Block
        </Button>
      </form>

      <TableScrollFrame className="otari-blocked-emails-table">
        <DataTable
          mobileLayout="cards"
          ariaLabel="Blocked emails"
          columns={columns}
          rows={rows}
          getRowKey={(entry) => entry.id}
          isLoading={entries.isLoading}
          emptyContent="Nothing is blocked."
        />
      </TableScrollFrame>
      <TablePagination
        page={page}
        pageSize={pageSize}
        total={entries.data?.count ?? null}
        rowsOnPage={rows.length}
        onPageChange={setPage}
        onPageSizeChange={(size) => {
          setPageSize(size)
          setPage(0)
        }}
        isFetching={entries.isFetching}
        label="entries"
      />
    </section>
  )
}
