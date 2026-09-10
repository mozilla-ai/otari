import { Button, Chip } from "@heroui/react"
import { useRef, useState } from "react"

import type {
  CreateOrganizationDomainRequest,
  OrganizationDomain,
} from "@/client"
import { CopyField } from "@/design-system/actions/CopyField"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { Section } from "@/design-system/layout/Section"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import {
  useCreateOrganizationDomain,
  useDeleteOrganizationDomain,
  useOrganizationContext,
  useOrganizationDomains,
  useUpdateOrganizationDomain,
  useVerifyOrganizationDomain,
} from "@/shared/api/organizations"
import { formatRelative } from "@/shared/helpers/format"

import { canManage, membershipLabel } from "./roles"

// Email-domain auto-join: anyone who signs in with a verified address at a
// domain this organization has *proven* it controls becomes a member, at the
// role the claim names.
//
// The page is built around one asymmetry. Claiming a domain is free and means
// nothing, and it is the DNS record that makes a claim act on anybody, so a
// pending claim is shown as inert rather than as a step someone forgot: the row
// says what it will do once verified, and the record to publish is the most
// prominent thing on it. Without that framing an unverified claim reads as
// working, and the one thing worse than a claim that does nothing is believing
// a claim is guarding a domain when it is not.
//
// Owner and admin are deliberately missing from the role picker, matching
// `ORGANIZATION_DOMAIN_ROLES` on the server: publishing a DNS record proves
// control of a domain, which is not a decision about any one person, so it must
// never be enough to mint someone who can manage the organization.
//
// A proof expires, so "verified" is not a terminal state and the page has three
// of them rather than two: never proven, proven, and proven-but-stale. The last
// gets the same card as the first, because the admin's next action is the same.

/**
 * Whether a claim's DNS proof has aged out.
 *
 * `proof_expires_at` is computed by the server from its own TTL, so this reads
 * the answer rather than holding a second copy of the constant that could drift
 * from it.
 */
function proofExpired(row: OrganizationDomain): boolean {
  const expiresAt = row.proof_expires_at
  // Absent as well as null: the field is optional on the wire, and a claim with
  // no expiry is one with no proof, which the caller handles as "not verified".
  return expiresAt != null && new Date(expiresAt).getTime() <= Date.now()
}

/** The roles a claim may hand out. Narrower than `MEMBERSHIP_ROLES` on purpose. */
const AUTO_JOIN_ROLE_OPTIONS = [
  { value: "member", label: "Member" },
  { value: "viewer", label: "Viewer" },
]

function ClaimForm({
  isOpen,
  onClose,
}: {
  isOpen: boolean
  onClose: () => void
}) {
  const create = useCreateOrganizationDomain()
  const [domain, setDomain] = useState("")
  const [role, setRole] = useState("member")
  // One snapshot of everything the form owns, seeded on mount: dirty means
  // "differs from what was seeded", and a field added to the form is added here
  // or the guard cannot see it. A predicate of the fields had already forgotten
  // `role`, so changing who joins and pressing Escape discarded it unguarded.
  const draft = JSON.stringify({ domain, role })
  const seededDraft = useRef(draft)

  const submit = () => {
    const body: CreateOrganizationDomainRequest = {
      domain: domain.trim(),
      default_role: role === "viewer" ? "viewer" : "member",
      enabled: true,
    }
    create.mutate(body, { onSuccess: onClose })
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      size="sm"
      title="New domain"
      submitLabel="Claim domain"
      onSubmit={submit}
      isPending={create.isPending}
      isSubmitDisabled={domain.trim() === ""}
      isDirty={draft !== seededDraft.current}
      error={create.error}
    >
      <Field
        label="Domain"
        value={domain}
        onChange={setDomain}
        isRequired
        autoFocus
        placeholder="example.com"
        description="The domain your colleagues' addresses end in. A whole address works too; only its domain is stored. Public providers like gmail.com can't be claimed."
      />
      <Select
        label="They join as"
        value={role}
        onChange={setRole}
        options={AUTO_JOIN_ROLE_OPTIONS}
        reserveMessage={false}
      />
      <p className="text-caption">
        Nothing happens until you publish the DNS record this creates and verify
        it. Anyone who already has an account joins on their next sign-in.
      </p>
    </FormDialog>
  )
}

/** The record to publish, shown while a claim has no proof it can act on. */
function PendingProof({ row }: { row: OrganizationDomain }) {
  const verify = useVerifyOrganizationDomain()
  const expired = proofExpired(row)
  return (
    <Section
      className="border-y border-border py-5"
      contentClassName="flex flex-col gap-4"
    >
      <div className="flex flex-col gap-1">
        <h2 className="text-title">
          {expired ? `Re-verify ${row.domain}` : `Verify ${row.domain}`}
        </h2>
        <p className="text-caption">
          {expired ? (
            <>
              This domain's proof has expired, so the claim has stopped
              admitting anyone. Domains change hands, so a proof is good for a
              limited time and is renewed by checking the record again. The
              record has not changed: it should still be published at the apex
              of <code>{row.domain}</code>.
            </>
          ) : (
            <>
              Publish this as a TXT record at the apex of{" "}
              <code>{row.domain}</code>, then verify. Until then the claim
              admits nobody. DNS changes can take a while to propagate, so a
              first attempt that fails is normal.
            </>
          )}
        </p>
      </div>
      <ErrorBanner error={verify.error} />
      {/* The verify action sits in the field's row rather than under it: the
          record is one line an operator copies and then acts on, so the copy
          affordance moved inside the field and the button it hands off to is
          beside it. */}
      <CopyField
        label={`TXT record for ${row.domain}`}
        value={row.verification_record}
        action={
          <Button
            variant="primary"
            isPending={verify.isPending}
            onPress={() => verify.mutate(row.id)}
          >
            {expired ? "Re-verify domain" : "Verify domain"}
          </Button>
        }
      />
    </Section>
  )
}

export function OrganizationDomainsPage() {
  const context = useOrganizationContext()
  const canEdit = canManage(context.data)
  const domains = useOrganizationDomains(canEdit)
  const update = useUpdateOrganizationDomain()
  const remove = useDeleteOrganizationDomain()
  const [adding, setAdding] = useState(false)
  const [openCount, setOpenCount] = useState(0)
  const [pendingDelete, setPendingDelete] = useState<OrganizationDomain>()

  const rows = domains.data?.data ?? []
  // Both states need the same card: one has never had a proof, the other's has
  // aged out, and in each case the claim is admitting nobody until it verifies.
  const pending = rows.filter(
    (row) => row.verified_at === null || proofExpired(row),
  )

  const columns: DataTableColumn<OrganizationDomain>[] = [
    {
      id: "domain",
      header: "Domain",
      isRowHeader: true,
      cell: (row) => <span className="font-medium">{row.domain}</span>,
    },
    {
      id: "status",
      header: "Status",
      cell: (row) =>
        row.verified_at === null ? (
          <Chip size="sm" color="warning">
            Not verified
          </Chip>
        ) : proofExpired(row) ? (
          // Distinct from "Not verified": this domain *was* proven, and the
          // claim is one re-check from working rather than never having run.
          <Chip size="sm" color="warning">
            Proof expired
          </Chip>
        ) : row.enabled ? (
          <Chip size="sm" color="accent">
            Active
          </Chip>
        ) : (
          // Verified but switched off: the proof still stands, so this is one
          // toggle away from working and is not the same as "not verified".
          <Chip size="sm" color="default">
            Paused
          </Chip>
        ),
    },
    {
      id: "role",
      header: "Joins as",
      cell: (row) => membershipLabel(row.default_role),
    },
    {
      id: "added",
      header: "Added",
      cell: (row) => formatRelative(row.created_at),
    },
  ]

  if (canEdit) {
    columns.push({
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (row) => (
        <div className="flex items-center justify-end gap-1.5">
          {/* Only offered once the claim is proven: pausing an unverified
              claim would suggest it was otherwise admitting people. */}
          {row.verified_at !== null ? (
            <Button
              size="sm"
              variant="ghost"
              isDisabled={update.isPending}
              onPress={() =>
                update.mutate({
                  domainId: row.id,
                  body: { enabled: !row.enabled },
                })
              }
            >
              {row.enabled ? "Pause" : "Resume"}
            </Button>
          ) : null}
          <Button
            size="sm"
            variant="ghost"
            onPress={() => setPendingDelete(row)}
          >
            Remove domain
          </Button>
        </div>
      ),
    })
  }

  return (
    <div className="flex flex-col">
      <PageIntro
        title="Email domains"
        action={
          canEdit ? (
            <Button
              // Visible while the dialog is open: the dialog is over the page.
              variant="primary"
              onPress={() => {
                setOpenCount((count) => count + 1)
                setAdding(true)
              }}
            >
              Claim domain
            </Button>
          ) : null
        }
      >
        Let colleagues join this organization automatically. Anyone who signs in
        with a verified address at a domain you have proven you control becomes
        a member, at the role you choose. A claim does nothing until its DNS
        record is verified.
      </PageIntro>

      <ErrorBanner error={context.error ?? domains.error ?? update.error} />

      {/* Held back until the context has answered, so an admin is not told for
          one paint that they may not be here. */}
      {context.data && !canEdit ? (
        <InfoBanner>
          Only organization owners and admins can manage email domains.
        </InfoBanner>
      ) : null}

      {/* Keyed on the open count, so each open remounts a blank form. Clearing
          the draft on close instead would blank the fields while the dialog is
          still animating away. */}
      <ClaimForm
        key={openCount}
        isOpen={adding}
        onClose={() => setAdding(false)}
      />

      {pending.map((row) => (
        <PendingProof key={row.id} row={row} />
      ))}

      {canEdit || context.isPending ? (
        <TableScrollFrame className="otari-domains-table">
          <DataTable
            ariaLabel="Organization email domains"
            columns={columns}
            rows={rows}
            getRowKey={(row) => row.id}
            isLoading={context.isPending || domains.isLoading}
            emptyContent="No email domains yet. Claim one so colleagues join automatically instead of being added by hand."
          />
        </TableScrollFrame>
      ) : null}

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        // Cleared on the way out: a refusal otherwise sits on the mutation and
        // greets the next row's confirm as if that row had failed.
        onOpenChange={(open) => {
          if (open) return
          setPendingDelete(undefined)
          remove.reset()
        }}
        heading="Remove email domain"
        body={
          pendingDelete
            ? `${pendingDelete.domain} stops admitting anyone to this organization. Members who already joined through it keep their membership, and claiming it again means proving the DNS record over.`
            : null
        }
        confirmLabel="Remove claim"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          if (!pendingDelete) return
          remove.mutate(pendingDelete.id, {
            onSuccess: () => setPendingDelete(undefined),
          })
        }}
      />
    </div>
  )
}
