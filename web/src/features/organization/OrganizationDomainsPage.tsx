import { Button, Card, Chip } from "@heroui/react"
import { useState } from "react"

import type {
  CreateOrganizationDomainRequest,
  OrganizationDomain,
} from "@/client"
import {
  useCreateOrganizationDomain,
  useDeleteOrganizationDomain,
  useOrganizationContext,
  useOrganizationDomains,
  useUpdateOrganizationDomain,
  useVerifyOrganizationDomain,
} from "@/shared/api/hooks"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import { Field } from "@/shared/components/Field"
import {
  ConfirmButton,
  CopyField,
  ErrorBanner,
  FilterSelect,
  InfoBanner,
  PageHeader,
} from "@/shared/components/ui"
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

function ClaimForm({ onClose }: { onClose: () => void }) {
  const create = useCreateOrganizationDomain()
  const [domain, setDomain] = useState("")
  const [role, setRole] = useState("member")

  const submit = () => {
    const body: CreateOrganizationDomainRequest = {
      domain: domain.trim(),
      default_role: role === "viewer" ? "viewer" : "member",
      enabled: true,
    }
    create.mutate(body, {
      onSuccess: () => {
        setDomain("")
        onClose()
      },
    })
  }

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <h2 className="text-title">Claim an email domain</h2>
        <ErrorBanner error={create.error} />
        <Field
          label="Domain"
          value={domain}
          onChange={setDomain}
          isRequired
          autoFocus
          placeholder="example.com"
          description="The domain your colleagues' addresses end in. A whole address works too; only its domain is stored. Public providers like gmail.com can't be claimed."
        />
        <FilterSelect
          label="They join as"
          value={role}
          onChange={setRole}
          options={AUTO_JOIN_ROLE_OPTIONS}
        />
        <p className="text-caption">
          Nothing happens until you publish the DNS record this creates and
          verify it. Anyone who already has an account joins on their next
          sign-in.
        </p>
        <div className="flex gap-2">
          <Button
            variant="primary"
            isDisabled={domain.trim() === ""}
            isPending={create.isPending}
            onPress={submit}
          >
            Claim domain
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

/** The record to publish, shown while a claim has no proof it can act on. */
function PendingProof({ row }: { row: OrganizationDomain }) {
  const verify = useVerifyOrganizationDomain()
  const expired = proofExpired(row)
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
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
        <CopyField
          label={`TXT record for ${row.domain}`}
          value={row.verification_record}
        />
        <div>
          <Button
            variant="primary"
            isPending={verify.isPending}
            onPress={() => verify.mutate(row.id)}
          >
            {expired ? "Re-verify domain" : "Verify domain"}
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

export function OrganizationDomainsPage() {
  const context = useOrganizationContext()
  const canEdit = canManage(context.data)
  const domains = useOrganizationDomains(canEdit)
  const update = useUpdateOrganizationDomain()
  const remove = useDeleteOrganizationDomain()
  const [adding, setAdding] = useState(false)

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
              variant="outline"
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
          <ConfirmButton
            confirmLabel="Remove"
            isPending={remove.isPending}
            onConfirm={() => remove.mutate(row.id)}
          >
            Remove
          </ConfirmButton>
        </div>
      ),
    })
  }

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Email domains"
        description="Let colleagues join this organization automatically. Anyone who signs in with a verified address at a domain you have proven you control becomes a member, at the role you choose. A claim does nothing until its DNS record is verified."
        action={
          canEdit && !adding ? (
            <Button variant="primary" onPress={() => setAdding(true)}>
              Claim domain
            </Button>
          ) : null
        }
      />

      <ErrorBanner
        error={context.error ?? domains.error ?? update.error ?? remove.error}
      />

      {/* Held back until the context has answered, so an admin is not told for
          one paint that they may not be here. */}
      {context.data && !canEdit ? (
        <InfoBanner>
          Only organization owners and admins can manage email domains.
        </InfoBanner>
      ) : null}

      {adding ? <ClaimForm onClose={() => setAdding(false)} /> : null}

      {pending.map((row) => (
        <PendingProof key={row.id} row={row} />
      ))}

      {canEdit || context.isPending ? (
        <DataTable
          ariaLabel="Organization email domains"
          columns={columns}
          rows={rows}
          getRowKey={(row) => row.id}
          isLoading={context.isPending || domains.isLoading}
          emptyContent="No email domains yet. Claim one so colleagues join automatically instead of being added by hand."
        />
      ) : null}
    </div>
  )
}
