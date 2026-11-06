import { useState } from "react"

import type {
  OidcConnection,
  OidcConnectionDomain,
  OidcConnectionRequest,
  OrganizationMember,
  OrganizationSsoEnforcement,
  OrganizationSsoPreflight,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { CopyField } from "@/design-system/actions/CopyField"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { Select } from "@/design-system/forms/Select"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { Chip } from "@/design-system/indicators/Chip"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { Section } from "@/design-system/layout/Section"
import {
  useDeleteOrganizationSso,
  useDeleteOrganizationSsoLink,
  useOrganizationContext,
  useOrganizationMembers,
  useOrganizationSso,
  useOrganizationSsoPreflight,
  useSaveOrganizationSso,
  useSetOrganizationSsoEnforcement,
  useTestOrganizationSso,
} from "@/shared/api/organizations"
import { formatDate, formatRelative } from "@/shared/helpers/format"

import { canManage, membershipLabel } from "./roles"

// An organization's own bring-your-own OIDC connection, for signing in through
// an identity provider the organization administers instead of Otari's own
// sign-in methods.
//
// The redirect URI is shown whether or not a connection exists yet, because an
// administrator needs it to register at their provider before saving one.
// Once a connection exists, admission runs through the same email-domain
// claims the Email domains page manages: this page shows them again, each with
// its proof's expiry, so an administrator reads here that Otari does not trust
// an identity provider's own claim about whether an address is verified. A
// domain's DNS proof is what admits someone, and nothing admits anyone once
// every domain's proof has lapsed.

/** The roles a sign-in through this connection may be given. Owner and admin are
    deliberately missing, matching the server's validation: a claim proven by DNS
    is not a decision about any one person, so it must never mint someone who can
    manage the organization. */
const DEFAULT_ROLE_OPTIONS = [
  { value: "member", label: "Member" },
  { value: "viewer", label: "Viewer" },
]

interface Draft {
  displayName: string
  issuer: string
  clientId: string
  clientSecret: string
  defaultRole: string
  enabled: boolean
}

function initialDraft(connection: OidcConnection | null): Draft {
  return {
    displayName: connection?.display_name ?? "",
    issuer: connection?.issuer ?? "",
    clientId: connection?.client_id ?? "",
    // Never prefilled: the server does not serve the secret back.
    clientSecret: "",
    defaultRole: connection?.default_role ?? "member",
    enabled: connection?.enabled ?? true,
  }
}

function ConnectionDialog({
  connection,
  isOpen,
  onClose,
}: {
  /** The connection being edited; absent to add one. */
  connection: OidcConnection | null
  isOpen: boolean
  onClose: () => void
}) {
  const isEdit = connection !== null
  const save = useSaveOrganizationSso()
  const [draft, setDraft] = useState<Draft>(() => initialDraft(connection))
  const { isDirty } = useDirtySnapshot(draft)
  const set = (patch: Partial<Draft>) => setDraft({ ...draft, ...patch })

  // The stored secret belongs to one client at one issuer, so it can only be kept while the
  // submission still names both. The server enforces this and refuses a save that cannot keep it.
  const needsSecret =
    !isEdit ||
    draft.issuer.trim() !== connection.issuer ||
    draft.clientId.trim() !== connection.client_id

  const isReady =
    draft.displayName.trim() !== "" &&
    draft.issuer.trim() !== "" &&
    draft.clientId.trim() !== "" &&
    (!needsSecret || draft.clientSecret !== "")

  const submit = () => {
    const body: OidcConnectionRequest = {
      display_name: draft.displayName.trim(),
      issuer: draft.issuer.trim(),
      client_id: draft.clientId.trim(),
      client_secret: draft.clientSecret === "" ? null : draft.clientSecret,
      default_role: draft.defaultRole === "viewer" ? "viewer" : "member",
      enabled: draft.enabled,
    }
    save.mutate(body, { onSuccess: onClose })
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      size="md"
      title={isEdit ? "Edit connection" : "Add connection"}
      submitLabel={isEdit ? "Save connection" : "Add connection"}
      onSubmit={submit}
      isPending={save.isPending}
      isSubmitDisabled={!isReady}
      isDirty={isDirty}
      error={save.error}
    >
      <Field
        label="Display name"
        value={draft.displayName}
        onChange={(next) => set({ displayName: next })}
        isRequired
        autoFocus
        placeholder="Acme Okta"
        description="Shown to people on the sign-in screen."
      />
      <Field
        label="Issuer"
        value={draft.issuer}
        onChange={(next) => set({ issuer: next })}
        isRequired
        placeholder="https://login.acme.example"
        description="Your identity provider's issuer URL, as it claims it."
      />
      <Field
        label="Client ID"
        value={draft.clientId}
        onChange={(next) => set({ clientId: next })}
        isRequired
      />
      <SecretField
        label="Client secret"
        value={draft.clientSecret}
        onChange={(clientSecret) => set({ clientSecret })}
        isRequired={needsSecret}
        description={
          needsSecret
            ? "Encrypted at rest and never shown again."
            : "Encrypted at rest and never shown again. Leave it blank to keep the stored one. Changing the issuer or the client ID needs a new secret, because the stored one belongs to the old pair."
        }
      />
      <Select
        label="New sign-ins join as"
        value={draft.defaultRole}
        onChange={(next) => set({ defaultRole: next })}
        options={DEFAULT_ROLE_OPTIONS}
        shouldReserveMessage={false}
      />
      <Checkbox
        isSelected={draft.enabled}
        onChange={(enabled) => set({ enabled })}
      >
        Allow sign-in through this connection
      </Checkbox>
    </FormDialog>
  )
}

/** One domain's admission state, in the words an administrator reads rather than a status code. */
function domainStatusLabel(
  domain: OidcConnectionDomain,
  now: number = Date.now(),
): string {
  if (domain.proof_expires_at === null) {
    return "has not verified its DNS proof yet"
  }
  const hasExpired = new Date(domain.proof_expires_at).getTime() <= now
  if (hasExpired) {
    return `proof expired ${formatDate(domain.proof_expires_at)}`
  }
  if (!domain.enabled) {
    return "claim is paused"
  }
  return `proof expires ${formatDate(domain.proof_expires_at)}`
}

function ConnectionDomains({
  domains,
}: {
  domains: readonly OidcConnectionDomain[]
}) {
  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-title">Domains that admit sign-in</h3>
      <p className="text-caption">
        Only someone whose address matches one of these domains is admitted
        through this connection. Once every domain&rsquo;s proof has expired,
        this connection stops admitting anyone until one is re-verified.
      </p>
      {domains.length === 0 ? (
        <p className="text-caption text-subtle">
          This organization has no email domains yet. Claim one on the Email
          domains page before anyone can sign in through this connection.
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {domains.map((domain) => (
            <li
              key={domain.domain}
              className="flex flex-wrap items-center justify-between gap-2 border-b border-border-subtle pb-2 last:border-b-0 last:pb-0"
            >
              <span className="font-medium">{domain.domain}</span>
              <span className="text-caption">{domainStatusLabel(domain)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function ConnectionSummary({
  connection,
  onEdit,
  onRemove,
}: {
  connection: OidcConnection
  onEdit: () => void
  onRemove: () => void
}) {
  const test = useTestOrganizationSso()
  return (
    <Section
      className="border-y border-border py-5"
      contentClassName="flex flex-col gap-4"
    >
      <div className="flex items-center gap-2">
        <h2 className="text-title">{connection.display_name}</h2>
        <Chip tone={connection.enabled ? "success" : "neutral"}>
          {connection.enabled ? "Enabled" : "Disabled"}
        </Chip>
      </div>
      <dl className="grid items-baseline gap-x-6 gap-y-3 sm:grid-cols-[10rem_1fr]">
        <dt className="text-muted">Issuer</dt>
        <dd className="break-all text-emphasis">{connection.issuer}</dd>
        <dt className="text-muted">Client ID</dt>
        <dd className="break-all text-emphasis">{connection.client_id}</dd>
        <dt className="text-muted">Client secret</dt>
        <dd>
          <code className="text-muted">
            {`••••${connection.client_secret_last4}`}
          </code>
        </dd>
        <dt className="text-muted">New sign-ins join as</dt>
        <dd className="text-emphasis">
          {membershipLabel(connection.default_role)}
        </dd>
        <dt className="text-muted">Last checked</dt>
        <dd className="text-emphasis">
          {formatRelative(connection.discovered_at)}
        </dd>
      </dl>
      <ErrorBanner error={test.error} />
      <p className="text-muted text-sm">
        Refreshing re-reads the endpoints and signing keys your provider
        publishes. It does not check the client ID or secret, which are only
        used when somebody signs in.
      </p>
      <div className="flex flex-wrap items-center justify-end gap-2 border-t border-border pt-4">
        <Button
          variant="ghost"
          isPending={test.isPending}
          onPress={() => test.mutate()}
        >
          Refresh from provider
        </Button>
        <Button variant="ghost" onPress={onRemove}>
          Remove connection
        </Button>
        <Button variant="primary" onPress={onEdit}>
          Edit connection
        </Button>
      </div>
      <ConnectionDomains domains={connection.domains} />
    </Section>
  )
}

function displayName(member: {
  full_name?: string | null
  email?: string | null
}): string {
  return member.full_name ?? member.email ?? "Unnamed member"
}

/** The one wording of what an identity change strands on the old account. */
function strandedKeysExplanation(): string {
  return "Their API keys stay on the old account and cannot be moved, so they should issue new keys on the new account and revoke the old ones."
}

function preflightSummary(preflight: OrganizationSsoPreflight): string {
  if (preflight.affected_count === 0) {
    return `All ${preflight.total_members} active members already sign in through this connection.`
  }
  return (
    `${preflight.total_members} members, ${preflight.affected_count} of whom sign in some other way. ` +
    `Those ${preflight.affected_count} will get a new account the first time they use your provider. ` +
    strandedKeysExplanation()
  )
}

function PreflightPanel() {
  const preflight = useOrganizationSsoPreflight()
  const data = preflight.data
  return (
    <div className="flex flex-col gap-2">
      <ErrorBanner error={preflight.error} />
      {data ? (
        <>
          <p className="text-muted">{preflightSummary(data)}</p>
          {data.affected_count > 0 ? (
            <ul className="flex flex-col gap-1">
              {data.affected_members.map((member) => (
                <li key={member.user_id} className="text-caption">
                  {displayName(member)}
                </li>
              ))}
              {data.affected_members_truncated ? (
                <li className="text-caption text-subtle">
                  and {data.affected_count - data.affected_members.length}{" "}
                  others
                </li>
              ) : null}
            </ul>
          ) : null}
        </>
      ) : null}
    </div>
  )
}

function EnforcementSection({
  enforcement,
  owners,
  onSaved,
}: {
  enforcement: OrganizationSsoEnforcement
  owners: readonly OrganizationMember[]
  onSaved: () => void
}) {
  const set = useSetOrganizationSsoEnforcement()
  const [required, setRequired] = useState(enforcement.required)
  const [breakGlassUserId, setBreakGlassUserId] = useState(
    enforcement.break_glass_user_id ?? "",
  )
  const [breakGlassWaived, setBreakGlassWaived] = useState(
    enforcement.break_glass_waived,
  )

  const chosenOwner = owners.find((owner) => owner.user_id === breakGlassUserId)

  const submit = () => {
    set.mutate(
      {
        required,
        break_glass_user_id: breakGlassWaived ? null : breakGlassUserId || null,
        break_glass_waived: breakGlassWaived,
      },
      { onSuccess: onSaved },
    )
  }

  return (
    <Section
      className="border-b border-border py-5"
      contentClassName="flex flex-col gap-4"
    >
      <div className="flex flex-col gap-1">
        <h2 className="text-title">Require sign-in</h2>
        <p className="text-caption">
          Every other way to act in this organization, including a live session,
          is refused once this is on.
        </p>
      </div>
      <PreflightPanel />
      <Checkbox isSelected={required} onChange={setRequired}>
        Require sign-in through this connection
      </Checkbox>
      {required ? (
        <div className="flex flex-col gap-3">
          <Select
            label="Owner who keeps a non-SSO route"
            value={breakGlassWaived ? "" : breakGlassUserId}
            onChange={(value) => {
              setBreakGlassUserId(value)
              setBreakGlassWaived(false)
            }}
            options={owners.map((owner) => ({
              value: owner.user_id ?? "",
              label: displayName(owner),
            }))}
            placeholder="Choose an owner"
            isDisabled={breakGlassWaived}
            shouldReserveMessage={false}
          />
          {chosenOwner ? (
            <p className="text-caption">
              {displayName(chosenOwner)} keeps a non-SSO route and may still
              sign in without your provider.
            </p>
          ) : null}
          <Checkbox
            isSelected={breakGlassWaived}
            onChange={(waived) => {
              setBreakGlassWaived(waived)
              if (waived) setBreakGlassUserId("")
            }}
          >
            This organization keeps no non-SSO route
          </Checkbox>
          {breakGlassWaived ? (
            <p className="text-caption">
              If sign-in through your provider ever fails, nobody in this
              organization can sign in until a deployment operator steps in.
            </p>
          ) : null}
        </div>
      ) : null}
      <ErrorBanner error={set.error} />
      <div className="flex justify-end border-t border-border pt-4">
        <Button variant="primary" onPress={submit} isPending={set.isPending}>
          Save enforcement
        </Button>
      </div>
    </Section>
  )
}

/** What removing ``name``'s link costs them, in the three facts an administrator needs. */
function unlinkWarning(name: string): string {
  return (
    `The next sign-in through this connection for ${name} creates a new account. ` +
    "The old account keeps its budgets and history. " +
    `${strandedKeysExplanation()} ` +
    'This does not revoke access. "Remove member" is the control for that.'
  )
}

function UnlinkSection({
  members,
}: {
  members: readonly OrganizationMember[]
}) {
  const unlink = useDeleteOrganizationSsoLink()
  const [selectedUserId, setSelectedUserId] = useState("")
  const [isConfirming, setIsConfirming] = useState(false)
  const candidate = members.find((member) => member.user_id === selectedUserId)

  return (
    <Section
      className="border-b border-border py-5"
      contentClassName="flex flex-col gap-4"
    >
      <div className="flex flex-col gap-1">
        <h2 className="text-title">Remove a sign-in link</h2>
        <p className="text-caption">
          Lets that member link afresh at their next sign-in. It removes a link
          only. It does not delete an account or end a membership.
        </p>
      </div>
      <div className="flex flex-wrap items-end gap-3">
        <Select
          label="Member"
          value={selectedUserId}
          onChange={setSelectedUserId}
          options={members
            .filter((member) => Boolean(member.user_id))
            .map((member) => ({
              value: member.user_id as string,
              label: displayName(member),
            }))}
          placeholder="Choose a member"
          shouldReserveMessage={false}
        />
        <Button
          variant="ghost"
          className="shrink-0"
          isDisabled={selectedUserId === ""}
          onPress={() => {
            unlink.reset()
            setIsConfirming(true)
          }}
        >
          Remove link
        </Button>
      </div>
      <ConfirmDialog
        isOpen={isConfirming}
        onOpenChange={setIsConfirming}
        heading="Remove this sign-in link"
        body={candidate ? unlinkWarning(displayName(candidate)) : null}
        confirmLabel="Remove link"
        isPending={unlink.isPending}
        error={unlink.error}
        onConfirm={() =>
          unlink.mutate(selectedUserId, {
            onSuccess: () => {
              setIsConfirming(false)
              setSelectedUserId("")
            },
          })
        }
      />
    </Section>
  )
}

export function OrganizationSsoPage() {
  const context = useOrganizationContext()
  const canEdit = canManage(context.data)
  const sso = useOrganizationSso(canEdit)
  const remove = useDeleteOrganizationSso()
  const [isEditing, setIsEditing] = useState(false)
  const [formOpenCount, setFormOpenCount] = useState(0)
  const [isRemoving, setIsRemoving] = useState(false)
  const [enforcementSavedCount, setEnforcementSavedCount] = useState(0)

  const connection = sso.data?.connection ?? null
  const members = useOrganizationMembers(canEdit && connection !== null)
  const activeMembers = members.data?.filter((m) => m.status === "active") ?? []
  const activeOwners = activeMembers.filter((m) => m.role === "owner")

  const openForm = () => {
    setFormOpenCount((count) => count + 1)
    setIsEditing(true)
  }

  return (
    <div className="flex flex-col">
      <PageIntro title="Single sign-on">
        Let this organization&rsquo;s own identity provider handle sign-in.
        Otari does not trust an identity provider&rsquo;s own{" "}
        <code>email_verified</code> claim: a domain&rsquo;s DNS proof, the same
        one the Email domains page verifies, is what admits someone, never what
        the provider asserts about their address.
      </PageIntro>

      <ErrorBanner error={context.error ?? sso.error} />

      {context.data && !canEdit ? (
        <InfoBanner>
          Only organization owners and admins can manage single sign-on.
        </InfoBanner>
      ) : null}

      {canEdit && sso.data ? (
        <>
          <Section
            className="border-y border-border py-5"
            contentClassName="flex flex-col gap-3"
          >
            <div className="flex flex-col gap-1">
              <h2 className="text-title">Provider setup</h2>
              <p className="text-caption">
                Register this redirect URI with your identity provider before
                saving a connection below. It does not change afterward.
              </p>
            </div>
            <CopyField label="Redirect URI" value={sso.data.redirect_uri} />
          </Section>

          {connection ? (
            <ConnectionSummary
              connection={connection}
              onEdit={openForm}
              onRemove={() => {
                remove.reset()
                setIsRemoving(true)
              }}
            />
          ) : (
            <Section
              className="border-b border-border py-5"
              contentClassName="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"
            >
              <div className="flex min-w-0 flex-col gap-1">
                <h2 className="text-title">No connection yet</h2>
                <p className="text-muted">
                  Add one once the redirect URI above is registered at your
                  provider.
                </p>
              </div>
              <Button variant="primary" className="shrink-0" onPress={openForm}>
                Add connection
              </Button>
            </Section>
          )}

          {connection && sso.data ? (
            <>
              <EnforcementSection
                key={enforcementSavedCount}
                enforcement={sso.data.enforcement}
                owners={activeOwners}
                onSaved={() => setEnforcementSavedCount((count) => count + 1)}
              />
              <UnlinkSection members={activeMembers} />
            </>
          ) : null}
        </>
      ) : null}

      <ConnectionDialog
        key={formOpenCount}
        connection={connection}
        isOpen={isEditing}
        onClose={() => setIsEditing(false)}
      />

      <ConfirmDialog
        isOpen={isRemoving}
        onOpenChange={setIsRemoving}
        heading="Remove connection"
        body={
          connection
            ? `${connection.display_name} is removed. Anyone signing in through it is refused from then on; setting it up again means entering its client secret once more. Members who already joined keep their membership.`
            : null
        }
        confirmLabel="Remove connection"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() =>
          remove.mutate(undefined, { onSuccess: () => setIsRemoving(false) })
        }
      />
    </div>
  )
}
