import { Button, Card } from "@heroui/react"
import { useState } from "react"
import {
  useCreateOrganization,
  useDeleteOrganization,
  useOrganizationContext,
  useOrganizationMemberships,
  useSwitchOrganization,
  useUpdateOrganization,
} from "@/shared/api/hooks"
import { Field } from "@/shared/components/Field"
import {
  CopyableValue,
  ErrorBanner,
  FilterSelect,
  InfoBanner,
  PageHeader,
  PageLoading,
} from "@/shared/components/ui"

import { canManage, isOwner } from "./roles"

// The organization the caller is pointed at, and the three things an operator
// does to it: rename it, move to another one, or retire it. The roster is its
// own page (Members), which is how otari.ai splits the same surface.

function OrganizationDetails({
  name,
  slug,
  canEdit,
}: {
  name: string
  slug: string
  canEdit: boolean
}) {
  const update = useUpdateOrganization()
  const [draft, setDraft] = useState(name)
  const trimmed = draft.trim()
  const isUnchanged = trimmed === name
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-foreground">Details</div>
        <ErrorBanner error={update.error} />
        <Field
          label="Organization name"
          value={draft}
          onChange={setDraft}
          isRequired
          description="What this tenant is called across the dashboard. The slug below is set when the organization is created and does not follow a rename."
        />
        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium text-foreground">Slug</span>
          <CopyableValue value={slug} label="organization slug">
            <code className="text-xs text-muted">{slug}</code>
          </CopyableValue>
        </div>
        <div>
          <Button
            variant="primary"
            isDisabled={!canEdit || isUnchanged || trimmed === ""}
            isPending={update.isPending}
            onPress={() => update.mutate({ name: trimmed })}
          >
            Save name
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

function SwitchOrganization({
  activeId,
  options,
}: {
  activeId: string
  options: { value: string; label: string }[]
}) {
  const switchTo = useSwitchOrganization()
  const [selected, setSelected] = useState(activeId)
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-foreground">
          Active organization
        </div>
        <p className="text-sm text-muted">
          Every page in this section reads the organization your identity is
          pointed at. Switching moves that pointer; it does not change what any
          organization contains.
        </p>
        <ErrorBanner error={switchTo.error} />
        <FilterSelect
          label="Organization"
          value={selected}
          onChange={setSelected}
          options={options}
        />
        <div>
          <Button
            variant="primary"
            isDisabled={selected === activeId}
            isPending={switchTo.isPending}
            onPress={() => switchTo.mutate({ organization_id: selected })}
          >
            Switch
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

function CreateOrganizationForm({ onClose }: { onClose: () => void }) {
  const create = useCreateOrganization()
  const [name, setName] = useState("")
  const trimmed = name.trim()
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-foreground">
          Create organization
        </div>
        <ErrorBanner error={create.error} />
        <Field
          label="Name"
          value={name}
          onChange={setName}
          placeholder="Platform team"
          isRequired
          autoFocus
          description="You become its owner, it gets a default workspace, and you are switched into it."
        />
        <div className="flex gap-2">
          <Button
            variant="primary"
            isDisabled={trimmed === ""}
            isPending={create.isPending}
            onPress={() =>
              create.mutate({ name: trimmed }, { onSuccess: onClose })
            }
          >
            Create organization
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

function DeleteOrganization({
  name,
  isOnlyOrganization,
}: {
  name: string
  isOnlyOrganization: boolean
}) {
  const remove = useDeleteOrganization()
  const [armed, setArmed] = useState(false)
  const [confirmation, setConfirmation] = useState("")
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-danger">Danger zone</div>
        <p className="text-sm text-muted">
          Deleting <strong>{name}</strong> removes its workspaces and
          memberships. Usage the gateway already recorded is not deleted with
          it.
        </p>
        {isOnlyOrganization ? (
          <InfoBanner tone="warning">
            This is the only organization your identity belongs to, and every
            identity has to be pointed at one, so it cannot be deleted. Create
            another organization first.
          </InfoBanner>
        ) : null}
        <ErrorBanner error={remove.error} />
        {armed ? (
          <>
            <Field
              label="Type the organization name to confirm"
              value={confirmation}
              onChange={setConfirmation}
              placeholder={name}
              autoFocus
            />
            <div className="flex gap-2">
              <Button
                variant="danger"
                isDisabled={confirmation !== name || isOnlyOrganization}
                isPending={remove.isPending}
                onPress={() => remove.mutate()}
              >
                Delete permanently
              </Button>
              <Button variant="ghost" onPress={() => setArmed(false)}>
                Cancel
              </Button>
            </div>
          </>
        ) : (
          <div>
            <Button
              variant="danger-soft"
              isDisabled={isOnlyOrganization}
              onPress={() => {
                setConfirmation("")
                setArmed(true)
              }}
            >
              Delete organization
            </Button>
          </div>
        )}
      </Card.Content>
    </Card>
  )
}

export function OrganizationGeneralPage() {
  const context = useOrganizationContext()
  const memberships = useOrganizationMemberships()
  const [creating, setCreating] = useState(false)

  if (context.isLoading) {
    return <PageLoading label="Loading organization…" />
  }
  if (context.error || !context.data) {
    return (
      <div className="flex flex-col gap-6">
        <PageHeader title="Organization" />
        <ErrorBanner error={context.error ?? new Error("No organization.")} />
      </div>
    )
  }

  const { organization, role } = context.data
  const canEdit = canManage(context.data)
  const options = (memberships.data ?? []).map((membership) => ({
    value: membership.organization.id,
    label: membership.organization.name,
  }))

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Organization"
        description={`The tenant this gateway's workspaces, members, and roles belong to. Your role here is ${role}.`}
        action={
          creating ? null : (
            <Button variant="primary" onPress={() => setCreating(true)}>
              Create organization
            </Button>
          )
        }
      />

      {creating ? (
        <CreateOrganizationForm onClose={() => setCreating(false)} />
      ) : null}

      {canEdit ? null : (
        <InfoBanner>
          You are a {role} in {organization.name}. Only owners and admins can
          change it.
        </InfoBanner>
      )}

      {/* Keyed so switching organizations reseeds the name draft; the prefix
          keeps it from colliding with the switcher's key below, which is the
          same id under the same parent. */}
      <OrganizationDetails
        key={`details-${organization.id}`}
        name={organization.name}
        slug={organization.slug}
        canEdit={canEdit}
      />

      {options.length > 1 ? (
        // Keyed on the active organization so switching reseeds the picker
        // rather than leaving it on the value that was just applied.
        <SwitchOrganization
          key={`switch-${organization.id}`}
          activeId={organization.id}
          options={options}
        />
      ) : null}

      {/* Held back until the membership list has actually answered: the delete
          control's whole story is whether there is another organization to be
          moved to, and a list that has not loaded yet looks exactly like a
          caller who belongs to one organization. */}
      {isOwner(context.data) && memberships.isSuccess ? (
        <DeleteOrganization
          name={organization.name}
          isOnlyOrganization={options.length <= 1}
        />
      ) : null}
    </div>
  )
}
