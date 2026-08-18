import { Button, Card } from "@heroui/react"
import { useState } from "react"

import {
  useOrganizationContext,
  useUpdateOrganization,
} from "@/shared/api/hooks"
import { Field } from "@/shared/components/Field"
import {
  CopyableValue,
  ErrorBanner,
  InfoBanner,
  PageHeader,
  PageLoading,
} from "@/shared/components/ui"

import { canManage } from "./roles"

// The organization this deployment is, and the one thing an operator does to
// it: rename it.
//
// A self-hosted gateway is one tenant with several people in it, not several
// tenants, so the organization is provisioned at first boot and fixed: the
// gateway mounts no endpoint to create, switch between, or delete one, and this
// page offers none. Adding those surfaces is additive and belongs behind the
// entitlement gate an overlay contributes, not here. The roster is its own page
// (Members), which is how otari.ai splits the same surface.

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
          description="What this deployment's tenant is called across the dashboard. The slug below is set when the organization is provisioned and does not follow a rename."
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

export function OrganizationGeneralPage() {
  const context = useOrganizationContext()

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

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Organization"
        description={`The tenant this gateway's workspaces, members, and roles belong to. There is one per deployment, provisioned on first boot. Your role here is ${role}.`}
      />

      {canEdit ? null : (
        <InfoBanner>
          You are a {role} in {organization.name}. Only owners and admins can
          change it.
        </InfoBanner>
      )}

      {/* Keyed on the organization so a change of tenant reseeds the name
          draft, which is seeded on mount only. */}
      <OrganizationDetails
        key={organization.id}
        name={organization.name}
        slug={organization.slug}
        canEdit={canEdit}
      />
    </div>
  )
}
