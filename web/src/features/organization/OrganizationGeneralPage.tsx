import { Button } from "@heroui/react"
import { useState } from "react"

import {
  useOrganizationContext,
  useUpdateOrganization,
} from "@/shared/api/hooks"
import { Field } from "@/shared/components/Field"
import { PageIntro, Section } from "@/shared/components/surface"
import {
  CopyableValue,
  ErrorBanner,
  InfoBanner,
  PageLoading,
} from "@/shared/components/ui"

import { canManage } from "./roles"

// The organization this deployment is, and the one thing an operator does to
// it: rename it.
//
// Creating an organization and moving between them live in the scope switcher
// above the rail, not here: they are about which organization you are looking
// at, where this page is about the one you are in. Deleting one is nowhere,
// because the gateway mounts no endpoint for it (every historical attribution
// resolves through rows hanging off an organization). The roster is its own page
// (Members), which is how otari.ai splits the same surface.

/**
 * The form band: a rule above it, a rule below it, and a third rule dividing
 * the fields from the row that commits them. No card, because a page with one
 * form on it does not need a box to say where the form is; the rules already
 * do, and the box was the only thing making this page look like a different
 * page from Keys, which has the same shape.
 */
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
    <Section
      aria-labelledby="organization-details-title"
      className="border-y border-border py-5"
      contentClassName="flex flex-col gap-4"
    >
      <h2 id="organization-details-title" className="text-title">
        Details
      </h2>
      <ErrorBanner error={update.error} />
      <Field
        label="Organization name"
        value={draft}
        onChange={setDraft}
        isRequired
        isDisabled={!canEdit}
        description="What this deployment's tenant is called across the dashboard. The slug below is set when the organization is provisioned and does not follow a rename."
      />
      <div className="flex flex-col gap-1">
        <span className="text-sm font-medium text-foreground">Slug</span>
        <CopyableValue value={slug} label="organization slug">
          <code className="text-xs text-muted">{slug}</code>
        </CopyableValue>
      </div>
      {/* The actions sit under a rule of their own, so the row that commits the
          form is divided from the fields rather than floating after them. */}
      <div className="flex items-center justify-end border-t border-border pt-4">
        <Button
          variant="primary"
          isDisabled={!canEdit || isUnchanged || trimmed === ""}
          isPending={update.isPending}
          onPress={() => update.mutate({ name: trimmed })}
        >
          Save name
        </Button>
      </div>
    </Section>
  )
}

export function OrganizationGeneralPage() {
  const context = useOrganizationContext()

  if (context.isLoading) {
    return <PageLoading label="Loading organization…" />
  }
  if (context.error || !context.data) {
    return (
      <div className="flex flex-col">
        <PageIntro title="Organization" />
        <ErrorBanner error={context.error ?? new Error("No organization.")} />
      </div>
    )
  }

  const { organization, role } = context.data
  const canEdit = canManage(context.data)

  return (
    <div className="flex flex-col">
      <PageIntro title="Organization">
        The organization you are acting in, and what this page renames. The
        first one is provisioned on first boot; the switcher above the rail is
        where you create another or move between them. Your role here is {role}.
      </PageIntro>

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
