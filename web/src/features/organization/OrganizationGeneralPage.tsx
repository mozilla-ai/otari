import { useState } from "react"
import { Button } from "@/design-system/actions/Button"
import { CopyableValue } from "@/design-system/actions/CopyField"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { Section } from "@/design-system/layout/Section"
import {
  useOrganizationContext,
  useUpdateOrganization,
} from "@/shared/api/organizations"

import { RenameOrganizationDialog } from "./RenameOrganizationDialog"
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
 * The detail band: a rule above it, a rule below it, and a third rule dividing
 * what the organization is from the button that changes it. No card, because a
 * page with one band on it does not need a box to say where the band is; the
 * rules already do, and the box was the only thing making this page look like a
 * different page from Keys, which has the same shape.
 *
 * The values read rather than edit: a name in an open input is one stray
 * keystroke from being changed, so the rename is a dialog away.
 */
function OrganizationDetails({
  name,
  slug,
  canEdit,
  onRename,
}: {
  name: string
  slug: string
  canEdit: boolean
  onRename: () => void
}) {
  return (
    <Section
      aria-labelledby="organization-details-title"
      className="border-y border-border py-5"
      contentClassName="flex flex-col gap-4"
    >
      <h2 id="organization-details-title" className="text-title">
        Details
      </h2>
      <dl className="grid items-baseline gap-x-6 gap-y-3 sm:grid-cols-[10rem_1fr]">
        <dt className="text-muted">Organization name</dt>
        <dd className="text-emphasis">{name}</dd>
        <dt className="text-muted">Slug</dt>
        <dd className="flex flex-col gap-1">
          <CopyableValue value={slug} label="organization slug">
            <code className="text-xs text-muted">{slug}</code>
          </CopyableValue>
          <span className="text-caption text-subtle">
            Set when the organization is provisioned. It does not follow a
            rename.
          </span>
        </dd>
      </dl>
      {/* The action sits under a rule of its own, so the control that changes
          the organization is divided from what the organization is. Absent
          rather than disabled for a member, the way Email domains and Provider
          keys drop their own primary action: the banner above has already said
          why, and a rule under the values with nothing beneath it is a band
          that looks unfinished. */}
      {canEdit ? (
        <div className="flex items-center justify-end border-t border-border pt-4">
          <Button variant="primary" onPress={onRename}>
            Change organization name
          </Button>
        </div>
      ) : null}
    </Section>
  )
}

export function OrganizationGeneralPage() {
  const context = useOrganizationContext()
  const update = useUpdateOrganization()
  const [isRenaming, setIsRenaming] = useState(false)

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

      <OrganizationDetails
        name={organization.name}
        slug={organization.slug}
        canEdit={canEdit}
        onRename={() => {
          // A rejected attempt must not be waiting in the dialog the next time
          // it opens.
          update.reset()
          setIsRenaming(true)
        }}
      />

      <RenameOrganizationDialog
        isOpen={isRenaming}
        onOpenChange={setIsRenaming}
        currentName={organization.name}
        isPending={update.isPending}
        error={update.error}
        onSubmit={(name) =>
          update.mutate({ name }, { onSuccess: () => setIsRenaming(false) })
        }
      />
    </div>
  )
}
