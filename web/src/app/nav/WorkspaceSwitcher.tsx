import { Button, Modal, Popover } from "@heroui/react"
import { useNavigate } from "@tanstack/react-router"
import { useState } from "react"
import { FiCheck, FiChevronDown, FiPlus } from "react-icons/fi"
import { CreateOrganizationForm } from "@/features/organization/CreateOrganizationForm"
import { canManage } from "@/features/organization/roles"
import { CreateWorkspaceForm } from "@/features/workspaces/WorkspacesPage"
import {
  useOrganizationContext,
  useOrganizationMemberships,
  useSwitchOrganization,
} from "@/shared/api/hooks"
import { ErrorBanner } from "@/shared/components/ui"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { NAV_TRANSITION, navIndicatorClass } from "./rowStyles"

// The menu's own rhythm, which is the rail's: a 44px row and a 32px heading
// block. The eyebrow above the organization is shorter (28px), because it opens
// the menu rather than separating two groups inside it.
const MENU_HEADING = "flex items-center px-2.5 text-overline"
const MENU_ROW = `flex min-h-11 w-full items-center gap-2.5 rounded-md px-2.5 text-left text-sm font-medium ${NAV_TRANSITION}`
const MENU_ROW_RESTING = "text-foreground hover:bg-surface-alt"
// The current *workspace* is a tinted chip here, unlike the rail, where
// selection is a lifted one. In a menu the tint is the only thing that can carry
// "this is the one you are in" alongside the check; on the rail the fill is read
// against the rail's own ground, which a tint fights.
const MENU_ROW_CURRENT = "bg-primary-subtle text-primary-subtle-foreground"
// The dividers are inset to the rows' own text lane rather than run edge to
// edge, so they separate the groups without drawing a line across the card.
const MENU_DIVIDER = "mx-2.5 my-1 h-px shrink-0 bg-border"

// The marks `otari-ai/frontend`'s scope switcher uses for the same two rows:
// a 16px check on the current scope, a 20px plus on the row that creates one.
function CheckMark() {
  return (
    <>
      <FiCheck
        aria-hidden="true"
        className="size-4 shrink-0 text-primary-subtle-foreground"
      />
      <span className="sr-only">Selected</span>
    </>
  )
}

function PlusMark() {
  return <FiPlus aria-hidden="true" className="size-5 shrink-0" />
}

// The organization and the workspace the shell is looking at, and the control
// that changes the second. Sits above the nav rather than in it because it
// scopes the destinations below it rather than being one.
//
// It scopes what the gateway both records and resolves per workspace: members,
// API keys, the request log, and the spend and volume charts over it. Two
// things it does not, and the copy in the popover says so rather than implying
// a scope that is not there. Routing policies and aliases carry a workspace
// column but are all stored in the default one on purpose, because resolution
// reads a process-wide name-keyed cache, so filtering them would hide live
// policies. Provider credentials are process-wide config rather than a
// workspace row.
export function WorkspaceSwitcher({
  collapsed,
  createHold,
}: {
  collapsed: boolean
  // Forwarded to the create form's own hold, and only ever set by tests:
  // supplying the beat as a gate is what lets a test enter and leave the
  // dismissal window on purpose, instead of sleeping past a duration it cannot
  // await. The app never passes it, so the default stands.
  createHold?: () => Promise<void>
}) {
  const { memberships, selected, select, isLoading } = useSelectedWorkspace()
  const context = useOrganizationContext()
  const navigate = useNavigate()
  const organizations = useOrganizationMemberships()
  const switchOrganization = useSwitchOrganization()
  const [open, setOpen] = useState(false)
  const [creating, setCreating] = useState(false)
  const [creatingOrganization, setCreatingOrganization] = useState(false)
  // Owners and admins only, which is what the server says of
  // `POST /v1/workspaces` and what the Workspaces page gates its own create
  // control on. Without it a member or a viewer is handed the whole form from
  // the scope menu and meets the refusal as a 403 after typing a name.
  const managesOrganization = canManage(context.data)
  // Every organization the caller is an active member of. One row is the
  // overwhelmingly common case (a deployment provisions one and most keep
  // exactly that), and a single row is stated rather than offered as a control:
  // a menu item whose only effect is to close the menu reads as broken.
  const organizationRows = organizations.data ?? []
  const switchable = organizationRows.length > 1

  // Both hops optional: the context can answer without an organization (a
  // failed read, or a shape a test supplies), and the switcher is chrome that
  // renders around whatever else is wrong rather than taking the shell down.
  const organizationName = context.data?.organization?.name ?? "Organization"

  const workspaceName = isLoading
    ? "Loading…"
    : (selected?.name ?? "No workspace")

  return (
    <>
      <Popover isOpen={open} onOpenChange={setOpen}>
        {/* HeroUI's Button, not a plain one: the popover wires its trigger through
          react-aria. `w-auto!` overrides the width the variant sets, which
          otherwise stops this short of the rail rather than spanning it.

          Collapsing narrows the trigger to the mark, but it stays the same
          trigger: the rail's collapsed state is remembered, so a switcher that
          became a plain <div> there would leave an operator unable to change
          workspace until they expanded the rail again. */}
        <Button
          variant="ghost"
          // Names the current workspace rather than replacing it: the label
          // overrides the visible text for assistive tech, so "Switch workspace"
          // alone would make the one thing this control reports unreadable.
          // No `title` companion, which HeroUI's Button does not take: collapsed,
          // the popover itself is what names the current workspace (it marks it
          // with a check), and this label is what assistive tech reads.
          aria-label={`Switch workspace, currently ${workspaceName} in ${organizationName}`}
          // 56px tall in both states, so the rail's first block is the same height
          // whichever context it is in: the organization rail's "Back to" row sits
          // in a box of exactly this height. The fill is the rail's own ground with
          // a border, not a white card, which is what keeps it reading as part of
          // the chrome rather than as the first item in the list.
          className={
            collapsed
              ? `min-h-14 w-full! items-center justify-center rounded-[0.625rem] border border-border bg-background-alt px-0 hover:border-accent ${NAV_TRANSITION}`
              : `min-h-14 w-full! items-center justify-start gap-2.5 rounded-[0.625rem] border border-border bg-background-alt px-2.5 py-2 text-left hover:border-accent ${NAV_TRANSITION}`
          }
        >
          {/* The mark is the switcher's hero, as in the prototype: the product
            name is not repeated in the header, so this is where it lives. */}
          <img
            src="/favicon.svg"
            alt=""
            className="h-[1.875rem] w-[1.875rem] shrink-0"
          />
          {collapsed ? null : (
            <>
              <span className="flex min-w-0 flex-1 flex-col gap-px">
                <span className="truncate text-sm leading-[1.125rem] font-semibold tracking-[-0.01em] text-foreground">
                  {workspaceName}
                </span>
                <span className="truncate text-chrome-meta font-medium text-muted">
                  {organizationName}
                </span>
              </span>
              <FiChevronDown
                aria-hidden="true"
                className={`text-muted ${navIndicatorClass({ open })}`}
              />
            </>
          )}
        </Button>
        <Popover.Content placement="bottom start">
          {/* Named for the same reason the create modal below is: a dialog with
              no accessible name is announced as an unnamed one, and the
              trigger's name does not carry over to the surface it opens. */}
          <Popover.Dialog
            aria-label="Switch workspace or organization"
            className="flex w-[19.75rem] flex-col"
          >
            <p className={`${MENU_HEADING} min-h-7`}>Organization</p>
            {/* The switch is the one write this menu makes that changes what
              every other page is looking at, so its failure is reported here
              rather than by closing the menu on a scope that did not move. */}
            <ErrorBanner error={switchOrganization.error} />
            {switchable ? (
              <ul className="flex flex-col">
                {organizationRows.map((membership) => (
                  <li key={membership.organization.id}>
                    <button
                      type="button"
                      className={`${MENU_ROW} ${
                        membership.is_active_organization
                          ? MENU_ROW_CURRENT
                          : MENU_ROW_RESTING
                      }`}
                      // Pending on the whole group rather than per row: the
                      // mutation is one at a time and a second click during it
                      // would queue a switch onto a cache already being
                      // rebuilt.
                      disabled={switchOrganization.isPending}
                      onClick={() => {
                        if (membership.is_active_organization) {
                          setOpen(false)
                          return
                        }
                        switchOrganization.mutate(
                          membership.organization.id,
                          // Closed on success only, so the banner above has
                          // somewhere to be read.
                          { onSuccess: () => setOpen(false) },
                        )
                      }}
                    >
                      <span className="min-w-0 flex-1 truncate">
                        {membership.organization.name}
                      </span>
                      {membership.is_active_organization ? (
                        <CheckMark />
                      ) : (
                        <span aria-hidden="true" className="size-4 shrink-0" />
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              /* One organization, so this states the scope rather than offering
                to change it. It still carries the check, because the design's
                menu marks the current scope at both levels and a check that only
                ever appears on one of them reads as an incomplete list. */
              <div className={`${MENU_ROW} text-foreground`}>
                <span className="min-w-0 flex-1 truncate">
                  {organizationName}
                </span>
                <CheckMark />
              </div>
            )}
            <div className={MENU_DIVIDER} />
            <p className={`${MENU_HEADING} min-h-8`}>
              Workspaces ({memberships.length})
            </p>
            {isLoading ? (
              <p className="px-2.5 py-1 text-chrome-meta text-muted">
                Loading workspaces…
              </p>
            ) : memberships.length === 0 ? (
              <p className="px-2.5 py-1 text-chrome-meta text-muted">
                You do not belong to a workspace yet.
              </p>
            ) : (
              <ul className="flex flex-col">
                {memberships.map((membership) => {
                  const isCurrent =
                    membership.workspace_id === selected?.workspace_id
                  return (
                    <li key={membership.workspace_id}>
                      <button
                        type="button"
                        className={`${MENU_ROW} ${
                          isCurrent ? MENU_ROW_CURRENT : MENU_ROW_RESTING
                        }`}
                        onClick={() => {
                          select(membership.workspace_id)
                          setOpen(false)
                        }}
                      >
                        <span className="min-w-0 flex-1 truncate">
                          {membership.name}
                        </span>
                        {/* The check is the only thing distinguishing the current
                          workspace, so it carries text a screen reader reads
                          rather than an attribute the role does not. */}
                        {isCurrent ? (
                          <CheckMark />
                        ) : (
                          <span
                            aria-hidden="true"
                            className="size-4 shrink-0"
                          />
                        )}
                      </button>
                    </li>
                  )
                })}
              </ul>
            )}
            <div className={MENU_DIVIDER} />
            {managesOrganization ? (
              <button
                type="button"
                className={`${MENU_ROW} font-semibold text-muted hover:bg-surface-alt hover:text-foreground`}
                onClick={() => {
                  setOpen(false)
                  setCreating(true)
                }}
              >
                <PlusMark />
                <span className="min-w-0 flex-1 truncate">
                  Create workspace
                </span>
              </button>
            ) : null}
            {/* Ungated, unlike Create workspace above it. Creating an
              organization is not an action inside one, so there is no role in
              one to check it against, which is also why the server gates it on
              the management credential alone. */}
            <button
              type="button"
              className={`${MENU_ROW} text-muted hover:bg-surface-alt hover:text-foreground`}
              onClick={() => {
                setOpen(false)
                setCreatingOrganization(true)
              }}
            >
              <PlusMark />
              <span className="min-w-0 flex-1 truncate">
                Create organization
              </span>
            </button>
          </Popover.Dialog>
        </Popover.Content>
      </Popover>
      {/* Reuses the Workspaces page's own form rather than restating its fields:
          the popover has no room for a form, and it dismisses on the first click
          outside itself, which a name field cannot survive. */}
      <Modal isOpen={creating} onOpenChange={setCreating}>
        {/* The menu row is the trigger, and it lives inside a popover that has
            already dismissed by the time this opens, so the modal is driven from
            state instead. HeroUI still renders a press responder for the trigger
            slot and warns when nothing fills it, which is why this is hidden
            rather than absent; `SettingsPage` does the same for its own dialog. */}
        <Modal.Trigger className="hidden">Create workspace</Modal.Trigger>
        {/* An explicit dim: HeroUI maps `--backdrop` to opaque black, which the
            AlertDialog softens itself and the Modal does not, so without this the
            page behind the form goes fully black. */}
        <Modal.Backdrop className="bg-backdrop/50">
          {/* A fixed width, not the content's own. The container is the
              `sm:w-fit` element and `size="md"` only caps the dialog at
              `max-w-md`, so the modal was as wide as whatever was in it: a
              message longer than the fields made it jump out to the cap the
              moment one appeared. 28rem is that cap, so this pins the width it
              was already growing to, and the dialog's own `w-full` fills it.
              Below `sm` the container is `w-full` and this does not apply. */}
          <Modal.Container
            placement="center"
            size="md"
            className="sm:w-[28rem]"
          >
            <Modal.Dialog aria-label="Create workspace" className="p-0">
              <CreateWorkspaceForm
                onClose={() => setCreating(false)}
                // Creating from the scope switcher is a request to work in the
                // new workspace, so the flow ends inside it rather than back on
                // the page it was started from: the shell's scope moves, and the
                // overview is where that scope reads. Selecting by id is enough
                // even though the switcher's list comes from the organization
                // context: the mutation invalidates that context, and the
                // selection resolves against the membership as soon as it
                // arrives. Navigating also leaves whatever organization-scoped
                // page the operator was on, which the new scope does not apply
                // to.
                onCreated={(workspace) => {
                  select(workspace.id)
                  void navigate({ to: "/" })
                }}
                hold={createHold}
              />
            </Modal.Dialog>
          </Modal.Container>
        </Modal.Backdrop>
      </Modal>
      {/* Its own modal rather than one dialog switching on which row opened it:
          the two forms share no field, and a single state would have to encode
          "which" as well as "open". */}
      <Modal
        isOpen={creatingOrganization}
        onOpenChange={setCreatingOrganization}
      >
        <Modal.Trigger className="hidden">Create organization</Modal.Trigger>
        <Modal.Backdrop className="bg-backdrop/50">
          <Modal.Container placement="center" size="md">
            <Modal.Dialog aria-label="Create organization" className="p-0">
              <CreateOrganizationForm
                onClose={() => setCreatingOrganization(false)}
              />
            </Modal.Dialog>
          </Modal.Container>
        </Modal.Backdrop>
      </Modal>
    </>
  )
}
