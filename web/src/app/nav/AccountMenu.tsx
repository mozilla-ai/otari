import { Button, Popover } from "@heroui/react"
import { Link } from "@tanstack/react-router"
import { useState } from "react"
import type { IconType } from "react-icons"
import {
  FiBookOpen,
  FiChevronDown,
  FiFileText,
  FiLogOut,
  FiMoon,
  FiSettings,
  FiShield,
} from "react-icons/fi"

import { useAuth } from "@/features/auth/AuthContext"
import { EntitlementGate } from "@/shared/components/EntitlementGate"
import { useDeployment } from "@/shared/hooks/useDeployment"
import {
  THEME_PREFERENCES,
  type ThemePreference,
  useTheme,
} from "@/shared/hooks/useTheme"
import {
  NAV_ICON_CLASS,
  NAV_TRANSITION,
  navIndicatorClass,
  navRowClass,
} from "./rowStyles"

// The control that ends the sidebar, and the menu it opens: account settings,
// appearance, the legal pages, who you are signed in as, and the way out. The
// design's "Menu member · Linear order" artboard is the order and the geometry.
//
// Two of these are real in a standalone gateway. Appearance drives the dark
// token block globals.css has carried since the design foundation was rehomed,
// and logging out ends the master-key session. The rest describe a per-user
// account, and this deployment has one session shared by whoever holds the
// master key: Account settings and Data & Privacy are disabled with the reason
// rather than omitted, because they are coming (otari-ai#1716) and a menu that
// silently lacks them reads as a menu that will never have them. Terms of
// service is different again, and gated: it is a hosted document, so it appears
// only for a deployment that has one to point at.

const THEME_LABELS: Record<ThemePreference, string> = {
  system: "System",
  light: "Light",
  dark: "Dark",
}

// 36px rows at 13.5px, which is the menu's own scale: a step down from the
// rail's 44px/14px, because a menu row is read once on the way to a decision
// rather than scanned as a standing list.
//
// That 36px is a desk figure. This menu also renders inside the mobile drawer,
// where these rows are tapped, so below `md` they take the 44px touch floor and
// the menu's own scale resumes at the breakpoint.
const MENU_ROW = `flex min-h-11 w-full items-center gap-2.5 rounded-md px-2.5 text-left text-chrome-row font-medium md:min-h-9 ${NAV_TRANSITION}`
// `cursor-pointer` because these rows are `<button>`s and Tailwind's reset
// leaves a button on the default arrow, which reads as inert next to the
// anchors in the same menu that get the hand for free.
const MENU_ROW_RESTING = "cursor-pointer text-foreground hover:bg-surface-alt"
const MENU_ROW_DISABLED = "cursor-not-allowed text-muted opacity-60"
// No vertical margin: the dialog's own 6px gap is the menu's rhythm, and a
// divider that adds to it sits in a wider gutter than the design draws.
// The menu's glyphs are the marks `otari-ai/frontend/src/app/nav/accountDestinations.ts`
// names for the same rows, at the rail's own 16px and muted against the label.
const MENU_ICON_CLASS = `${NAV_ICON_CLASS} text-muted`
const MENU_DIVIDER = "h-px shrink-0 bg-border"

// Whose session this is. A standalone gateway issues one, for the operator
// identity it provisioned itself, and the membership context does not carry the
// caller's own name, so the session kind is the most this can honestly say. The
// design's identity block shows a name over an email; there is no email here, so
// the second line names the credential instead of inventing an address.
function sessionIdentity(sessionType: string): {
  name: string
  initials: string
  detail: string
} {
  if (sessionType === "local_operator") {
    return {
      name: "Operator",
      initials: "OP",
      detail: "Master-key session",
    }
  }
  return { name: "Signed in", initials: "··", detail: "This gateway" }
}

function MenuItem({
  label,
  icon: Icon,
  onPress,
  isDisabled,
  title,
  trailing,
  ariaLabel,
}: {
  label: string
  /** A Feather mark, named at the call site and dressed here. */
  icon: IconType
  onPress?: () => void
  isDisabled?: boolean
  title?: string
  trailing?: string
  ariaLabel?: string
}) {
  return (
    <button
      type="button"
      disabled={isDisabled}
      title={title}
      // A disabled button takes no focus, so the tooltip is pointer-only. Fold
      // the reason into the name instead, which a screen reader still reads
      // when browsing past the item.
      aria-label={
        isDisabled && title ? `${label} (${title})` : (ariaLabel ?? undefined)
      }
      onClick={onPress}
      className={`${MENU_ROW} ${isDisabled ? MENU_ROW_DISABLED : MENU_ROW_RESTING}`}
    >
      <Icon aria-hidden="true" className={MENU_ICON_CLASS} />
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {/* The trailing lane is held open on every row, at the width a value
          takes, so the one row that carries a value does not push its own label
          out of the column the others sit in. */}
      {trailing ? (
        <span className="w-11 shrink-0 text-right text-xs font-normal text-muted">
          {trailing}
        </span>
      ) : (
        <span aria-hidden="true" className="h-0 w-11 shrink-0" />
      )}
    </button>
  )
}

/**
 * Documentation, which the top bar owns above `md` and this menu carries below
 * it: that cluster is `hidden md:flex`, and this menu is the one surface that
 * renders inside the mobile drawer, so without this row the guide bundled with
 * the gateway has no control pointing at it on a phone. Hidden from `md` up
 * rather than shown everywhere, because the design's menu draws no such row and
 * above the breakpoint the top bar already answers.
 *
 * A router `Link`, not `MenuExternalLink`: `/docs` is a route in this app, and
 * an `<a href>` to it would reload the whole shell.
 */
function MenuDocumentationLink({ onNavigate }: { onNavigate: () => void }) {
  return (
    <Link
      to="/docs"
      onClick={onNavigate}
      className={`${MENU_ROW} ${MENU_ROW_RESTING} md:hidden`}
    >
      <FiBookOpen aria-hidden="true" className={MENU_ICON_CLASS} />
      <span className="min-w-0 flex-1 truncate">Documentation</span>
      <span aria-hidden="true" className="h-0 w-11 shrink-0" />
    </Link>
  )
}

function MenuExternalLink({
  label,
  icon: Icon,
  href,
}: {
  label: string
  icon: IconType
  href: string
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={`${MENU_ROW} ${MENU_ROW_RESTING}`}
    >
      <Icon aria-hidden="true" className={MENU_ICON_CLASS} />
      <span className="min-w-0 flex-1 truncate">{label}</span>
      <span aria-hidden="true" className="h-0 w-11 shrink-0" />
    </a>
  )
}

/**
 * Appearance: one row that names the current preference and cycles through the
 * three, system → light → dark → system.
 *
 * The design draws only the closed state, a row with "System" on the right,
 * which is what a menu wants: the setting is one line, not three. It was
 * previously a radio group, which spent three rows of the menu on a setting
 * nobody opened the menu for, and then a segmented control, which spent one row
 * on three targets 40px wide. A cycling row is the same one line the design
 * draws and needs no second surface to open into.
 *
 * `System` stays in the cycle rather than being the off state of a light/dark
 * pair: it is a preference in its own right, and the only value that keeps
 * following the OS after the fact.
 *
 * The trailing value is the visible state, and `aria-label` is the same fact
 * for a screen reader plus what activating will do, because a button whose
 * meaning changes on every press cannot say it in a static name.
 */
function AppearanceControl() {
  const { preference, setPreference } = useTheme()
  const next =
    THEME_PREFERENCES[
      (THEME_PREFERENCES.indexOf(preference) + 1) % THEME_PREFERENCES.length
    ]

  return (
    <MenuItem
      label="Appearance"
      icon={FiMoon}
      trailing={THEME_LABELS[preference]}
      ariaLabel={`Appearance: ${THEME_LABELS[preference]}. Switch to ${THEME_LABELS[next]}.`}
      onPress={() => setPreference(next)}
    />
  )
}

export function AccountMenu({ collapsed }: { collapsed: boolean }) {
  const { logout } = useAuth()
  const { session_type, management_url } = useDeployment()
  const [open, setOpen] = useState(false)
  const identity = sessionIdentity(session_type)

  return (
    <Popover isOpen={open} onOpenChange={setOpen}>
      {/* HeroUI's Button, not a plain one: the popover wires its trigger through
          react-aria, and a bare <button> leaves it unopenable. `w-auto!` is what
          makes it span the rail, overriding the width the variant sets, which
          otherwise leaves this a pill in the corner. */}
      <Button
        variant="ghost"
        aria-label="Account"
        className={`${navRowClass({ collapsed })} w-auto! justify-start`}
      >
        <span className="flex h-[1.625rem] w-[1.625rem] shrink-0 items-center justify-center rounded-full border border-border bg-surface-alt text-chrome-initials font-semibold text-muted">
          {identity.initials}
        </span>
        {collapsed ? null : (
          <>
            <span className="min-w-0 flex-1 truncate text-left text-foreground">
              {identity.name}
            </span>
            <FiChevronDown
              aria-hidden="true"
              className={`text-muted ${navIndicatorClass({ open })}`}
            />
          </>
        )}
      </Button>
      {/* Opens upward: the control is pinned to the bottom of the rail. */}
      <Popover.Content placement="top start">
        {/* Named, like the collapsed nav flyout and the switcher's create modal:
            a dialog with no accessible name is announced as an unnamed one, and
            the trigger's own name does not carry over to it. */}
        <Popover.Dialog
          aria-label="Account"
          className="flex w-[17rem] flex-col gap-1.5"
        >
          <MenuItem
            label="Account settings"
            icon={FiSettings}
            title="A standalone gateway has one shared session, so there is no per-user account yet."
            isDisabled
          />
          <AppearanceControl />
          <div className={MENU_DIVIDER} />
          <MenuDocumentationLink onNavigate={() => setOpen(false)} />
          {/* Hosted-only, and gated twice over: the entitlement says the
              deployment has terms to show, and `management_url` is where they
              are. A self-hosted gateway is neither, so the row is absent rather
              than pointing somewhere invented. */}
          {management_url ? (
            <EntitlementGate capability="legal.terms">
              <MenuExternalLink
                label="Terms of service"
                icon={FiFileText}
                href={`${management_url.replace(/\/$/, "")}/terms`}
              />
            </EntitlementGate>
          ) : null}
          <MenuItem
            label="Data & Privacy"
            icon={FiShield}
            title="The gateway stores its data locally and reports nothing outward, so there is nothing to configure here yet."
            isDisabled
          />
          <div className={MENU_DIVIDER} />
          {/* Who you are, at the foot of the menu rather than in the trigger:
              the rail has room for one line, and this is where the design puts
              the second. */}
          <div className="flex items-center gap-2.5 px-2.5 py-2">
            <span className="flex h-[1.875rem] w-[1.875rem] shrink-0 items-center justify-center rounded-full bg-surface-alt text-xs leading-[0.875rem] font-semibold text-muted">
              {identity.initials}
            </span>
            <span className="flex min-w-0 flex-1 flex-col gap-0.5 text-left">
              <span className="truncate text-chrome-row font-semibold text-foreground">
                {identity.name}
              </span>
              <span className="truncate text-chrome-meta text-muted">
                {identity.detail}
              </span>
            </span>
          </div>
          {/* Neutral, not danger-colored. Ending a session is reversible by
              signing in again, so red here spends the color that marks the
              deletes on the pages behind this menu. */}
          <MenuItem label="Log out" icon={FiLogOut} onPress={logout} />
        </Popover.Dialog>
      </Popover.Content>
    </Popover>
  )
}
