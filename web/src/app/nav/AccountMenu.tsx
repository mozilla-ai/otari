import { Button, Popover } from "@heroui/react"
import { Link, type LinkProps } from "@tanstack/react-router"
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

import type { OrganizationContext } from "@/client"
import { useAuth } from "@/features/auth/AuthContext"
import { useOrganizationContext } from "@/shared/api/hooks"
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
// appearance, the legal pages, and the way out. The design's
// "Menu member · Linear order" artboard is the order and the geometry. Who you
// are signed in as is the trigger's own line, not a row inside the menu: the
// menu repeated it under an avatar the trigger already draws, so that block is
// gone and /account is where an identity is actually read.
//
// Three of these are real in a standalone gateway. Appearance drives the dark
// token block globals.css has carried since the design foundation was rehomed,
// logging out ends the session, and Account settings opens the page that owns
// the credential it was minted from (otari#653): a session is per-identity
// since otari#647, and the identity holds a password of its own since otari#649,
// so the row this menu carried disabled from the start now has a destination.
// Data & Privacy stays a disabled row when unset rather than vanishing: the
// settings surface it will become is coming, and a menu that silently lacks it
// reads as a menu that never will.

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
const MENU_ROW = `flex min-h-11 w-full items-center gap-2.5 rounded-md px-2.5 text-left text-shell-label font-medium md:min-h-9 ${NAV_TRANSITION}`
const MENU_ROW_RESTING = "text-foreground hover:bg-surface-alt"
const MENU_ROW_DISABLED = "cursor-not-allowed text-muted opacity-60"
// No vertical margin: the dialog's own 6px gap is the menu's rhythm, and a
// divider that adds to it sits in a wider gutter than the design draws.
// The menu's glyphs are the marks `otari-ai/frontend/src/app/nav/accountDestinations.ts`
// names for the same rows, at the rail's own 16px and muted against the label.
const MENU_ICON_CLASS = `${NAV_ICON_CLASS} text-muted`
const MENU_DIVIDER = "h-px shrink-0 bg-border"

// Whose session this is, as the trigger at the foot of the rail names it: the
// person, not their standing. It used to name a standing, because nothing
// authenticated reported the caller's own identity and the bootstrap's
// `session_type` was the only thing to hand; keyed on that, the trigger read
// "Operator" for every signed-in caller, which is a role rather than a name and
// on a hosted deployment was not even true of the person reading it (#832).
// `OrganizationMembershipContextPublic.caller` is the identity itself, on the
// read `AppShellChrome` already makes, so naming the person costs no request.
//
// A standalone operator still reads "Operator", now because that *is* their
// name: first boot provisions the identity with it (`OPERATOR_FULL_NAME`), and
// the roster shows the same word in the same place.
//
// Three answers, each for a real identity this deployment can hold. A member
// added to the roster by address has no name until they claim it, so the
// address stands in and is a better answer than a role. An identity with
// neither, and a context that has not landed or could not be read, get
// "Signed in": naming nobody is the honest answer, and it is what the trigger
// showed before the first paint anyway.
function sessionIdentity(caller: OrganizationContext["caller"]): {
  name: string
  initials: string
} {
  const named = caller?.full_name?.trim()
  if (named) {
    return { name: named, initials: initialsFor(named) }
  }
  const addressed = caller?.email?.trim()
  if (addressed) {
    // An address has no words to take initials from, so its local part stands
    // in, and that part is split on its own punctuation as well: an address
    // spells a name with dots where a name uses spaces, so `ada.lovelace@…`
    // initials as AL. A name is split on spaces alone, because the same
    // punctuation means something else in one: `Ada Lovelace-Byron` is two
    // names, and splitting the hyphen would initial her as AB.
    return {
      name: addressed,
      initials: initialsFor(localPart(addressed), true),
    }
  }
  return { name: "Signed in", initials: "··" }
}

/** The part of an address that names a person, which is the part before the host. */
function localPart(email: string): string {
  return email.split("@")[0] ?? email
}

// Two letters, taken the way an avatar takes them: the first letter of the
// first and last word, and the first two letters of a source that is one word.
//
// Counted in characters rather than in the code units `[0]` and `slice` count,
// because a name outside the basic plane (an extension-B CJK character, an
// emoji) is two units per character: indexing one takes half a surrogate pair
// and the avatar draws the replacement mark instead of the letter.
function initialsFor(source: string, splitPunctuation = false): string {
  const words = source
    .split(splitPunctuation ? /[\s._-]+/ : /\s+/)
    .filter(Boolean)
  if (words.length === 0) {
    return "··"
  }
  const first = Array.from(words[0])
  const last = Array.from(words[words.length - 1])
  const letters =
    words.length > 1 ? `${first[0]}${last[0]}` : first.slice(0, 2).join("")
  return letters.toUpperCase()
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
        <span className="w-11 shrink-0 text-right text-shell-secondary font-normal text-muted">
          {trailing}
        </span>
      ) : (
        <span aria-hidden="true" className="h-0 w-11 shrink-0" />
      )}
    </button>
  )
}

/**
 * A row that goes to a route in this app.
 *
 * A router `Link`, not `MenuExternalLink`: an `<a href>` to a route would
 * reload the whole shell. It closes the menu on the way, since the popover
 * outlives a client-side navigation that leaves it open over the new page.
 */
function MenuLink({
  label,
  icon: Icon,
  to,
  onNavigate,
  className = "",
}: {
  label: string
  icon: IconType
  /** Typed off `Link` itself, so the route tree is what validates it. */
  to: LinkProps["to"]
  onNavigate: () => void
  className?: string
}) {
  return (
    <Link
      to={to}
      onClick={onNavigate}
      className={`${MENU_ROW} ${MENU_ROW_RESTING} ${className}`}
    >
      <Icon aria-hidden="true" className={MENU_ICON_CLASS} />
      <span className="min-w-0 flex-1 truncate">{label}</span>
      <span aria-hidden="true" className="h-0 w-11 shrink-0" />
    </Link>
  )
}

function MenuExternalLink({
  label,
  icon: Icon,
  href,
  className = "",
}: {
  label: string
  icon: IconType
  href: string
  className?: string
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={`${MENU_ROW} ${MENU_ROW_RESTING} ${className}`}
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
  const { docs_url, terms_url, privacy_url } = useDeployment()
  const organization = useOrganizationContext()
  const [open, setOpen] = useState(false)
  const identity = sessionIdentity(organization.data?.caller)

  return (
    <Popover isOpen={open} onOpenChange={setOpen}>
      {/* HeroUI's Button, not a plain one: the popover wires its trigger through
          react-aria, and a bare <button> leaves it unopenable. `w-auto!` is what
          makes it span the rail, overriding the width the variant sets, which
          otherwise leaves this a pill in the corner. */}
      <Button
        variant="ghost"
        // The identity this control exists to draw, folded into the name: a
        // static `aria-label` wins over the children, so a screen reader
        // otherwise hears "Account" and never who is signed in. On the
        // collapsed rail the name is not rendered at all, so this is the only
        // place it could reach anybody there. `AppearanceControl` folds its own
        // visible state in for the same reason.
        aria-label={`Account: ${identity.name}`}
        className={`${navRowClass({ collapsed })} w-auto! justify-start`}
      >
        <span className="flex h-[1.625rem] w-[1.625rem] shrink-0 items-center justify-center rounded-full border border-border bg-surface-alt text-shell-monogram font-semibold text-muted">
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
          <MenuLink
            label="Account settings"
            icon={FiSettings}
            to="/account"
            onNavigate={() => setOpen(false)}
          />
          <AppearanceControl />
          <div className={MENU_DIVIDER} />
          {/* The top bar owns Documentation above `md` (that cluster is
              `hidden md:flex`), and this menu is the one surface that renders
              inside the mobile drawer, so this row is what keeps documentation
              reachable on a phone. Hidden from `md` up rather than shown
              everywhere, because the design's menu draws no such row.
              It follows the top bar's target: the deployment's own docs site
              when it named one, the bundled guide otherwise. */}
          {docs_url ? (
            <MenuExternalLink
              label="Documentation"
              icon={FiBookOpen}
              href={docs_url}
              className="md:hidden"
            />
          ) : (
            <MenuLink
              label="Documentation"
              icon={FiBookOpen}
              to="/docs"
              onNavigate={() => setOpen(false)}
              className="md:hidden"
            />
          )}
          {terms_url ? (
            <MenuExternalLink
              label="Terms of service"
              icon={FiFileText}
              href={terms_url}
            />
          ) : null}
          {privacy_url ? (
            <MenuExternalLink
              label="Data & Privacy"
              icon={FiShield}
              href={privacy_url}
            />
          ) : (
            <MenuItem
              label="Data & Privacy"
              icon={FiShield}
              title="The gateway stores its data locally and reports nothing outward, so there is nothing to configure here yet."
              isDisabled
            />
          )}
          <div className={MENU_DIVIDER} />
          {/* Neutral, not danger-colored. Ending a session is reversible by
              signing in again, so red here spends the color that marks the
              deletes on the pages behind this menu. */}
          <MenuItem label="Log out" icon={FiLogOut} onPress={logout} />
        </Popover.Dialog>
      </Popover.Content>
    </Popover>
  )
}
