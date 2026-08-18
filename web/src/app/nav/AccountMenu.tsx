import { Button, Popover } from "@heroui/react"
import { Link } from "@tanstack/react-router"
import { useState } from "react"

import { useAuth } from "@/features/auth/AuthContext"
import { useDeployment } from "@/shared/hooks/useDeployment"
import {
  THEME_PREFERENCES,
  type ThemePreference,
  useTheme,
} from "@/shared/hooks/useTheme"

// The account control at the foot of the sidebar, and the menu it opens:
// account settings, appearance, data & privacy, sign out.
//
// Two of those four are real here. Appearance drives the dark token block that
// globals.css has carried since the design foundation was rehomed, and Sign out
// ends the master-key session. The other two are disabled rather than omitted:
// they describe a per-user account, and a standalone gateway has one session
// shared by whoever holds the master key, so there is no account to settle
// preferences against until sign-in lands (otari-ai#1716).

const THEME_LABELS: Record<ThemePreference, string> = {
  system: "System",
  light: "Light",
  dark: "Dark",
}

// Whose session this is. A standalone gateway issues one, for the operator
// identity it provisioned itself, and the membership context does not carry the
// caller's own name, so the session kind is the most this can honestly say.
function sessionIdentity(sessionType: string): {
  name: string
  initials: string
} {
  if (sessionType === "local_operator") {
    return { name: "Operator", initials: "OP" }
  }
  return { name: "Signed in", initials: "··" }
}

function MenuItem({
  label,
  icon,
  onPress,
  isDisabled,
  title,
  danger,
}: {
  label: string
  icon: React.ReactNode
  onPress?: () => void
  isDisabled?: boolean
  title?: string
  danger?: boolean
}) {
  return (
    <button
      type="button"
      disabled={isDisabled}
      title={title}
      onClick={onPress}
      className={`flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-sm ${
        isDisabled
          ? "cursor-not-allowed text-muted opacity-60"
          : danger
            ? "text-danger hover:bg-content3"
            : "text-foreground hover:bg-content3"
      }`}
    >
      {icon}
      {label}
    </button>
  )
}

const iconClass = "h-4 w-4 shrink-0"

const rowClass =
  "flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-sm text-foreground hover:bg-content3"

function MenuLink({
  to,
  label,
  icon,
  onNavigate,
}: {
  to: "/docs"
  label: string
  icon: React.ReactNode
  onNavigate: () => void
}) {
  return (
    <Link to={to} className={rowClass} onClick={onNavigate}>
      {icon}
      {label}
    </Link>
  )
}

export function AccountMenu({ collapsed }: { collapsed: boolean }) {
  const { logout } = useAuth()
  const { session_type } = useDeployment()
  const { preference, setPreference } = useTheme()
  const [open, setOpen] = useState(false)
  const identity = sessionIdentity(session_type)

  return (
    <Popover isOpen={open} onOpenChange={setOpen}>
      <Button
        variant="ghost"
        aria-label="Account"
        className={`flex h-auto items-center gap-2.5 rounded-lg px-2 py-2 text-left ${
          collapsed ? "mx-2 justify-center" : "mx-3"
        }`}
      >
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-content3 text-[10px] font-semibold text-foreground">
          {identity.initials}
        </span>
        {collapsed ? null : (
          <span className="text-sm text-foreground">{identity.name}</span>
        )}
      </Button>
      {/* Opens upward: the control is pinned to the bottom of the rail. */}
      <Popover.Content placement="top start">
        <Popover.Dialog className="flex w-56 flex-col gap-0.5">
          <MenuItem
            label="Account settings"
            title="A standalone gateway has one shared session, so there is no per-user account yet."
            isDisabled
            icon={
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className={iconClass}
              >
                <circle cx="12" cy="8" r="3.5" />
                <path d="M5 20a7 7 0 0 1 14 0" strokeLinecap="round" />
              </svg>
            }
          />
          {/* The dashboard's own docs, bundled with the running gateway. In the
              menu rather than beside it, so the footer is one control. */}
          <MenuLink
            to="/docs"
            label="User guide"
            onNavigate={() => setOpen(false)}
            icon={
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className={iconClass}
              >
                <path
                  d="M12 6.5C10.5 5 8 4.5 4 4.5V18c4 0 6.5.5 8 2 1.5-1.5 4-2 8-2V4.5c-4 0-6.5.5-8 2z"
                  strokeLinejoin="round"
                />
                <path d="M12 6.5V20" strokeLinecap="round" />
              </svg>
            }
          />
          <div className="px-2 pt-1.5 pb-1">
            <div className="flex items-center gap-2.5 pb-1.5 text-sm text-foreground">
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className={iconClass}
              >
                <path
                  d="M20 13a8 8 0 1 1-9-9 6 6 0 0 0 9 9z"
                  strokeLinejoin="round"
                />
              </svg>
              Appearance
            </div>
            <fieldset className="flex gap-1">
              <legend className="sr-only">Appearance</legend>
              {THEME_PREFERENCES.map((option) => (
                <label
                  key={option}
                  className={`flex-1 cursor-pointer rounded-md border px-2 py-1 text-center text-xs ${
                    preference === option
                      ? "border-accent bg-primary-subtle text-foreground"
                      : "border-border text-muted hover:bg-content3"
                  }`}
                >
                  {/* A real radio, visually replaced by the label around it, so
                      the group is one tab stop and arrow-navigable. */}
                  <input
                    type="radio"
                    name="appearance"
                    className="sr-only"
                    checked={preference === option}
                    onChange={() => setPreference(option)}
                  />
                  {THEME_LABELS[option]}
                </label>
              ))}
            </fieldset>
          </div>
          <MenuItem
            label="Data & privacy"
            title="The gateway stores its data locally and reports nothing outward, so there is nothing to configure here yet."
            isDisabled
            icon={
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className={iconClass}
              >
                <path
                  d="M12 3l7 3v6c0 4-3 7-7 9-4-2-7-5-7-9V6z"
                  strokeLinejoin="round"
                />
              </svg>
            }
          />
          <div className="my-1 border-t border-border" />
          <MenuItem
            label="Sign out"
            danger
            onPress={logout}
            icon={
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className={iconClass}
              >
                <path
                  d="M10 5H6a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h4M16 15l3-3-3-3M19 12H10"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            }
          />
        </Popover.Dialog>
      </Popover.Content>
    </Popover>
  )
}
