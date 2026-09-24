import { type ReactNode, useRef } from "react"
import { FiMonitor, FiMoon, FiSun } from "react-icons/fi"
import { IconButton } from "@/design-system/actions/IconButton"
import { siteHomeHref } from "@/features/models/publicCatalog"
import { useDeployment } from "@/shared/hooks/useDeployment"
import { THEME_PREFERENCES, useTheme } from "@/shared/hooks/useTheme"
import { LoginBackground } from "./background/LoginBackground"
import savedBackground from "./background/login-background.json"

function BrandMark() {
  const href = siteHomeHref(useDeployment())
  const mark = (
    <>
      <img
        src={`${import.meta.env.BASE_URL}favicon.svg`}
        alt=""
        width={273}
        height={250}
        className="h-6 w-[1.638rem]"
      />
      <span className="text-title transition-colors group-hover:text-muted motion-reduce:transition-none">
        Otari
      </span>
    </>
  )
  if (!href) return <div className="flex items-center gap-3">{mark}</div>
  return (
    <a
      href={href}
      aria-label="Otari home"
      className="group -mx-1 flex min-h-11 items-center gap-3 px-1"
    >
      {mark}
    </a>
  )
}

export function LoginPageShell({ children }: { children: ReactNode }) {
  const panelRef = useRef<HTMLDivElement>(null)
  const { preference, setPreference } = useTheme()
  const next =
    THEME_PREFERENCES[
      (THEME_PREFERENCES.indexOf(preference) + 1) % THEME_PREFERENCES.length
    ]
  const ThemeIcon =
    preference === "system" ? FiMonitor : preference === "dark" ? FiMoon : FiSun

  return (
    <div className="relative isolate flex min-h-svh flex-col bg-background">
      <header className="relative z-10 flex min-h-14 shrink-0 items-center justify-between gap-4 border-b border-border bg-background px-4 md:px-6">
        <BrandMark />
        {/* No `md:` step down: the header is `min-h-14`, so a 44px target fits
            inside it at every width without moving the row. */}
        <IconButton
          variant="ghost"
          isIconOnly
          label={`Appearance: ${preference}. Switch to ${next}.`}
          onPress={() => setPreference(next)}
        >
          <ThemeIcon aria-hidden />
        </IconButton>
      </header>
      <main className="relative isolate flex flex-1 flex-col items-center justify-center px-4 py-4">
        <LoginBackground panelRef={panelRef} config={savedBackground} />
        <div
          ref={panelRef}
          className="relative z-10 flex w-full max-w-md shrink-0 flex-col gap-4 border border-border-strong bg-surface p-6 md:px-8"
        >
          <span
            aria-hidden
            className="pointer-events-none absolute -top-px -left-px size-2 border-t-2 border-l-2 border-accent"
          />
          <span
            aria-hidden
            className="pointer-events-none absolute -right-px -bottom-px size-2 border-r-2 border-b-2 border-accent"
          />
          {children}
        </div>
      </main>
      <footer className="relative z-10 flex min-h-14 shrink-0 items-center justify-between gap-4 border-t border-border bg-background px-4 text-subtle md:px-6">
        <span>One gateway. Every model.</span>
        <span className="text-mono-caption">Mozilla AI</span>
      </footer>
    </div>
  )
}
