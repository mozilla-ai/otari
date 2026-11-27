import type { ReactNode } from "react"
import { FiMoon, FiSun } from "react-icons/fi"
import { IconButton } from "@/design-system/actions/IconButton"
import { ProductLockup } from "@/design-system/ProductMark"
import { useTheme } from "@/shared/hooks/useTheme"

/**
 * The frame the dashboard shows before it knows which deployment it is on: the
 * sign-in shell's header, card and footer, without anything that reads the
 * bootstrap.
 *
 * Its geometry is `LoginPageShell`'s, copied rather than shared because that
 * shell reads `useDeployment` (the lockup links to the deployment's home page),
 * and nothing here has a deployment yet. Matching it is the point: when the
 * gateway answers, the sign-in card takes this card's exact place and only the
 * contents change. `index.html` paints the same frame before the bundle loads,
 * in plain CSS; a change to the geometry here goes there too.
 *
 * `isScanning` runs the border sweep `Dialog` uses for a frame that is waiting
 * on something, in danger ink, for as long as retries continue.
 */
export function BootFrame({
  children,
  isScanning = false,
}: {
  children: ReactNode
  isScanning?: boolean
}) {
  const { resolved, toggle } = useTheme()
  const isDark = resolved === "dark"
  const ThemeIcon = isDark ? FiMoon : FiSun

  return (
    <div className="relative isolate flex min-h-svh flex-col bg-background">
      <header className="relative z-10 flex min-h-14 shrink-0 items-center justify-between gap-4 border-b border-border bg-background px-4 md:px-6">
        <div className="flex items-center">
          <ProductLockup className="h-9 w-auto text-accent" />
        </div>
        <IconButton
          variant="ghost"
          isIconOnly
          label={isDark ? "Switch to light theme" : "Switch to dark theme"}
          onPress={toggle}
          className="-mr-3"
        >
          <ThemeIcon aria-hidden />
        </IconButton>
      </header>
      {/* Pinned to the top rather than centered, as the sign-in shell is: the
          card's height differs between failures, and a centered card would move
          its heading whenever the failure changed. */}
      <main className="relative isolate flex flex-1 flex-col items-center px-4 py-4 md:pt-15">
        <div
          data-boot-card=""
          className={`relative z-10 flex w-full max-w-md shrink-0 flex-col gap-4 border border-border-strong bg-surface p-6 md:px-8 ${
            isScanning
              ? "otari-scan-border [--scan-ink:var(--color-danger)]"
              : ""
          }`}
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
      <footer className="relative z-10 flex min-h-14 shrink-0 items-center justify-between gap-4 border-t border-border bg-background px-4 pb-[env(safe-area-inset-bottom)] text-subtle md:px-6">
        <span>One gateway. Every model.</span>
        <span className="text-mono-caption">Mozilla AI</span>
      </footer>
    </div>
  )
}

/**
 * The frame while the first attempt is out, for a caller that mounts before it
 * settles. `main.tsx` does not, so in the product this is `index.html`'s copy.
 */
export function BootConnecting() {
  return (
    <BootFrame>
      <p className="py-6 text-center text-sm text-subtle">
        Connecting to the gateway…
      </p>
    </BootFrame>
  )
}
