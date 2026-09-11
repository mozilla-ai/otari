import { type ReactNode, useLayoutEffect, useRef, useState } from "react"
import { FiMoon, FiPause, FiPlay, FiSun } from "react-icons/fi"
import { Button } from "@/design-system/actions/Button"
import { useTheme } from "@/shared/hooks/useTheme"
import { BAR_ROW_RATIO } from "./background/config"
import { LoginBackground } from "./background/LoginBackground"
import savedBackground from "./background/login-background.json"
import "./login.css"

export function LoginPageShell({ children }: { children: ReactNode }) {
  const panelRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const [paused, setPaused] = useState(false)
  const { resolved, setPreference } = useTheme()
  useLayoutEffect(() => {
    const panel = panelRef.current
    const content = contentRef.current
    if (!panel || !content) return
    const align = () => {
      const pitch =
        (panel.getBoundingClientRect().width / savedBackground.columns) *
        BAR_ROW_RATIO
      if (!pitch) return
      const rows = Math.ceil(
        (content.getBoundingClientRect().height + 2) / pitch,
      )
      // Geometry is measured once per resize; animation never changes layout.
      panel.style.setProperty("--login-panel-height", `${rows * pitch}px`)
    }
    align()
    let frame = 0
    const resize = new ResizeObserver(() => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(align)
    })
    resize.observe(content)
    return () => {
      cancelAnimationFrame(frame)
      resize.disconnect()
    }
  }, [])
  return (
    <div className="login-page">
      <LoginBackground
        panelRef={panelRef}
        config={savedBackground}
        paused={paused}
      />
      <header className="login-header">
        <div className="flex items-center gap-3">
          <img
            src={`${import.meta.env.BASE_URL}favicon.svg`}
            alt=""
            className="h-6 w-auto"
          />
          <span className="text-title">Otari</span>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            isIconOnly
            aria-label={
              paused
                ? "Play background animation"
                : "Pause background animation"
            }
            onPress={() => setPaused(!paused)}
            className="motion-reduce:hidden"
          >
            {paused ? <FiPlay aria-hidden /> : <FiPause aria-hidden />}
          </Button>
          <Button
            variant="ghost"
            isIconOnly
            aria-label={
              resolved === "dark" ? "Use light theme" : "Use dark theme"
            }
            onPress={() =>
              setPreference(resolved === "dark" ? "light" : "dark")
            }
          >
            {resolved === "dark" ? (
              <FiSun aria-hidden />
            ) : (
              <FiMoon aria-hidden />
            )}
          </Button>
        </div>
      </header>
      <main className="login-main">
        <div ref={panelRef} className="login-panel">
          <span aria-hidden className="login-corner login-corner-start" />
          <span aria-hidden className="login-corner login-corner-end" />
          <div ref={contentRef} className="login-panel-content">
            {children}
          </div>
        </div>
      </main>
      <footer className="login-footer">
        <span>One gateway. Every model.</span>
        <span className="font-mono text-xs">Mozilla AI</span>
      </footer>
    </div>
  )
}
