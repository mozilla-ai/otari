import { useEffect, useRef } from "react"
import { FiCheck } from "react-icons/fi"
import { fireSetupConfetti } from "@/features/onboarding/setupConfetti"

/** Mounted inside the portal so the dialog exists before the effect runs. */
export function SetupSuccessMark() {
  const markRef = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    const dialog = markRef.current?.closest('[role="dialog"]')
    const controller = new AbortController()
    if (dialog) void fireSetupConfetti(dialog, controller.signal)
    return () => controller.abort()
  }, [])
  return (
    <span ref={markRef} className="mt-0.5 flex shrink-0">
      <FiCheck aria-hidden className="text-success size-6" />
    </span>
  )
}
