import { Dot } from "@/design-system/indicators/Dot"
import { type KeyState, STATE_WORDS } from "@/features/keys/keyList"

const TONES: Record<KeyState, { dot: string; ink: string }> = {
  active: { dot: "bg-success", ink: "text-muted" },
  expired: { dot: "bg-danger", ink: "text-muted" },
  disabled: { dot: "bg-text-subtle", ink: "text-subtle" },
}

/**
 * A key's state: a square dot and the word, never a pill. Disabled takes the
 * subtle rung because somebody chose it; Expired takes the danger dot because
 * the key stopped working on its own.
 */
export function KeyStateMark({ state }: { state: KeyState }) {
  const { dot, ink } = TONES[state]
  return (
    <span
      className={`flex items-center gap-2 whitespace-nowrap text-mono-caption ${ink}`}
    >
      <Dot className={dot} />
      {STATE_WORDS[state]}
    </span>
  )
}
