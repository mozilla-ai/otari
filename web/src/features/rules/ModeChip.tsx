import { Chip, type ChipTone } from "@/design-system/indicators/Chip"
import type { RuleMode } from "@/features/rules/rules"

const TONE: Record<RuleMode, ChipTone> = {
  Enforce: "accent",
  Watch: "neutral",
  Off: "neutral",
}

export function ModeChip({ mode }: { mode: RuleMode }) {
  return <Chip tone={TONE[mode]}>{mode}</Chip>
}
