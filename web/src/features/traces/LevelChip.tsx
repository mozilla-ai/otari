import { Chip } from "@/design-system/indicators/Chip"

import type { Observation } from "./types"

// DEFAULT is the normal case, so only a warning or an error earns a chip.
export function LevelChip({ level }: { level: Observation["level"] }) {
  if (level === "WARNING") return <Chip tone="warning">Warning</Chip>
  if (level === "ERROR") return <Chip tone="danger">Error</Chip>
  return null
}
