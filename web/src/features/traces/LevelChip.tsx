import { Chip } from "@/design-system/indicators/Chip"

import type { Observation } from "./types"

// DEFAULT is the normal case, so only a warning or an error earns a chip. A
// tool's error result is the agent's to read, so it is marked but not alarming.
export function LevelChip({
  level,
  returnedError = false,
}: {
  level: Observation["level"]
  returnedError?: boolean
}) {
  if (level === "WARNING") return <Chip tone="warning">Warning</Chip>
  if (level === "ERROR") return <Chip tone="danger">Error</Chip>
  if (returnedError) return <Chip tone="neutral">Error result</Chip>
  return null
}
