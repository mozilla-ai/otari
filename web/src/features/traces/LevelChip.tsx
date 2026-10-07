import { Chip } from "@/design-system/indicators/Chip"

import type { Observation } from "./types"

// DEFAULT is not shown, as in Langfuse.
export function LevelChip({ level }: { level: Observation["level"] }) {
  if (level === "WARNING") return <Chip tone="warning">Warning</Chip>
  if (level === "ERROR") return <Chip tone="danger">Error</Chip>
  if (level === "DEBUG") return <Chip>Debug</Chip>
  return null
}
