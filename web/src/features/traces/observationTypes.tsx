import type { IconType } from "react-icons"
import {
  FiArrowRight,
  FiDisc,
  FiGitBranch,
  FiMessageSquare,
  FiTerminal,
  FiTool,
} from "react-icons/fi"

import type { ObservationType } from "./types"

export type TypeKey = ObservationType | "TRACE"

// The trace, generation and tool marks are the shell's own Sessions,
// Playground and Tools glyphs. The types that do work take chart slots 1 to 3
// in order; a span only groups them, so it stays neutral. Full class strings,
// so Tailwind sees every one.
export const OBSERVATION_TYPES: Record<
  TypeKey,
  { label: string; icon: IconType; inkClass: string; barClass: string | null }
> = {
  TRACE: {
    label: "Trace",
    icon: FiGitBranch,
    inkClass: "text-foreground",
    barClass: null,
  },
  GENERATION: {
    label: "Generation",
    icon: FiMessageSquare,
    inkClass: "text-chart-cat-1",
    barClass: "bg-[var(--trace-bar-generation)]",
  },
  SPAN: {
    label: "Span",
    icon: FiArrowRight,
    inkClass: "text-muted",
    barClass: "bg-[var(--trace-bar-span)]",
  },
  AGENT: {
    label: "Agent",
    icon: FiTerminal,
    inkClass: "text-chart-cat-3",
    barClass: "bg-[var(--trace-bar-agent)]",
  },
  TOOL: {
    label: "Tool",
    icon: FiTool,
    inkClass: "text-chart-cat-2",
    barClass: "bg-[var(--trace-bar-tool)]",
  },
  EVENT: {
    label: "Event",
    icon: FiDisc,
    inkClass: "text-subtle",
    barClass: null,
  },
}

export function TypeMark({
  type,
  className = "size-4",
}: {
  type: TypeKey
  className?: string
}) {
  const { icon: Icon, inkClass } = OBSERVATION_TYPES[type]
  return <Icon aria-hidden="true" className={`${className} ${inkClass}`} />
}
