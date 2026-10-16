import type { IconType } from "react-icons"
import {
  FiArrowRight,
  FiGitBranch,
  FiMessageSquare,
  FiServer,
  FiShield,
  FiTerminal,
  FiTool,
} from "react-icons/fi"

import type { ObservationType } from "./types"

export type TypeKey = ObservationType | "TRACE"

// The trace, generation, tool, guardrail and MCP marks are the shell's own
// Sessions, Playground, Tools, Guardrails and MCP servers glyphs. The types
// that do work take chart slots 1 to 5 in order; a span or the session only
// groups them, so it stays neutral. Full class strings, so Tailwind sees every one.
export const OBSERVATION_TYPES: Record<
  TypeKey,
  { label: string; icon: IconType; inkClass: string; barClass: string }
> = {
  TRACE: {
    label: "Trace",
    icon: FiGitBranch,
    inkClass: "text-foreground",
    barClass: "bg-trace-bar-span",
  },
  GENERATION: {
    label: "Generation",
    icon: FiMessageSquare,
    inkClass: "text-chart-cat-1",
    barClass: "bg-trace-bar-generation",
  },
  SPAN: {
    label: "Span",
    icon: FiArrowRight,
    inkClass: "text-muted",
    barClass: "bg-trace-bar-span",
  },
  AGENT: {
    label: "Agent",
    icon: FiTerminal,
    inkClass: "text-chart-cat-3",
    barClass: "bg-trace-bar-agent",
  },
  TOOL: {
    label: "Tool",
    icon: FiTool,
    inkClass: "text-chart-cat-2",
    barClass: "bg-trace-bar-tool",
  },
  GUARDRAIL: {
    label: "Guardrail",
    icon: FiShield,
    inkClass: "text-chart-cat-4",
    barClass: "bg-trace-bar-guardrail",
  },
  MCP: {
    label: "MCP",
    icon: FiServer,
    inkClass: "text-chart-cat-5",
    barClass: "bg-trace-bar-mcp",
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
