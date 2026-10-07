import type { IconType } from "react-icons"
import {
  LuBot,
  LuCircleDot,
  LuFan,
  LuListTree,
  LuMoveHorizontal,
  LuWrench,
} from "react-icons/lu"

import type { ObservationType } from "./types"

export type TypeKey = ObservationType | "TRACE"

// Lucide glyphs for these six marks only: they are Langfuse's identity for an
// observation type. Full class strings, so Tailwind sees every one.
export const OBSERVATION_TYPES: Record<
  TypeKey,
  { label: string; icon: IconType; inkClass: string; barClass: string | null }
> = {
  TRACE: {
    label: "Trace",
    icon: LuListTree,
    inkClass: "text-foreground",
    barClass: null,
  },
  GENERATION: {
    label: "Generation",
    icon: LuFan,
    inkClass: "text-chart-cat-5",
    barClass: "bg-[var(--trace-bar-generation)]",
  },
  SPAN: {
    label: "Span",
    icon: LuMoveHorizontal,
    inkClass: "text-chart-cat-6",
    barClass: "bg-[var(--trace-bar-span)]",
  },
  AGENT: {
    label: "Agent",
    icon: LuBot,
    inkClass: "text-chart-cat-3",
    barClass: "bg-[var(--trace-bar-agent)]",
  },
  TOOL: {
    label: "Tool",
    icon: LuWrench,
    inkClass: "text-chart-cat-2",
    barClass: "bg-[var(--trace-bar-tool)]",
  },
  EVENT: {
    label: "Event",
    icon: LuCircleDot,
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
