import type { WorkspaceMcpServer } from "@/client"

/**
 * Who a server is used by, as one choice of three. The API keeps it as two
 * fields, `enabled` and `attach_by_default`, and this is the only place that
 * maps between them: an attached server that is disabled reaches nobody, so
 * the fourth combination reads as Off.
 */
export type McpAvailability = "off" | "on_request" | "every_request"

export const AVAILABILITY: Record<
  McpAvailability,
  {
    label: string
    badge: string
    dot: string
    /** The line under the choice in a list; Off needs none. */
    subline?: string
    /** The same thing as a sentence, for prose and the tooltip. */
    help: string
  }
> = {
  off: {
    label: "Off",
    badge: "OFF",
    dot: "bg-text-subtle",
    help: "Nobody can use it.",
  },
  on_request: {
    label: "On request",
    badge: "ON REQUEST",
    dot: "bg-success",
    subline: "Available to people building things, invisible to coding agents",
    help: "Available to people building things, invisible to coding agents.",
  },
  every_request: {
    label: "Every request",
    badge: "EVERY REQUEST",
    dot: "bg-accent",
    subline: "Available everywhere",
    help: "Available everywhere.",
  },
}

export const AVAILABILITY_ORDER: readonly McpAvailability[] = [
  "off",
  "on_request",
  "every_request",
]

export function availabilityOf(
  server: Pick<WorkspaceMcpServer, "enabled" | "attach_by_default">,
): McpAvailability {
  if (!server.enabled) return "off"
  return server.attach_by_default ? "every_request" : "on_request"
}

/** The two stored fields one choice sets. */
export function availabilityFields(availability: McpAvailability): {
  enabled: boolean
  attach_by_default: boolean
} {
  return {
    enabled: availability !== "off",
    attach_by_default: availability === "every_request",
  }
}

/** The three choices with the line under each, for the Availability select. */
export const AVAILABILITY_OPTIONS = AVAILABILITY_ORDER.map((value) => ({
  value,
  label: AVAILABILITY[value].label,
  description: AVAILABILITY[value].subline,
}))
