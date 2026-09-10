import { createFileRoute } from "@tanstack/react-router"

import { UnavailableHere } from "@/shared/components/access/UnavailableHere"

// Declared so the organization rail can name the destination the navigation
// design draws, and gated on a surface this gateway does not report, so the link
// is absent here and present on a deployment that serves it. The shell answers a
// gated-off route with its own panel, so this component is only reached by a
// deployment that reports the surface without composing the overlay behind it.
export const Route = createFileRoute("/organization/guardrails")({
  component: () => <UnavailableHere title="Guardrails" />,
})
