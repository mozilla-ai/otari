import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/organization/usage")({
  // Retired: Home and People answer what this page did. In `beforeLoad`, so a
  // bookmark lands on Home without mounting anything or leaving a history entry.
  beforeLoad: () => {
    throw redirect({ to: "/", replace: true })
  },
})
