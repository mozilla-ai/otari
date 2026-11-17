import { createFileRoute } from "@tanstack/react-router"

import { PeoplePage } from "@/features/people/PeoplePage"

export const Route = createFileRoute("/people/")({
  component: PeoplePage,
})
