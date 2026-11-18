import { createFileRoute } from "@tanstack/react-router"

import { PersonPage } from "@/features/people/PersonPage"

function SelectedPerson() {
  const { personId } = Route.useParams()
  return <PersonPage personId={personId} />
}

export const Route = createFileRoute("/people/$personId")({
  component: SelectedPerson,
})
