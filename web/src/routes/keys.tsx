import { createFileRoute } from "@tanstack/react-router"

import { KeysPage } from "@/features/keys/KeysPage"

export const Route = createFileRoute("/keys")({
  component: KeysPage,
})
