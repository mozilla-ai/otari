import { createFileRoute } from "@tanstack/react-router"

import { AccountPage } from "@/features/account/AccountPage"

export const Route = createFileRoute("/account")({
  component: AccountPage,
})
