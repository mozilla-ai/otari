import { createFileRoute } from "@tanstack/react-router"

import { RouterSavingsPage } from "@/features/routerSavings/RouterSavingsPage"

export const Route = createFileRoute("/_overview/router-savings")({
  component: RouterSavingsPage,
})
