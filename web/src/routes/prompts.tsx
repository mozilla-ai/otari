import { createFileRoute } from "@tanstack/react-router"
import { PromptsPage } from "@/features/prompts/PromptsPage"

export const Route = createFileRoute("/prompts")({
  component: PromptsPage,
})
