import { useMutation } from "@tanstack/react-query"
import type { FeedbackSubmission } from "@/client"
import { apiFetch } from "@/shared/api/client"

export function useSendFeedback() {
  return useMutation({
    mutationFn: (body: FeedbackSubmission) =>
      apiFetch<void>("/feedback", {
        method: "POST",
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(15_000),
      }),
    retry: false,
    // An offline press must fail now, never queue a later delivery.
    networkMode: "always",
    gcTime: 0,
  })
}
