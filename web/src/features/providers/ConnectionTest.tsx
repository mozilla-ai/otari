import { useEffect, useRef } from "react"

import { Button } from "@/design-system/actions/Button"
import { errorMessage } from "@/design-system/feedback/errorMessage"

// Testing the form's credentials before they are saved, in two nodes because
// they belong in two places: the button sits in the footer beside the submit,
// its outcome at the end of the body. The unverified case below is four lines
// plus the provider's own reply, and a footer that grows shoves the form up
// under the operator's hands; in the body it scrolls with everything else.
//
// Shared by the deployment providers form and the hosted providers forms, whose
// test routes answer in the same shape and take different bodies.

/** What a connection test answers. */
export interface ConnectionTestOutcome {
  ok: boolean
  model_count: number
  error?: string | null
  discovery_unsupported: boolean
}

/** The test mutation, as either page's hook returns it. */
export interface ConnectionTestState<Payload> {
  isPending: boolean
  data: ConnectionTestOutcome | undefined
  error: unknown
  mutate: (payload: Payload) => void
}

// `getPayload` returns null when the minimum fields for a test are not filled
// in yet, which disables the button.
export function ConnectionTestButton<Payload>({
  test,
  getPayload,
}: {
  test: ConnectionTestState<Payload>
  getPayload: () => Payload | null
}) {
  const payload = getPayload()
  return (
    <Button
      variant="ghost"
      isDisabled={payload === null || test.isPending}
      onPress={() => {
        if (payload) test.mutate(payload)
      }}
    >
      {test.isPending ? "Testing…" : "Test connection"}
    </Button>
  )
}

export function ConnectionTestResult<Payload>({
  test,
}: {
  test: ConnectionTestState<Payload>
}) {
  const answered = test.data ?? test.error
  const ref = useRef<HTMLSpanElement | null>(null)
  // The body scrolls, and this is its last child: on an `lg` dialog in an 800px
  // window the custom tab's own fields already fill it, so a verdict rendered
  // here can land below the fold with the footer button back to "Test
  // connection" and nothing else, to a sighted operator, having happened.
  useEffect(() => {
    if (answered) ref.current?.scrollIntoView({ block: "nearest" })
  }, [answered])
  return (
    // No wrapper, and `empty:hidden` rather than a conditional render: the
    // parent's `gap-4` would otherwise reserve a row before a test is ever run,
    // and the live region has to exist before it has something to say.
    <span
      ref={ref}
      role="status"
      aria-live="polite"
      className="flex flex-col gap-1.5 empty:hidden"
    >
      {test.isPending ? null : test.error ? (
        <span className="text-xs text-danger">{errorMessage(test.error)}</span>
      ) : test.data ? (
        test.data.ok ? (
          <span className="text-xs font-medium text-success">
            Connected. {test.data.model_count} model
            {test.data.model_count === 1 ? "" : "s"} available.
          </span>
        ) : test.data.discovery_unsupported ? (
          // No /v1/models on this backend: the test cannot confirm the key, but
          // it is not evidence the key is wrong either (issue #447). The error is
          // kept because this is the form where the operator just typed api_base,
          // and a wrong one 404s exactly like an absent listing endpoint.
          <span className="block max-w-md break-words text-xs text-warning">
            This provider does not list models, so the key could not be verified
            here. Save it and use the provider; declare its model ids under{" "}
            <code>models:</code> to have them show up in the catalog. If you did
            not expect this, check the provider's reply below.
            {test.data.error ? (
              <span className="mt-0.5 block text-muted">{test.data.error}</span>
            ) : null}
          </span>
        ) : (
          <span className="block max-w-md break-words text-caption text-danger">
            {test.data.error ?? "Connection failed."}
          </span>
        )
      ) : null}
    </span>
  )
}
