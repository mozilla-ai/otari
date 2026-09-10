# Feedback

A banner is a band across the page, not a card floating on it. Every message names
what happened, what the system did about it, and the one control that fixes it.

## Which one?

```
Did a request fail?
 └── ErrorBanner            (it sanitizes the error; see below)
Is there a standing condition the operator should know about?
 ├── Is something wrong, or close to wrong?
 │    └── InfoBanner tone="warning"
 └── Is it merely informational (read-only, a mode)?
      └── InfoBanner tone="info"
Is someone waiting on a person to act?
 └── The attention band                (not a fifth status; see colors.md)
Is the destination empty?
 ├── Has the operator never created one of these?
 │    └── EmptyState        (says what the thing is, and offers the action)
 └── Did a filter or a range empty it?
      └── EmptyMessage      (one line, inside the section that already has a heading)
Is it loading?
 ├── The whole route -> PageLoading
 └── One section    -> the section's own isLoading, which keeps the heading
Did the page fail to render at all?
 └── PageError             (the catch boundaries' panel; see below)
Does the action delete a record?
 └── ConfirmDialog          (always, one row or many; see actions.md)
Is it destructive but deletes nothing (regenerate, archive, reset)?
 └── ConfirmButton          (the two-step confirm; see actions.md)
```

## Signatures

```ts
InfoBanner: { tone = "info" | "warning", children }
ErrorBanner: { error: unknown }
EmptyState: { title, description?, actionLabel?, onAction?, isActionDisabled?,
  children? }
EmptyMessage: { children, minHeight? }
PageLoading: { label = "Loading…" }
PageError: { error: unknown, children? }
ConfirmDialog: { isOpen, onOpenChange, heading, body, confirmLabel, onConfirm,
  confirmVariant = "danger", isPending?, error? }
ErrorBoundary: { children, resetKey? }
```

`PageError` is `PageLoading`'s counterpart, for a failure that took the whole page
rather than a band inside one: a gateway that never answered, and the two catch
boundaries below. Its `children` is the sentence about what to do next.

`ErrorBoundary` is not one a page reaches for: it is the catch above the router in
`App.tsx`, and the only one the pre-session screens have. Everything inside
`RouterProvider` is covered by TanStack Router's own catch boundary, which
`router.tsx` points at the same `PageError` so the two look like one product.

`EmptyState` takes `actionLabel` plus `onAction`, not a rendered button:
it owns the variant so no empty state can pick the wrong one. `ErrorBanner` takes the
caught value, typed `unknown`, and never a string.

## Banner anatomy

Mark, then what happened, then what the system did, then the one control.

```tsx
// Correct
<InfoBanner tone="warning">
  Data platform is at 92% of its ceiling. At the current rate the workspace stops
  serving requests in about two days.
</InfoBanner>

// Incorrect: a provider's raw message leaks internals and tells the operator
// nothing they can act on
<InfoBanner tone="warning">{error.response.data.detail}</InfoBanner>
```

`ErrorBanner` takes the caught value, not a string, and runs it through
`errorMessage()`. **Never render a provider's own message, a stack trace, or a raw
`detail` field.** The API boundary sanitizes; the banner explains. That is a
security property, not a style choice.

## EmptyState

Says what the thing *is*, not that the list is empty, and carries the same action
the toolbar does.

```tsx
// Correct
<EmptyState
  title="No keys yet"
  description="A key is how an application authenticates to the gateway. Create one to send your first request."
  actionLabel="Create key"
  onAction={openCreate}
/>

// Incorrect: tells the operator what they can already see, and offers no way out
<EmptyMessage>No data.</EmptyMessage>
```

`EmptyMessage` is the other case: inside a section that already has a heading, one
line is enough and it never gets its own illustration.

A disabled action in an empty state must carry its reason (Providers' "Add your
first provider" with no server secret key). A control shown disabled instead of
hidden is carrying that meaning on its own, at `opacity: 0.4`.

## ConfirmDialog

**Every delete of a record**, and any other destructive action that needs a
sentence of context or has to report an error in place. `confirmVariant` defaults
to `danger`. It owns `isPending` and `error` so the caller does not build a second
error surface inside a modal, which is also why the page's own `ErrorBanner` stops
carrying that mutation: reporting it in both puts the message the operator needs
behind the backdrop they are looking at.

`ConfirmButton`'s two-step is what is left, for a destructive action that deletes
nothing. See [actions.md](actions.md) for both.

## Copy

Write what changed, in the operator's terms, with the object named.

| Instead of | Write |
| --- | --- |
| "An error occurred" | "Could not reach Anthropic" |
| "Invalid input" | "No spaces. Use a hyphen." |
| "Are you sure?" | "Remove permanently" |
| "Success!" | Nothing. The row changed; that is the confirmation. |
