# Feedback

A banner is a band across the page, not a card floating on it. Every message names
what happened, what the system did about it, and the one control that fixes it.

## Which one?

```text
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
Is the operator creating or editing an object?
 └── FormDialog             (every one of them; see the placement rule in actions.md)
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
FormDialog: { isOpen, onOpenChange, title, description?, size = "md",
  submitLabel, onSubmit, isPending, error?, isDirty?, footerStart?, tabs?,
  children }
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

## FormDialog

Every create and every edit opens here. Not a panel that appears under the
table, not a section appended to the page: one surface, so an operator who has
created a key knows what adding a provider will do.

Built on HeroUI's `Modal` rather than `AlertDialog`, and that is the difference
between the two dialogs rather than a detail of them. An alert interrupts to ask
one question; a form is a place to work.

```tsx
// Correct: the title names the object, the submit repeats the trigger
<FormDialog
  isOpen={isCreating}
  onOpenChange={setCreating}
  title="New key"
  description="The secret is shown once, right after you create it."
  submitLabel="Create key"
  onSubmit={submit}
  isPending={create.isPending}
  error={create.error}
  isDirty={name !== ""}
>
  <Field
    label="Key name"
    value={name}
    onChange={setName}
    description="Lowercase, hyphens, no spaces."
    reserveMessage
  />
</FormDialog>

// Incorrect: the title restates the button, and nothing says what was made
<FormDialog title="Create key" submitLabel="Save" …>
```

**Sizes.** `sm` 440 for one or two fields, `md` 520 by default, `lg` 640 for
tabs or six fields and up. Below a 640px viewport every size is a full-screen
sheet. Those widths cannot be spelled as a class: `globals.css` pins
`.modal__dialog` unlayered, which outranks `@layer utilities` and puts a 448px
floor under it, so the component sets `--form-dialog-width` inline and the
geometry is settled beside the rule it has to beat.

**A field reserves its message line only where it has a description**, which is
[forms.md](forms.md)'s rule and not a dialog rule: the reserved line exists so an
error can replace a description rather than push the footer down, so a field with
nothing to say under it holds nothing. Spell that as `reserveMessage={false}`
rather than by leaving the prop off, which currently reserves anyway: forms.md
says the prop defaults to off and it does not, because `FieldMessages` defaults
its own `reserve` to true and the four controls forward an undefined prop into
it. Measured, a bare field in a dialog is 83px against the 60px it should be.
The first field takes `autoFocus`.

**Fields fill the dialog.** `Field` and `SecretField` cap themselves at 448px,
which is right on a page and wrong in a 640px dialog; `globals.css` lifts the cap
for this place, so no call site sets a width.

**The footer's height never changes, and the primary keeps its width.** The
spinner replaces the label in place rather than sitting beside it. The primary
is **not** disabled while it runs: disabled is one treatment at 0.4 opacity and
it has to read as denied, and a submit in flight is working rather than refused,
so it keeps its fill and blocks its own press. Cancel and the close control *are*
disabled, because they genuinely are refused until it lands.

**`isDirty` arms a guard in the footer, not a second dialog.** Escape and a
click outside swap the actions for "Unsaved changes · Keep editing · Discard".
A dialog never opens a dialog.

**The success step is not a prop.** When a mutation has something to hand back
(a key's secret), the caller swaps the children and the submit label to "Done"
and the frame stays where it was. `footerStart` is where "Create another" goes.

## Copy

Write what changed, in the operator's terms, with the object named.

| Instead of | Write |
| --- | --- |
| "An error occurred" | "Could not reach Anthropic" |
| "Invalid input" | "No spaces. Use a hyphen." |
| "Are you sure?" | "Remove permanently" |
| "Success!" | Nothing. The row changed; that is the confirmation. |
