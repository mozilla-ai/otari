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
Is it a frame that is neither a form nor a question?
 └── Dialog                 (a guided step, a receipt, a thing to read and copy)
Is the product waiting on something to arrive?
 └── ScanBorder             (around the panel that is doing the waiting)
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
  submitLabel, onSubmit, isPending, error?, isDirty?, isDismissable = true,
  isSubmitDisabled?, returnFocusRef?, footerStart?, tabs?, children }
Dialog: { isOpen, onOpenChange, title, description?, size = "md",
  mark?, isAnnouncement?, isDismissable = true, status?, footerStart?,
  actions?, children }
DialogSection: { className?, children }
ScanBorder: { isActive, tone = "accent" | "danger", className?, children }
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

## Dialog

The third dialog, and the one to reach for when the other two would be a lie
about what the frame does. `FormDialog` is a place to work and owns a submit;
`ConfirmDialog` is an `AlertDialog` that interrupts to ask one question. This is
neither: a header, a scrolling body, an optional footer, and a body that is
whatever is being presented.

It shares `FormDialog`'s geometry class family in `globals.css`, so both sit at
the same height, cap at the same viewport budget and become the same full-screen
sheet below 640px in either dimension, and it takes the same three widths.

**Its body has no padding, and that is the component's one structural rule.**
Every band is a `DialogSection`, which carries its own padding and the hairline
above it, so the divisions run edge to edge the way a page's do. A frame
presenting three things divides them; it does not float three cards in a padded
column, which is the thing this system does not do anywhere.

`status` is the one slot outside the scrolling body: a readout pinned above the
footer on its own tinted band, for the thing the frame is actually about rather
than more of what it is presenting. A frame with a long body and a live status
would otherwise push that status below the fold at the moment it starts
changing, which is what happened to the first-run sheet's listening panel. The
tint is what separates it without a second border, and it is the only fill in
the frame, so what goes in it draws no surface of its own.

`mark` is a glyph at the head of the title row, beside the heading rather than
centered above it: a frame reporting that something worked is still a frame, and
centering one screen of a flow that is otherwise left-aligned reads as a
different product. `isAnnouncement` takes `text-display-sub` instead of the
section head, which the type scale reserves for one thing per page, "a
get-started strip, a first-run panel".

```tsx
// Correct: a frame presenting something, with one way out
<Dialog
  isOpen={isOpen}
  onOpenChange={setIsOpen}
  size="lg"
  title="Send your first request"
  description="It lands in Default workspace."
  status={<ListeningPanel />}
  footerStart={<p className="text-caption">Usage stays empty until one lands.</p>}
  actions={<Button onPress={skip}>Skip</Button>}
>
  <DialogSection>
    <CopyField label="Your API key" value={key} concealed={CONCEALED_SECRET} />
  </DialogSection>
  <DialogSection>
    <CodeBlock label="curl" value={snippet} arrangement="bare" isBounded />
  </DialogSection>
</Dialog>

// Incorrect: a form belongs in FormDialog, which owns the submit and its
// pending state rather than leaving both to the caller. A bare child is also a
// band with no padding and no rule: every child is a DialogSection.
<Dialog title="New key" actions={<Button onPress={create}>Create</Button>}>
  <Field label="Key name" value={name} onChange={setName} />
</Dialog>
```

## ScanBorder

The one piece of decorative motion in this system, and it earns its place by
being literally true: an arc travels the band's edge **only while the product is
watching for something that has not arrived**, and stops when it has. Anywhere
else, motion on an edge is noise.

It is a masked conic gradient on an `::after`, with the angle animated through
an `@property`, so there is no dependency behind it and at radius 0 there is no
corner to get wrong. `tone` picks the arc's ink through a variable, which is how
a failure turns the sweep red without the stylesheet knowing what a failure is.

Under `prefers-reduced-motion` the arc holds still rather than disappearing: the
band should still read as the thing on the page that is waiting.

```tsx
// Correct: the wait is real, and the tone reports the last attempt
<ScanBorder isActive={!checkFailed} tone={failure ? "danger" : "accent"}>
  <ListeningRow />
</ScanBorder>

// Incorrect: nothing is being awaited, so the motion says nothing
<ScanBorder isActive>
  <KpiStrip />
</ScanBorder>
```

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
  isDirty={isDirty}
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

**An empty state or first-run panel whose action opens the dialog stays mounted
while the dialog is open; it is the node focus returns to.** Hiding it while the
form is up was right when the form was a band on the page and takes away the
only thing focus can go back to now that it is a dialog over one.

**A page whose empty state disappears after the first create passes
`returnFocusRef` to the control that survives.** React Aria restores focus to
whatever opened the dialog, and that node is gone when creating the first row is
what emptied the empty state; focus falls to `<body>` and the next Tab starts at
the top of the document. The check runs when the frame is actually gone, not when
it was asked to close: the overlay subtree lives through its exit animation, so a
`requestAnimationFrame` at close time finds the dialog still holding focus.

**`isDirty` arms a guard in the footer, not a second dialog.** Escape and a
click outside swap the actions for "Unsaved changes · Keep editing · Discard".
A dialog never opens a dialog.

**A draft is fresh on every open and untouched through the exit.** Reset on the
way in, never on the way out. The frame keeps its content while it animates out,
so clearing state when `isOpen` goes false blanks the body for the length of the
exit, and a success step blanks to an empty form in front of the operator.

**The component that renders the `FormDialog` owns everything that resets
between opens: the draft *and* its mutation**, meaning the hook that yields
`isPending` and `error`. Both live below the key. The page owns only `isOpen`,
the open counter that keys the mount, and the list the mutation refreshes.

```tsx
// Correct: the mutation is inside the component the key remounts
function CreateKeyDialog({ isOpen, onOpenChange }: …) {
  const create = useCreateKey()
  const [keyName, setKeyName] = useState("")
  …
}

// and the page holds only what does not reset
const [isOpen, setIsOpen] = useState(false)
const [openCount, setOpenCount] = useState(0)
…
<Button onPress={() => { setOpenCount((n) => n + 1); setIsOpen(true) }}>
  Create key
</Button>
<CreateKeyDialog key={openCount} isOpen={isOpen} onOpenChange={setIsOpen} />

// Incorrect: the mutation sits above the key, so the key cannot reset it
const createBudget = useCreateBudget()          // page level
…
<BudgetForm key={openCount} error={createBudget.error} … />
```

A mutation above the key is the shape to watch for, because the draft looks
right and the error does not: the key remounts the fields, the mutation keeps
its state, and a failed create's banner survives into a fresh form. `close()`
only sets `isOpen` false; a `requestAnimationFrame` does not cover the exit,
which is an animation rather than a frame.

**`isDirty` comes from `useDirtySnapshot`**: hand it every field the form owns
and it snapshots them on mount, so dirty means "differs from what was seeded".

```tsx
// Correct: the whole draft, and the hook holds the seed
const { isDirty } = useDirtySnapshot({
  name,
  target,
  chain,
  conditions,
  guardrails,
})
…
<FormDialog isDirty={isDirty} … />

// Incorrect: a predicate of empties, which reports an edit form dirty on arrival
const isPristine = name === "" && target === "" && chain.length === 0
```

A field whose default arrives after mount (the first budget in a list, a
workspace roster) is part of the seed, not a change: seed the state from the
resolved value, or call the hook's `reset` when it lands. The hook cannot tell
that default from a keystroke, and deliberately does not try.

Dirty means "differs from what was seeded", and a create form is the case where
the seed happens to be all empties. So a hand-listed predicate of empties is the
create-only degenerate form: it is right until the same component edits
something, and it drifts, because a field added to the form has to be remembered
in a second place. The hook cannot omit a field, because it is handed the draft
rather than a list of the fields to compare.

A guard that lies lets Escape discard work it cannot see. That has happened here
in both directions: a reopened dialog dirty on arrival, and a half-filled one
discarded without a word.

The remount is also what keeps one row's draft out of the next row's dialog,
which is the promise this component's own docstring makes; a page that holds the
draft above the dialog defeats it.

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
