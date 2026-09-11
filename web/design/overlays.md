# Overlays

Three components put something on top of the page, and picking the wrong one is
the most common mistake here, so start with the question rather than the list.

```text
Does the operator need to interact with what appears?
 ├── No, it is a label for the thing they are pointing at
 │    └── Tooltip                  (hover/focus, no focus of its own)
 └── Yes
      ├── Is it about the control that opened it?
      │    └── Yes -> Popover      (anchored, takes focus, not modal)
      └── Does it want the whole screen's attention?
           └── Yes -> Dialog       (modal, dismissed deliberately)
```

`Dialog` lives in [feedback.md](feedback.md)'s directory rather than this one,
because a dialog is nearly always feedback about an action; it is listed here
because it is the third answer to the same question.

## Tooltip

**A tooltip is never the only channel.** It does not exist on a phone, so
anything only it says is unsaid for a touch operator. That is the rule from
[motion-and-access.md](motion-and-access.md) applied at its hardest, and it
leaves exactly two honest uses:

- **A repetition.** Spelling out an icon-only control that already carries an
  `aria-label`, so a pointer user and a screen reader get the same sentence.
- **Precision beside a rounded value.** An exact timestamp next to "6m ago".

Anything an operator needs in order to *act* goes on the page. A description
under a field, a caption under a value, or a `Badge` on the row: all three are
readable on a phone and none of them needs a pointer.

It wraps its trigger rather than taking one as a prop, so the trigger keeps its
own type: an `IconButton` inside one is still an `IconButton`, with its required
label and its 44px box intact.

```tsx
// Correct: the tooltip repeats the button's own accessible name
<Tooltip content="Delete this key">
  <IconButton label="Delete this key" variant="danger">
    <FiTrash2 aria-hidden className="size-4" />
  </IconButton>
</Tooltip>

// Incorrect: the reason a control is refused has to be reachable without a
// pointer, and a disabled control takes no focus, so this reaches nobody who
// needs it. Put it in the accessible name (see RowAction's ariaLabel).
<Tooltip content="An organization owner manages this key">
  <RowAction isDisabled onPress={revoke}>Revoke</RowAction>
</Tooltip>
```

## Popover

Anchored to its trigger, takes focus, and is dismissed deliberately. Right for a
column picker, a small confirm about one row, a panel of detail about the thing
that opened it.

Uncontrolled by default, which is the opposite of `Dialog` and deliberate: a
popover's trigger is inside it, so it can own that state. Pass `isOpen` and
`onOpenChange` for the case where something else has to close it, such as a
route change or a mutation landing.

Its content is wrapped in HeroUI's popover dialog, which is what puts the panel
in the accessibility tree and traps focus while it is open. Without that it is a
div a keyboard operator tabs straight past, which is why the component does it
rather than leaving it to a call site.

**Dismiss it before reaching for the page behind it.** It holds focus while
open, which is also the thing that trips the Playwright suite; web/AGENTS.md
says the same about React Aria popovers under "Checks".

## Dialog

Modal. Three of them, and the question sorts them:

- **`FormDialog`** when the operator is creating or editing an object. Every
  create flow in the product, no exceptions. See [feedback.md](feedback.md).
- **`ConfirmDialog`** when the dialog's whole job is "are you sure", which
  includes every delete of a record.
- **`Dialog`** is the bare `AlertDialog` shell the other pattern was built from.
  It has no call sites; a form wants `FormDialog`, which is a `Modal`, because
  an alert interrupts to ask one question and a form is a place to work.

All three are controlled only, because a dialog opens from something elsewhere
on the page (a row's Edit, a heading row's Create) rather than from a trigger
inside itself. All three mount their body only while open, which is not an
optimization: the body of a form dialog holds controlled inputs, and leaving
them mounted carries one row's draft into the next row's dialog.

`isDismissable` is on by default. Turning it off takes away Escape and the
outside click, and the only honest reason is unsaved work that would be lost;
even then the better fix is usually to keep the dismiss and confirm the discard.
