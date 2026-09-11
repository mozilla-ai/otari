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
           └── Yes -> FormDialog   (a place to work: creating or editing)
                or ConfirmDialog   (one question: are you sure)
                or Dialog          (neither: a guided step, a receipt)
```

Three, not five: the tree names three dialogs and they are one answer between
them. A dialog is the answer to "does it want the whole screen's attention",
and which of the three is a second question, asked in the section below.

The dialogs live in [feedback.md](feedback.md)'s directory rather than this one,
because a dialog is nearly always feedback about an action; they are listed here
because they are the third answer to the same question.

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

It takes its trigger as `children` rather than as a prop, so the trigger keeps
its own type: an `IconButton` inside one is still an `IconButton`, with its
required label and its 44px box intact.

**Which of the two forms you want depends on whether the trigger takes DOM
props.** HeroUI's own trigger is a `div` the library gives `role="button"` and
`tabIndex=0`, which is what makes a non-interactive trigger reachable at all and
is wrong around a real `<button>`: it announces "Delete button" inside "Delete
button", takes a tab stop of its own, and gives every `getByRole("button")` two
matches instead of one. Hand the function form a native control instead, and it
spreads the props it is given onto that control, so the trigger and the control
are one element. `RowAction`'s glyph is the worked example.

A HeroUI `Button` (so `IconButton` too) is the exception and keeps the wrapper:
its props are react-aria's rather than the DOM's, so the trigger's `onClick` and
`onPointerEnter` do not typecheck against it, let alone reach an element. Its 44px
box and required label are unharmed by the wrapper; the duplicate name is the
cost, and #2123 is where that is written down rather than solved.

```tsx
// Correct: a native control takes the trigger's props rather than a wrapper
<Tooltip content="Delete">
  {(props) => (
    <button {...props} type="button" aria-label="Delete">
      <FiTrash2 aria-hidden className="h-3.5 w-3.5" />
    </button>
  )}
</Tooltip>

// Correct: a node that is not a control is wrapped, which is what gives it the
// keyboard reach it has none of on its own
<Tooltip content="2026-09-11 14:02:11 UTC">
  <span className="text-caption">6m ago</span>
</Tooltip>

// Incorrect: the reason a control is refused has to be reachable without a
// pointer, and a disabled control takes no focus, so this reaches nobody who
// needs it. Put it in the accessible name (see RowAction's ariaLabel), which is
// what `RowAction` does for its own glyphs, with a native `title` beside it for
// the pointer the tooltip cannot reach.
<Tooltip content="An organization owner manages this key">
  <RowAction isDisabled onPress={revoke}>Revoke</RowAction>
</Tooltip>
```

## Popover

Anchored to its trigger, takes focus, and is dismissed deliberately. Right for a
column picker, a small confirm about one row, a panel of detail about the thing
that opened it.

Uncontrolled by default, which is the opposite of the dialogs and deliberate: a
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

## The three dialogs

Modal. Three of them, and the question sorts them:

- **`FormDialog`** when the operator is creating or editing an object. Every
  create flow in the product, no exceptions. See [feedback.md](feedback.md).
- **`ConfirmDialog`** when the dialog's whole job is "are you sure", which
  includes every delete of a record.
- **`Dialog`** when it is neither: a guided step, a receipt, a thing to read and
  copy. It owns no submit and asks no question, which is exactly why the other
  two would be a lie about what the frame does.

`Dialog` is the plain frame and the newest of the three, and the one thing to
know about reaching for it is that "this is not a form" is not sufficient
reason: a create flow whose submit you would hand-roll in the footer still wants
`FormDialog`, which owns the submit, its pending state and its error surface.

There was a fourth, a bare `AlertDialog` shell the form pattern was built from.
It is gone: a form wants `FormDialog`, which is a `Modal`, because an alert
interrupts to ask one question and a form is a place to work, and once nothing
hand-rolled a form dialog the shell had no call sites left.

Both are controlled only, because a dialog opens from something elsewhere
on the page (a row's Edit, a heading row's Create) rather than from a trigger
inside itself. Both mount their body only while open, which is not an
optimization: the body of a form dialog holds controlled inputs, and leaving
them mounted carries one row's draft into the next row's dialog.

`isDismissable` is on by default. Turning it off takes away Escape and the
outside click, and the only honest reason is unsaved work that would be lost;
even then the better fix is usually to keep the dismiss and confirm the discard.
