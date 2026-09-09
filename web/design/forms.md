# Forms and controls

A field is white with a real 1px edge (`--field-border`, 0.48 alpha). On a white
page a fill alone leaves a field indistinguishable from what it sits on, so the
border is what makes it an object. HeroUI defaults `--field-border-width` to 0;
`globals.css` sets it to 1.

## Which control?

```
Is it free text, a number, or a date?
 ├── Is the value a secret (a provider key, a password)?
 │    └── Yes -> SecretField        (masked, revealable, copyable, autofill off)
 └── No -> Field
Is it a boolean?
 ├── Does it take effect on its own, without a Save?
 │    └── Yes -> Toggle             (role="switch", 44x24 track)
 └── No -> Checkbox                 (part of a form the operator submits)
Is it one of a short, closed set?
 ├── Does the choice filter a list on this page?
 │    └── Yes -> FilterSelect, or Segmented if there are 4 or fewer
 └── No -> Select
Is it many of a set?
 └── FilterMultiComboBox
```

## Signatures

```ts
Field: { label, value, onChange: (next: string) => void, placeholder?, type = "text",
  isRequired?, isDisabled?, isInvalid?, errorMessage?, description?, autoFocus?,
  reserveMessage? }
SecretField: { label, value, onChange, placeholder?, description?, reserveMessage? }
Toggle: { label, checked, onChange: (next: boolean) => void, disabled? }
Checkbox: { isSelected, onChange: (next: boolean) => void, isDisabled?, ariaLabel?,
  children }
FilterSelect: { value, onChange: (next: string) => void, options: { value, label }[],
  label?, ariaLabel?, id?, disabled? }
```

`FilterSelect` renders `label` as visible text beside the control, which is what a
filter in a toolbar wants; `ariaLabel` is for the case where the visible label would
repeat what is already on screen. Pass one or the other, never neither.

`onChange` takes the **value**, never an event. `reserveMessage` defaults to off, so
a field in a form has to opt in; a field in a table row or a toolbar leaves it off.
`Toggle` says `disabled`, the HeroUI-backed controls say `isDisabled`: that is a
real inconsistency in the tree, not a typo here.

## Field

```tsx
// Correct
<Field
  label="Key name"
  value={name}
  onChange={setName}
  placeholder="checkout-service"
  description="Lowercase, hyphens, no spaces."
  isInvalid={Boolean(error)}
  errorMessage={error}
  reserveMessage
/>

// Incorrect: a placeholder is an example, never a label. With the label gone the
// field has no accessible name and no name at all once the operator types.
<Input placeholder="Key name" value={name} onChange={setName} />
```

Rules:

- **The label is always visible and always associated.** `Field` wires it through
  HeroUI's `Label`, so never hand-roll a `<span>` above an `<input>`.
- Never add a manual `*` for `isRequired`. HeroUI marks it through CSS and you get
  two.
- `reserveMessage` holds one caption line open so an error does not move the form.
  On for a field in a form, off for one in a table row or a toolbar.
- An error message **replaces** the description line rather than adding a row.
- Inputs on public pages are 16px (`text-base`), which is what stops iOS from
  zooming on focus.

## Validation timing

Validate on submit and on blur. Never on the first keystroke: a message that
appears while someone is halfway through typing their own key name is telling them
they are wrong before they have finished being right.

On an autosaving page (see [layout.md](layout.md)) blur is also the commit, so
the two coincide: a text field validates and saves when it is left or when Enter
is pressed, and only if the value changed. A select saves on change, having
nothing typed to lose. Success is silent; a refused save keeps the value that
caused it, marks the control `aria-invalid`, and puts the message in the row
through `SettingRow`'s `error`. `useAutosave` owns that state, one instance per
control, and its `isSaving` is what disables the control mid-write.

## Toggle

`Toggle` is `role="switch"` with an `aria-label`. The visible track is 44x24 with a
1px edge, filled with the page ground rather than a surface step: on a flat plane
the track is a drawn outline, not a raised trough, so the state is carried entirely
by the knob's color (`control-thumb` off, `control-indicator` on).

The knob travels with `transition-transform`, not `transition-colors`: the fill
should read as instant, the travel is what benefits from being followed.

## Checkbox

One visual serves both the standalone checkbox and a `DataTable`'s selection box
(`CheckboxVisual`), so the two cannot drift apart. Built on react-aria rather than
HeroUI's own, which splits the control across subcomponents.

The box fills with `--color-control-indicator` and the mark is
`--color-accent-glyph` (white). That mark is a graphic at a 3:1 floor, not text at
4.5:1, which is why it is white where text on the accent is near-black.

Pass `ariaLabel` when the visible label repeats across the page (one workspace list
per guardrail, say). Keep the visible text inside it so speech input still reaches
the control.

## Buttons in a form

One `primary` at the foot of the group it saves. See [actions.md](actions.md).
Never a floating page-level Save; see [layout.md](layout.md).
