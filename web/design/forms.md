# Forms and controls

A field is white with a real 1px edge (`--field-border`, 0.48 alpha). On a white
page a fill alone leaves a field indistinguishable from what it sits on, so the
border is what makes it an object. HeroUI defaults `--field-border-width` to 0;
`globals.css` sets it to 1.

## Which control?

```
Is it free text, a number, or a date?
 ├── Is the value a secret (a provider key, a password)?
 │    ├── Collecting one -> SecretField   (masked, never prefilled, autofill off)
 │    └── Handing one out -> CopyField, `concealed` (see actions.md)
 ├── Does it need more than one line?
 │    └── Yes -> TextArea           (a floor rather than a height; it grows)
 ├── Does it filter a list as you type?
 │    └── Yes -> SearchField        (type="search": Escape clears, own history)
 └── No -> Field
Is it a boolean?
 ├── Does it take effect on its own, without a Save?
 │    └── Yes -> Toggle             (role="switch", 44x24 track)
 └── No -> Checkbox                 (part of a form the operator submits)
Is it one of a short, closed set?
 ├── Does the choice filter a list on this page?
 │    └── Yes -> FilterSelect, or Segmented if there are 4 or fewer
 ├── Do the options need explaining, or does seeing them all matter?
 │    └── Yes -> RadioGroup         (vertical; past ~5 options use Select)
 └── No -> Select
Is it one of a set too long to scroll, or open-ended?
 └── ComboBoxField                 (search as you type; `allowsCustomValue`
                                    for a list that is a shortcut, not a whitelist)
Is it many of a set?
 └── FilterMultiComboBox
```

`Select` and `FilterSelect` are two components rather than one with a mode, and
the same is true of `RadioGroup` against `Segmented`. In both pairs the filter
half wears a caption label beside the control and never speaks a validation
message, where the form half puts its label above, keeps it visible, and owns a
description and an error announced on the control. One component serving both
would have a prop list mostly ignored at each call site.

## Signatures

```ts
Field: { label, value, onChange: (next: string) => void, placeholder?, type = "text",
  isRequired?, isDisabled?, isInvalid?, errorMessage?, description?, autoFocus?,
  reserveMessage? }
TextArea: { label, value, onChange, placeholder?, rows = 4, description?, isRequired?,
  isDisabled?, isInvalid?, errorMessage?, reserveMessage?, className? }
SecretField: { label, value, onChange, placeholder?, description?, reserveMessage? }
SearchField: { label, value, onChange, placeholder = "Search", isDisabled?, className? }
Toggle: { label, isSelected, onChange: (next: boolean) => void, isDisabled? }
Checkbox: { isSelected, onChange: (next: boolean) => void, isDisabled?, ariaLabel?,
  children }
Select: { label, value, onChange, options: SelectOption[], description?, placeholder?,
  isRequired?, isDisabled?, isInvalid?, errorMessage?, reserveMessage?, className? }
ComboBoxField: { label, value, onChange, options: ComboBoxOption[], description?,
  placeholder?, isRequired?, isDisabled?, isInvalid?, errorMessage?, reserveMessage?,
  className?, allowsCustomValue?, autoFocus?, menuTrigger = "focus", shouldSelectOnFocus?,
  isSourceEmpty?, emptyMessage?, noMatchesMessage? }
RadioGroup: { label, value, onChange, options: RadioOption[], description?,
  orientation = "vertical", isRequired?, isDisabled?, isInvalid?, errorMessage?,
  className? }
FilterSelect: { value, onChange: (next: string) => void, options: { value, label }[],
  label?, ariaLabel?, id?, disabled? }
```

`FilterSelect` renders `label` as visible text beside the control, which is what a
filter in a toolbar wants; `ariaLabel` is for the case where the visible label would
repeat what is already on screen. Pass one or the other, never neither.

`onChange` takes the **value**, never an event. `reserveMessage` defaults to off, so
a field in a form has to opt in; a field in a table row or a toolbar leaves it off.

`Toggle` used to say `checked` and `disabled` where every HeroUI-backed control
says `isSelected` and `isDisabled`, and this file called that a real
inconsistency rather than a typo. It closed when the control was rehomed into
the design system: two controls an operator reads as a pair should not need two
prop vocabularies, so `Toggle` now matches `Checkbox` exactly. `FilterSelect` is
the one still saying `disabled`, because it is a filter rather than a field.

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

## ComboBoxField

`Select`'s vocabulary for everything the two share, plus what a combo box needs
beyond it. See its docstring for the contract; three things are worth knowing at
a call site.

**It filters nothing.** `options` is what the popover holds, already matched and
capped by the caller, because what counts as a match differs per field (an id as
well as a name) and a ceiling wants a "showing 50 of 300" line under it.

**An empty popover has to say which empty it is.** Every combo box here keeps
its menu open on an empty collection, so that a query matching nothing does not
read as a broken field. `noMatchesMessage` is about the query and `emptyMessage`
about the source, `isSourceEmpty` picks between them, and only the caller knows
what would fill an empty source: `ModelComboBox` names the provider credential
that discovery needs. The two sentences are `ComboBoxEmpty`, which the comboboxes
that are not form fields render through `ListBox`'s `renderEmptyState`.

**An option's `hint` is a second, muted line inside the row**, for text that
identifies the label rather than repeating it, such as the id a person is billed
under. It joins the row's accessible name, since it is what tells two rows with
one label apart.

`FilterMultiComboBox` stays a separate component for the reason `FilterSelect`
does: it is a filter, so its label is a caption beside the control and it never
speaks a validation message.

## Field height is a property of the place, not of the field

A field is **36px**. Inside a named dense place it is **32px**, and below `md`
every place raises it to **44px**, so the dense size is a desktop size the phone
layout takes back off. Three places exist:

| Place | Component that puts it on |
| --- | --- |
| `.otari-toolbar` | `layout/Toolbar` |
| `.otari-pagination` | `data/TablePagination` |
| `.otari-settings` | `layout/SettingsGroup`, and only with `bounded` |

`.otari-settings` differs from the other two in one way worth knowing: on a
phone its field also gains block padding and 16px type, where a toolbar's keeps
the dense 4px and 14px. That is because the row's control stops sharing the row
and stacks full width under its label there, so it is a form field again rather
than one of a strip of small ones, and iOS zooms the page when a field under
16px takes focus.

**No call site picks a height.** A page says "this row is a toolbar" and every
control in it agrees on a size, which is what stopped the six pages with a filter
row from each choosing their own.

The mechanism is worth knowing because it changed, and because the new shape is
the one to copy when adding a place. A place **declares two custom properties**:

```css
.otari-toolbar,
.otari-pagination {
  --field-height: 32px;
  --field-padding-block: 4px;
}
```

and the rule that reads them sits on `.input` and `.select__trigger`, because it
has to reach inside HeroUI's own DOM to find a select trigger. Adding a place is
therefore those two lines plus a class on the container, and `.otari-settings`
is the one that proves it: it arrived written the old way and converting it
moved no pixel, measured in all three places at both widths. It used to be
spelled the other way, as `.otari-toolbar .input { height: 32px }`, which worked
and had two costs. It was invisible from the call site: nothing on `<Toolbar>`
said it restyled the controls inside it. And it was unconditional, because a
descendant selector at (0,2,0) outranks anything a nested component can say about
its own field, so a control that legitimately wanted the form height inside a
toolbar had no way to ask. **A custom property inherits instead of winning**, so
the place still sets the density for everything inside it and any subtree can
reset it.

Adding a place is therefore two lines of CSS and a class on the container, and
`foundation.test.ts` holds the pair that matters: the dense value exists, and the
767px rule that undoes it exists. Drop the second and a phone gets a 32px search
box.

Two things are deliberately *not* places. `.table__cell .select__trigger` stays a
descendant selector, because a cell is HeroUI's own DOM rather than a place of
ours and a select is the only control that renders in one; making it a place
would pin a height on an `.input` a future cell might hold. And a bare
`input[type="search"]` outside a toolbar is untouched: the only two in the tree
are toolbar search boxes, and a native element carries none of HeroUI's classes,
so it reads the property through a rule of its own scoped to the place.

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

`Toggle` is `role="switch"` with an `aria-label` (its `label` prop, since a
switch has no visible text of its own). The visible track is 44x24 with a
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

## Control rows

A row of controls laid out `items-end` bottom-aligns each child's whole box, and a
field's box includes the caption line it reserves. These rows also wrap, and
`items-end` aligns each flex line to its own cross-end. So the trailing caption rung
is a property of the flex line, not of the control: in a control row every child
holds one caption line whether or not it speaks, the trailing action included. In a
stacked form the opposite holds, and a control that can never speak holds nothing.

Wrap a trailing action in `FieldAction`, which gives it that rung and nothing above
it. Reserve the line rather than nudging the control: a local `pb-*` on one child
guesses a number, and the guess is wrong the moment the caption is retuned.

```tsx
// Correct
<FieldAction>
  <Button variant="ghost" onPress={remove}>Remove</Button>
</FieldAction>

// Wrong: a hand-tuned nudge that does not track the caption
<Button variant="ghost" className="pb-2" onPress={remove}>Remove</Button>
```
