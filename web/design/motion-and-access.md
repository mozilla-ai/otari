# Motion and accessibility

## Durations

| Value | For |
| --- | --- |
| 100ms, ease-out | A control's color change (a button, a chip) |
| 150ms, ease-smooth | A rail row, a disclosure chevron, a knob's travel |
| 250ms, ease-out-fluid | A drawer or overlay sliding in |
| 0 | Everything, under `prefers-reduced-motion` |

**Every transition is guarded.** Write `motion-reduce:transition-none` beside every
`transition-*`. A vestibular disorder is not an edge case, and the guard costs one
utility.

## Press

`scale(0.98)` at every size. One value, not a ladder: the ladder deepened with
size, so a wider button displaced further on a deeper value (0.57px at `sm`,
2.78px at `lg`, per edge). A full-width control does not press at all.

The pressed color step is what makes a flat transform enough. If a variant loses
its press color, the transform ladder has to come back with it.

## Focus

```css
outline: 2px solid var(--focus-ring-ink, var(--color-focus));
outline-offset: 2px;
```

Applied by a base rule on `:focus-visible`, so you get it for free. Two places need
the utility form (`focus-visible:otari-focus-ring`) because they reach into vendor
DOM: HeroUI's `.button` carries `outline-none`, and a checkbox's visual box is a
sibling of the element that actually takes focus.

The ring's ink is chosen by **the ground being ringed**, not by a list of
components: a control whose own fill is an accent or a status color takes the
theme's ink, everything else keeps `--color-focus`. Naming fills rather than
components is the point, because a list of variants goes stale silently the next
time someone builds a filled thing.

**Never remove a focus ring.** Never replace it with a box-shadow.

## Disabled

`opacity: 0.4`, product-wide, from `--disabled-opacity`. A disabled control has to
read as denied rather than merely quiet. The cursor is `not-allowed`.

Show a control disabled rather than hiding it when its absence would be confusing,
and only when it carries its own reason nearby.

## Touch

**44px is the floor, everywhere.** Express it as `min-h-11` (or a 44x44 flex box for
an icon-only control), not as padding, so a longer label grows the target instead of
breaking it.

Two ways a small visual keeps a large target:
- An icon glyph is 16px inside a 44x44 flex container.
- A 24px toggle track grows its hit area with a `before:` pseudo-element
  (`before:-inset-y-2.5`), absolutely positioned so the row does not move.

**One control is knowingly under the floor**, and it is the shape of the exception
rather than a licence to add more. `DismissChip`'s remove button is 24px, because
neither device above reaches it: its row wraps at an 8px gap, so a `before:` bleed
would overlap the row above and a press near the seam would dismiss the neighboring
filter, and a real 44px target grows the filter area on three pages. #947 carries
the decision. A new control under 44px needs the same kind of argument, in writing,
or it is a bug.

Hover is never the only channel. A hover state that carries information alone does
not exist on a phone: pair it with a fill *and* an ink change, as the rail rows do.

## Names

- Every icon-only control takes an `aria-label`.
- Every chart takes an `ariaLabel`; a chart without one is a picture.
- Every `Meter` takes an `ariaLabel`, because a bar has no text.
- A status is never color alone. `SeverityMark` ships a word beside the mark.
- `DataTable` takes a required `ariaLabel`, because "Table" is not a name.

## Cursor

`button`, `summary`, `[role="button"]`, `[role="switch"]` and `[role="tab"]` take
`cursor: pointer` from one base rule, keyed on what the element is rather than on a
class a call site has to remember. A disabled one takes `not-allowed`.
