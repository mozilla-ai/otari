# Buttons

Use a `Button` for an action. Navigation that looks like a button is a `Link`.
An action inside a table row is a `RowAction`, not a `Button`.

## Three variants. Nothing else exists.

`variant`: `primary`, `ghost`, `danger`.

`secondary`, `tertiary`, `outline` and `danger-soft` are retired. HeroUI's union
still accepts all seven, so a retired name is **not** a type error: it is a silently
unstyled button. `src/styles/foundation.test.ts` is what stops it, and it names the
file and line.

- `primary`: the one thing a band exists to do. **One per band** (see the rule under
  the flowchart, which is the part people get wrong). Filled teal under white ink.
- `ghost`: everything else. Transparent, with a 1px opaque edge. It absorbed
  `outline`, `secondary` and `tertiary`.
- `danger`: irreversible or hard to undo. **Unfilled**: transparent ground, danger
  ink, danger edge, filling to `--color-danger-subtle` on hover. A filled
  destructive button read as a second primary, which is the whole reason it is not
  one.

```
Is it the single thing this band is for?
 ├── Yes -> variant="primary"          (two in one band means the band has no hierarchy)
 └── No
      ├── Is it destructive or hard to undo?
      │    └── Yes -> variant="danger", and reach for ConfirmButton
      └── Default -> variant="ghost"
```

**"One per screen" means one per band, not one per route.** A page with a
`PageIntro` action and a `SettingsGroup` Save has two primaries, and that is
correct: each is the one thing its own band is for, and the bands are read one at a
time. The rule is about two primaries competing inside the same band, which is what
leaves a reader with no idea which control the band wants. When two do land in one
band, the secondary one becomes a ghost.

```tsx
// Correct
<Button variant="primary" onPress={create}>Create key</Button>
<Button variant="ghost" onPress={exportCsv}>Export CSV</Button>

// Incorrect: "outline" is retired. This compiles, lints, ships, and paints
// an unstyled button.
<Button variant="outline" onPress={exportCsv}>Export CSV</Button>
```

`Button` takes `isPending` while a mutation is in flight; it disables itself and
shows its own spinner. Do not pair it with `isDisabled` for the same condition.

## Sizes

`sm` 32px, `md` 36px (the default), `lg` 40px. Press is `scale(0.98)` at every
rung, and a full-width button does not press at all: a wide element displaces too
far for the same scale.

## Places: where a ghost drops its edge

An edge that would be too loud is suppressed by **naming the container**, so a call
site never has to remember. Inside one of these, a ghost renders edgeless:

`.otari-toolbar` · `.otari-table` · `.otari-pagination` · `.otari-bulk-bar` ·
`.otari-breakdown` · `.otari-rail`

Put the class on the container, not on the button. If you are building a new
container that holds a row of ghosts and the edges read as a grid of boxes, add a
place rather than styling the buttons.

**An icon-only ghost never takes the edge**, and that rule is keyed on what the
control *is*, not where it sits: `CopyButton` renders in tables, panels, banners and
bare pages, so no container can reach it.

## The two-step destructive confirm

`ConfirmButton` for a page-level destructive action, `ConfirmRowAction` for one in
a table row. Both arm on the first click and destroy on the second.

The resting trigger is **neutral** and the armed confirm is danger. The first click
is safe, so spending the loudest signal in the product on it wastes it; the danger
hue marks the irreversible step.

Label the two steps differently: the trigger names the object, the armed confirm
names the consequence.

```tsx
// Correct
<ConfirmButton confirmLabel="Remove permanently" onConfirm={remove}>
  Remove tool
</ConfirmButton>

// Incorrect: the same word twice tells the operator nothing about what changed
<ConfirmButton confirmLabel="Remove" onConfirm={remove}>Remove</ConfirmButton>
```

**The Cancel that appears when armed is load-bearing. Do not simplify it away.**
The escalation is hue-only, and hue is the one channel a red-green deficiency
removes: measured, the ghost edge against the danger edge is 1.17:1 in light and
1.11:1 in dark, and 1.03:1 under simulated protanopia. What survives is structural:
a second control appears and the row's layout changes, which no color deficiency
hides.

## Other action shapes

| Component | Props | Use for |
| --- | --- | --- |
| `RowAction` | `onPress`, `isDanger?`, `isDisabled?`, `ariaLabel?`, children | An action in a table row. Caption-sized, not a `Button` |
| `RowActionRow` | children | The trailing lane those sit in. **Use this one** |
| `RowActions` | children | A near-duplicate with a tighter gap, on 2 call sites. Do not reach for it in new code |
| `ConfirmButton` | `confirmLabel`, `onConfirm`, `isPending?`, children | The page-level two-step confirm |
| `ConfirmRowAction` | `confirmLabel`, `onConfirm`, `isPending?`, children | The same two-step inside a row. It supplies its own `isDanger` and its own Cancel |
| `RefreshButton` | `onRefresh`, `isFetching?`, `updatedAt?`, `label?` | A refetch, with its own freshness caption. Pass `updatedAt` or the caption reads nothing |
| `CopyButton` | `value`, `label` | Copy one value. Icon-only, 44x44 hit area |
