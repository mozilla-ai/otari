# Navigation and selection

Three ways to switch, and they are not interchangeable.

```
Does the choice change what the page shows, as a sibling surface?
 └── TabRow            (Overview / Members / Domains)
Is it a closed set of alternatives to one value?
 └── Segmented         (24h / 7d / 30d / 90d)
Does it narrow a list that is already on screen?
 └── FilterChips + FilterSelect / FilterMultiComboBox
Does it change route?
 └── A rail row, from the nav registry. Never a Tab.
```

## Signatures

```ts
Tab: { isActive, onPress, children }
TabRow: { children }
Segmented: { label, value, onChange: (next: string) => void,
  options: { value, label }[] }
FilterChips: { chips: FilterChip[], children, onClearAll?, start?, end? }
FilterChip: { key, label, value, clearLabel?, onClear: () => void }
DismissChip: { value, onDismiss, label?, dismissLabel? }
```

`FilterChips` goes **inside** `Toolbar`, and it renders the selects (`children`) and
the chips they produce together, so a page places one component and not two. `start`
and `end` are slots on the control row for things that are not filters: a search
field at the start, a `RefreshButton` at the end.

`onClearAll` is the caller's to pass or omit; the component does not count for you.
Omit it below two chips.

A `FilterChip` splits the dimension (`label`, "Provider") from the current value
(`value`, "OpenAI") so the chip can print them at two weights, and its handler is
`onClear`, not `onRemove`. `FilterChips`'s `children` are the selects themselves: it
renders the controls and the chips together, so a page does not place both.

## TabRow

No track and no underline: the selected fill (`surface-subtle`) is the whole
signal. Deliberately not `role="tablist"`, because these do not promise the
roving-focus contract a tablist implies; they are buttons in a row.

## Segmented

The track is what says "these are the alternatives and there are no others". Native
radios underneath, so the semantics come free. Four options or fewer; past that use
a `FilterSelect`.

```tsx
// Correct
<Segmented
  label="Time range"
  value={range}
  onChange={setRange}
  options={[
    { value: "24h", label: "24h" },
    { value: "7d", label: "7d" },
    { value: "30d", label: "30d" },
  ]}
/>

// Incorrect: a switcher with no track reads as tabs, and this one does not
// change what the page is, only one of its values
<TabRow>{ranges.map((r) => <Tab key={r} isActive={r === range} onPress={…}>{r}</Tab>)}</TabRow>
```

## FilterChips

Each chip is removable and states its own dimension (`Provider: OpenAI`), so a
reader can tell why a list is short without opening a menu. `onClearAll` appears
once there are two. `DismissChip` is the single chip on its own.

## The rail

One row shell, four states. The 2px left border is **always present** and
transparent when unselected, so a selection adds no width and nothing shifts.

| State | Treatment |
| --- | --- |
| Resting | `text-muted`, transparent border |
| Hover | `surface-alt` fill, ink to full. Two channels, so it reads on touch |
| Selected | `surface-subtle` plus a 2px `border-foreground`. The one 2px rule in the product |
| Ancestor | Full ink, no fill: the open parent of the selected child |

Every row is `min-h-11` (44px) rather than vertical padding, so a two-line label
grows the row instead of breaking the touch floor.

Expanded rail is `16.5rem`, collapsed is `4.5rem` with icons centered in a 44px row
and labels in a tooltip. The states are the same values; only the label goes.

Never hand-write a nav row. `rowStyles.ts` owns the classes and
`rowStyles.test.ts` asserts them; a page adds a route to the registry instead.

## Section headings in the rail

`NAV_SECTION_HEADING_CLASS`, which is the `text-overline` role. A rail section
heading is a label above a group, which is exactly what that role is for.
