# Responsiveness, mobile, and touch: `web/`

The dashboard is installable to a phone home screen. `web/public/pwa/` ships a maskable
Android icon and an `apple-touch-icon`, `index.html` carries the `apple-mobile-web-app-*`
tags, and the gateway serves the manifest, so an operator can and does add
it to a home screen. A page that only works at 1440px is a broken page, not a desktop-first
one.

## What the shell already does, and what it does not

`src/app/AppShell.tsx` handles the chrome: below `MOBILE_QUERY` (`max-width: 767px`, Tailwind's
`md` boundary) the sidebar becomes an off-canvas drawer with a focus trap, the background is
`inert` while it is open, and the collapse chevron is hidden. That is the part that is done.

**Pages are the gap.** Outside the shell the tree has a handful of `sm:grid-cols-*` and three
`md:hidden`, and every table renders at its desktop width whatever the viewport. Treat a page
you are touching as unreviewed for mobile until you have looked at it at 390px wide.

## Breakpoints

| Prefix | Min width |
| --- | --- |
| `sm:` | 640px |
| `md:` | 768px |
| `lg:` | 1024px |
| `xl:` | 1280px |
| `2xl:` | 1536px |

Write mobile-first: the unprefixed classes are the phone layout, and `md:` / `lg:` add to it.
`md` is the one that matters most here, because it is where the shell switches, so a page that
assumes a sidebar next to it should assume it only from `md:` up.

The screenshot matrix (`web/e2e/screenshots/`) captures every page at 1920, 1280, and 390 in
both themes, so a layout that breaks at one of those widths fails CI rather than reaching an
operator. See [testing.md](./testing.md).

## Rules

- **Flex and grid, never a fixed width**, for anything that should reflow. `min-w-[11.25rem]` on
  a wrapping stat card is fine; a fixed-width container that decides the page is not.
- **`min-w-0` on flex children that can overflow** (long model ids, keys, code, tables).
  Without it a flex child's `min-content` width wins and the child pushes the layout wider
  than the viewport, which is what turns a small overflow into a horizontally scrolling page.
- **`rem`, not `px`, in arbitrary values.** `h-[20rem]`, not `h-[300px]`, because `rem`
  follows the reader's root font size and `px` overrides an accessibility setting they chose
  deliberately. A `1px` hairline border is the exception. A `text-[11px]` is a second
  exception that should not exist: that size is `text-overline` in the type scale, so use the
  role (see [design-tokens.md](./design-tokens.md)).
- **Touch targets are at least 44px** on the phone viewport, which Apple's HIG requires and
  HeroUI's `size="sm"` icon button (32px) does not meet. Size it up below `md`:
  `className="min-w-11 w-11 h-11 md:min-w-8 md:w-8 md:h-8"`.
- **Hover-only visibility is forbidden for anything functional.** A touch device has no hover,
  so `opacity-0 group-hover:opacity-100` is a control that does not exist on a phone. Write
  `opacity-100 md:opacity-0 md:group-hover:opacity-100` instead: always visible where there is
  no pointer, revealed on hover where there is.
- **A keyboard shortcut needs a button too.** If a feature is reachable only by a chord, it is
  unreachable on the device with no keyboard.
- **Guard `autoFocus` outside a modal.** On a page load it raises the soft keyboard over half
  the screen. Inside a dialog the operator just opened it is expected, so it is fine there.

## Tables

Tables are the dashboard's main content type and the thing that does not shrink. Two
acceptable answers, both better than a table clipped at the viewport edge:

```tsx
// Card list below md, table from md up
<div className="hidden md:block">
  <DataTable … />
</div>
<div className="md:hidden flex flex-col gap-3">
  {rows.map((row) => (
    <RowCard key={row.id} row={row} />
  ))}
</div>
```

```tsx
// Or scroll the table itself, when the row still reads without its leading columns
<div className="overflow-x-auto">
  <DataTable className="min-w-[40rem]" … />
</div>
```

The page body must never be what scrolls horizontally. Put the overflow on the table's own
wrapper so the header, filters, and pagination stay put.

## Anti-patterns

- A fixed-pixel layout container.
- `opacity-0 group-hover:*` on a control that does something.
- A 32px icon button with no mobile size override.
- `autoFocus` on a page mount, unguarded.
- A table with no answer below `md`.
- `px` in an arbitrary value where the thing should scale with text.
