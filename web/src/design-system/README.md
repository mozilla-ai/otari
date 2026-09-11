# Otari design system

Presentational components, on the semantic tokens, that know nothing about this
application.

- **The specification** is [web/design/DESIGN.md](../../design/DESIGN.md): which
  component to reach for, which variant applies where, which token layer is
  allowed. Ten topic files; read the one covering what you are building.
- **The catalog** is Storybook: every component here with its variants, its
  states and both themes. `pnpm --dir web run storybook`, and
  [.storybook/README.md](../../.storybook/README.md) for how it is published.

## The one rule

**Nothing here may import anything else under `src/`.** React, HeroUI,
react-aria-components, react-icons, recharts, react-markdown, and the modules
in this directory.
That is the whole allowance; `biome.jsonc` rejects the rest and
`src/architecture.test.ts` proves each rejection.

The question it answers is not "does this import point the wrong way" but
**"would this directory still compile with the rest of `src/` deleted"**, because
this is a library that happens to live in a repository, and the point of the rule
is that turning it into a package stays a folder move rather than an
untangling.

So a component here is dumb and stateless: no TanStack Query, no
`useDeployment()`, no router, no context of the app's. State is a prop and a
callback, which is why every control is controlled and why `Avatar` takes
`initials` rather than a name.

A component that needs the transport, a domain formatter or a generated API type
is an *application* component. It belongs in `src/shared/components/`, composing
the primitive from here. DESIGN.md's "The extraction contract" has the reasoning,
the two directories that stayed behind, and the five plausible primitives
deliberately not built.

## Adding one

1. Put it in the topic directory named for the `web/design/` file that specifies
   it. A new topic needs a new file there; `overlays.md` is the most recent one.
2. One component per module, named for the file. No barrels, no default exports.
3. Closed prop unions (`variant`, `size`, `tone`), booleans that read as
   questions (`isDisabled`, `isInvalid`). `className` is for layout and position
   at the call site, never for restyling what the component already styles.
4. Customize in this order: a variable (ours as a token, or one of HeroUI's
   aliased onto ours), then a shared utility, then the component's own prop, and
   only last a rule into HeroUI's DOM, with a comment saying why nothing above
   reached it.
5. A `.stories.tsx` beside it covering each variant, each state, and both themes
   where the tokens differ. **Every prop has to be passed by some story**, which
   `src/styles/foundation.test.ts` checks: an optional prop is the one thing that
   can fall outside the catalog without anything noticing, because the typecheck
   is happy and the story simply never mentions it. A prop whose effect cannot be
   put on screen goes in that gate's `CANNOT_BE_SHOWN` with the reason, rather
   than being passed somewhere to satisfy the check.

   Coverage counts across the whole catalog, not per file: `Field`'s `isInvalid`
   is exercised by `FieldMessages.stories.tsx`, which is the right place for it.
6. A `.test.tsx` beside it for behavior you changed.
