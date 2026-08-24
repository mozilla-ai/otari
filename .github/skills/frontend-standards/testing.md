# Testing the dashboard

Three suites, each with a different job:

| Suite | Command | Covers |
| --- | --- | --- |
| Vitest + Testing Library | `pnpm --dir web test` | Components, hooks, pure helpers. The bulk. |
| Playwright behavioral | `pnpm --dir web run e2e` | Multi-page flows against a real gateway. |
| Playwright screenshots | `pnpm --dir web run e2e:screenshots` | How every page renders, at three viewports in both themes. |

Backend testing is a different world (PostgreSQL, Testcontainers, the OSS smoke gate) and
lives in [AGENTS.md](../../../AGENTS.md) under "Test Notes".

## Vitest

`pnpm --dir web test` regenerates the API client first, then runs the suite. Tests are
colocated: `Foo.tsx` → `Foo.test.tsx`, `format.ts` → `format.test.ts`.

**Query the way an operator would.** `getByRole`, `getByLabelText`, `getByText`. Not
`getByTestId`, and never a class selector: `.bg-surface` is a token that will be renamed, and
a test that breaks on a restyle teaches everyone to stop trusting the suite.

**Mock the network boundary, nothing inside it.** The page tests spy on the transport and let
the real hooks, query keys, formatters, and derivations run:

```tsx
vi.spyOn(apiClient, "apiFetch").mockImplementation(async (input) => {
  const url = String(input)
  if (url.startsWith("/v1/models")) return CATALOG
  …
})
```

Mocking `useModels` or `formatCost` instead hides exactly the regressions worth catching: a
changed query key, a loading state nobody renders, a formatter that rounds wrong. There is no
`vi.mock("@/shared/api/hooks")` anywhere in this tree, and adding the first one needs a reason
in the diff.

**Render what the app renders.** A component that reads the URL needs a real router:
`withRouter` / `renderWithRouter` from `src/tests/router.tsx`. The router resolves its first
location asynchronously, so `await renderWithRouter(...)` or `flushRouter()` before asserting.
`src/tests/providers.tsx` re-exports the real provider tree for the tests that need it, and it
is the only module allowed to import `@/app`.

**Build fixtures from `src/tests/fixtures.ts`.** The builders fill a whole shape, which
matters because several code paths branch on `undefined` rather than zero
(`billedTokenTotal`, the share-card caveats): a hand-rolled partial fixture can pass a test
that the real response would fail.

**Drive with `userEvent`** and assert on what the operator sees, not on component internals.

**Wait on the event, not the clock.** `findBy*` and `waitFor` resolve the instant the DOM
changes; their timeout is a ceiling, not a sleep. Two habits that follow:

- Await the content, not its container. A wrapper can commit before its children do, so
  `await findByRole("tabpanel")` then a synchronous `getByText` inside it is a race. Await the
  text too.
- **A per-assertion `{ timeout: 5000 }` is a review blocker.** It hides which wait was slow.
  If the environment genuinely needs more headroom, raise it in one place
  (`configure({ asyncUtilTimeout })` in `src/tests/setup.ts`) and say why.

**Every test file is self-contained.** Vitest runs files in parallel across workers, so a
global one file leaves modified is a failure in another file that only reproduces at full
suite size:

- Restore any global you override (`window.location`, `matchMedia`, `localStorage`) in
  `afterEach`, and set `configurable: true` on every `Object.defineProperty` so the next test
  can redefine it.
- `vi.clearAllMocks()` and `localStorage.clear()` in `beforeEach`, not once at the top.
- `vi.useFakeTimers()` is always paired with `vi.useRealTimers()` in `afterEach`.
- Heavy polyfills go in the file that needs them. `src/tests/setup.ts` carries only what is
  universal (the jsdom gaps: `ResizeObserver`, `scrollTo`, object URLs).

**Keep test files small.** Vitest gives one file to one worker, so a 2,000-line file is the
suite's critical path while other workers idle, and `ActivityPage.test.tsx` is currently
exactly that. Split by concern (`Page.test.tsx`, `Page.deletion.test.tsx`,
`Page.filters.test.tsx`) when a file grows several independent `describe` blocks or needs
different mock setups per block.

## Playwright: behavioral

`pnpm --dir web run e2e` builds the bundle and boots a real gateway against a throwaway SQLite
database (`e2e/serve.sh`). Three ordered projects share that one database, so a spec scopes
itself to the rows it owns rather than to a global count, and an open react-aria combobox
popover `aria-hidden`s the rest of the page, so `dismissComboBox` runs before asserting
anything outside it. `playwright.config.ts` carries the rest.

A fourth project, `hybrid`, runs against a second gateway booted in hybrid mode
(`e2e/serve-hybrid.sh`), because a deployment attached to a control plane elsewhere is a
different shape rather than a different page, and only a server can put the browser in one.
It shares no state with the three above. Two mechanical rules come with adding any spec here:
a file collected by no project's `testMatch`/`testIgnore` is dropped from the run with a green
exit, and `package.json`'s `e2e` script names each project it runs, so a project missing from
that list never runs in CI (`src/e2eProjects.test.ts` fails on that half). The gate is
`otari-dashboard-parity.yml`, triggered by `src/gateway/**` as well as `web/**`: these flows
are gateway-served end to end, so a backend change can break one with no file under `web/`
touched.

## Playwright: screenshots

`e2e/screenshots/` is the visual-regression suite. Each spec is captured by six projects,
three viewports (1920×1080, 1280×800, 390×844) times both themes, so one entry covers a page
at every size and in both palettes. **The configuration is otari-ai/frontend's**, ported
rather than reinvented, because the two suites fail for the same reasons and only one of them
should have to learn each one. Keep them recognizably the same file.

**Adding a page means adding an entry**, in the registry that matches how it is reached: a
route in `WORKSPACE_ROUTES` in `authenticated.spec.ts` for anything behind a session, or a
test in `public.spec.ts` for anything in front of one. Either is one line and buys six
captures. A page with no entry is a page whose mobile and dark rendering nobody checks.

What the harness already handles, so you do not work around it:

- **The mobile project is a phone**, not a narrow desktop: `isMobile`, `hasTouch`, a
  `deviceScaleFactor` and an iPhone user agent, so touch guards, viewport meta and `:hover`
  rules behave as they would on a device. Playwright captures at CSS scale, so the PNG is
  390 wide whatever the scale factor.
- **A frozen clock** (`page.clock.setFixedTime`), so anything the page reads off the browser
  clock renders identically every run. What a fixed clock cannot fix is a timestamp relative to
  a row the seed created at run time, which is why those are masked below. Deliberately not
  `clock.install`, which also fakes timers and deadlocks TanStack Query's refetching and
  React's scheduler.
- **Animations and transitions frozen** by injected CSS, the scrollbar hidden (its width
  differs between a laptop and the CI container, and it runs the height of a full-page shot),
  SVG timelines paused at t=0, fonts awaited, scroll position reset, and a few frames allowed
  to land before the shutter. That is `waitForStable` in `fixtures.ts`.
- **Pinned `locale`, `timezoneId` and `reducedMotion`** on the screenshot projects only, so
  date and number formatting cannot vary while the behavioral suite keeps its own behavior.
- **Two comparison budgets, not one.** `maxDiffPixels: 2000` alongside
  `maxDiffPixelRatio: 0.002`, because Playwright takes the smaller of the two and a ratio
  alone scales with page height: a tall full-page capture would earn an allowance big enough
  to absorb a whole changed paragraph. `threshold: 0.12` is the per-pixel color distance that
  counts as a difference at all, set high enough that antialiasing does not register as one.
- **Masks** over the two things still not reproducible here: recharts, which animates through
  JavaScript that neither the CSS freeze nor `animations: "disabled"` reaches, and relative
  timestamps, which move with the gap between the frozen clock and rows the seed created at
  run time. otari-ai needs neither, because its captures run against static mocked responses;
  ours will not either once they do.

If you find a third source of noise, mask it in `fixtures.ts` with a comment rather than
loosening the budgets for everything.

The suite depends on the `seed` project and deliberately not on `parity`. Some database state
is needed, but each parity flow removes what it creates, so the seed is the fixture; depending
on the flows as well only means one flaky behavioral test takes all 108 captures with it,
which is exactly what it did before this was narrowed.

**No baselines are committed yet, and the suite is not a gate.** The dashboard is
mid-migration onto the rehomed design foundation, so pages still move for good reasons: a
committed set would fail most PRs and churn in every diff. So the `screenshots` job runs only
on `workflow_dispatch`, `e2e/screenshots/*-snapshots/` is gitignored, and a PR that changes
how a page looks owes no PNGs.

What a PR does owe is the entry above, because that is what makes the page covered the day
this becomes a gate rather than the day someone remembers it.

Running it is still worth doing when you have changed a layout: `pnpm --dir web run
e2e:screenshots` captures every page at every size locally and leaves the PNGs beside the spec
that took them. Treat them as a rendering check, not a diff: a macOS capture renders fonts
differently from CI's Linux, so the two sets are not comparable. To get a Linux set (which is
also what a future baseline commit needs), run the workflow from the Actions tab or with
`gh workflow run otari-dashboard.yml --ref <branch>` and download the `screenshot-baselines`
artifact.

`pnpm --dir web run e2e:screenshots:update` is the deliberate-update form, and note the
missing `--`: pnpm forwards the separator and Playwright's CLI discards everything after it,
so `-- --update-snapshots` is a flag that silently never arrives.

Turning it into a gate later is three edits: add `pull_request` back to the job's condition,
drop the `.gitignore` entry, and commit the Linux set that run captures.
