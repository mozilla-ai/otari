import { chromium } from "@playwright/test"

// Override with SMOKE_ORIGIN to point at another port; the catalog is served
// on 6006 by `storybook dev` and by the static server CI runs.
const ORIGIN = process.env.SMOKE_ORIGIN ?? "http://localhost:6006"

// A story has rendered when it put something on screen: real text, an SVG, a
// portalled dialog, or something that PAINTS.
//
// That last one is why this is a function rather than a text check. The design
// system has primitives whose whole output is a shape: a toggle's track, a
// divider's hairline, a meter's bar, a skeleton's block. Each renders correctly
// with no text node and no SVG anywhere, so a text-or-svg predicate reports
// nine of them as timeouts and the catalog gate fails on components that are
// fine.
//
// It cannot be relaxed to `childElementCount`, for the reason this probe was
// written around: several stories wrap their subject in a sizing <div>, which
// exists immediately and would satisfy a child count while the component inside
// is still null. So "paints" is the narrower question, asked of the descendants
// rather than the wrapper: a box with a non-zero size AND either a fill or a
// border. A sizing <div> has neither, which is exactly the case that has to keep
// failing.
const RENDERED = () => {
  const b = document.body.classList
  if (b.contains("sb-show-errordisplay")) return true
  if (!b.contains("sb-show-main")) return false
  const root = document.querySelector("#storybook-root")
  const portal = document.querySelector('[role="alertdialog"], [role="dialog"]')
  if (portal) return true
  const text =
    (root?.textContent ?? "").trim().length +
    (portal?.textContent ?? "").trim().length
  if (text > 0) return true
  if (document.querySelectorAll("#storybook-root svg").length > 0) return true
  return [...(root?.querySelectorAll("*") ?? [])].some((el) => {
    const box = el.getBoundingClientRect()
    if (box.width === 0 || box.height === 0) return false
    const style = getComputedStyle(el)
    const filled =
      style.backgroundColor !== "transparent" &&
      style.backgroundColor !== "rgba(0, 0, 0, 0)"
    const edged =
      Number.parseFloat(style.borderTopWidth) > 0 ||
      Number.parseFloat(style.borderRightWidth) > 0 ||
      Number.parseFloat(style.borderBottomWidth) > 0 ||
      Number.parseFloat(style.borderLeftWidth) > 0
    return filled || edged
  })
}

const index = await fetch(`${ORIGIN}/index.json`).then((r) => r.json())
const ids = Object.keys(index.entries).filter((id) => index.entries[id].type === "story")

// One flat work list, drained by a pool of pages.
//
// It used to be two sequential loops over one page each, which on a CI runner
// took over 45 minutes for 327 stories and printed nothing until the end, so a
// slow run and a hung one were indistinguishable. Both halves of that are fixed
// here: the pool gives it the runner's cores, and the progress line means an
// instrument that has stopped moving says so. AGENTS.md makes the same point
// about a probe whose silence reads as success.
//
// The pool size is deliberately modest. The catalog is served by a plain file
// server and each navigation pulls the whole bundle, so past a handful of pages
// the server, not the browser, is the limit.
const TASKS = ["light", "dark"].flatMap((theme) =>
  ids.map((id) => ({ theme, id })),
)
const CONCURRENCY = Number(process.env.SMOKE_CONCURRENCY ?? 6)

const browser = await chromium.launch()
const failures = []
let cursor = 0
let finished = 0

async function drain() {
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } })
  while (true) {
    const next = cursor++
    if (next >= TASKS.length) break
    const { theme, id } = TASKS[next]
    const errors = []
    const onErr = (e) => errors.push(String(e.message ?? e).split("\n")[0])
    // Only uncaught exceptions and React errors count. A bare "Failed to load
    // resource" is this probe's own fault: it reuses one page across many rapid
    // navigations, so in-flight requests abort as it moves on.
    const onConsole = (m) => {
      if (m.type() !== "error") return
      const text = m.text().split("\n")[0]
      // Aborted requests only, not every failed one. This used to drop the
      // whole "Failed to load resource" family, which is why it reported the
      // catalog clean while every story 404d on `/v1/organizations/me`: the
      // decorators mount a provider that queries the organization, and nothing
      // answered it. Now that `apiMock` owns every `/v1/` path, a load failure
      // is a real finding. The abort is still this probe's own fault, since it
      // reuses one page across hundreds of rapid navigations.
      if (/net::ERR_ABORTED/.test(text)) return
      if (text.trim() === "%o") return
      errors.push(text)
    }
    page.on("pageerror", onErr)
    page.on("console", onConsole)
    try {
      await page.goto(`${ORIGIN}/iframe.html?id=${id}&globals=theme:${theme}`, { waitUntil: "domcontentloaded" })
      // See RENDERED at the top for what counts as rendered and why. The one
      // thing worth repeating here: a card gated on two sequential queries
      // (context, then a per-workspace one) legitimately renders nothing until
      // the second resolves, which is why this waits rather than sampling once.
      // `polling: 100` and not the default, which is `raf`.
      //
      // A backgrounded page does not get animation frames, and this drains the
      // work list from a pool, so all but one page is backgrounded at any
      // moment: on the default the predicate is evaluated a handful of times a
      // second or not at all, and a story that renders instantly still burns
      // seconds waiting to be asked. web/AGENTS.md records the same mechanism
      // under "Measuring the running dashboard", where a probe that awaits a
      // frame in a background tab hangs rather than returning. A fixed interval
      // does not care whether the page is visible.
      await page.waitForFunction(RENDERED, undefined, {
        timeout: 20000,
        polling: 100,
      })
      const shown = await page.evaluate(() => ({
        errored: document.body.classList.contains("sb-show-errordisplay"),
        errorText: document.querySelector("#error-message")?.textContent?.slice(0, 300) || null,
        children:
          (document.querySelector("#storybook-root")?.childElementCount ?? 0) +
          (document.querySelector('[role="alertdialog"], [role="dialog"]') ? 1 : 0),
        text:
          (document.querySelector("#storybook-root")?.textContent ?? "").trim().length +
          (document.querySelector('[role="alertdialog"], [role="dialog"]')?.textContent ?? "").trim().length,
        svgs: document.querySelectorAll("#storybook-root svg").length,
        // A full-bleed region reaches the width of its container query, so a
        // story that clamps one inside a fixed-width wrapper pushes the band's
        // left edge off the viewport, where it cannot be scrolled back into
        // view. Seven stories shipped that way, cut about 114px in from the
        // left. Negative and not "left of the canvas": a band bleeding out to
        // the canvas edge is what it does on a page, and only what leaves the
        // viewport is unreachable.
        clipped: [...document.querySelectorAll(".otari-bleed")]
          .map((el) => Math.round(el.getBoundingClientRect().left))
          .filter((left) => left < 0),
      }))
      if (shown.errored) failures.push({ theme, id, why: "error display", errorText: shown.errorText })
      else if (shown.children === 0) failures.push({ theme, id, why: "empty root" })
      // Asks the same question the wait did. Without this a graphical primitive
      // clears the wait and is then reported as "no text and no svg".
      else if (shown.text === 0 && shown.svgs === 0 && !(await page.evaluate(RENDERED)))
        failures.push({ theme, id, why: "no text, no svg, and nothing painted" })
      else if (errors.length) failures.push({ theme, id, why: "console errors", errors: [...new Set(errors)].slice(0, 3) })
      else if (shown.clipped.length) failures.push({ theme, id, why: "a full-bleed region is cut off the left of the viewport", left: shown.clipped })
    } catch (e) {
      failures.push({ theme, id, why: "timeout", detail: String(e.message).split("\n")[0], errors: [...new Set(errors)].slice(0, 2) })
    }
    page.off("pageerror", onErr)
    page.off("console", onConsole)
    finished += 1
    if (finished % 50 === 0 || finished === TASKS.length) {
      console.log(`  rendered ${finished}/${TASKS.length}, ${failures.length} failing so far`)
    }
  }
  await page.close()
}

await Promise.all(Array.from({ length: CONCURRENCY }, drain))

console.log(`checked ${ids.length} stories x 2 themes`)
console.log(failures.length === 0 ? "ALL RENDERED CLEAN" : JSON.stringify(failures, null, 1))
await browser.close()
