import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it } from "vitest"

import { usePolicyWriter } from "@/features/tools/usePolicyWriter"

interface Stored {
  row: string
  a: string
  b: string
  readOnly: string
}
type Body = Pick<Stored, "a" | "b">

/**
 * A card with two fields over one row, which is the shape both workspace policy
 * groups have: each field commits the whole body, and the row can change under
 * them when the operator switches workspace.
 */
function Harness({
  server,
  put,
}: {
  server: Stored
  put: (body: Body) => Promise<Stored>
}) {
  const [row, setRow] = useState(server.row)
  const write = usePolicyWriter({
    server: { ...server, row },
    resetKey: row,
    toBody: (stored) => ({ a: stored.a, b: stored.b }),
    put,
  })
  return (
    <>
      <button type="button" onClick={() => void write({ a: "A!" }).catch(noop)}>
        save a
      </button>
      <button type="button" onClick={() => void write({ b: "B!" }).catch(noop)}>
        save b
      </button>
      <button type="button" onClick={() => setRow("second")}>
        switch row
      </button>
    </>
  )
}

const noop = () => undefined
const press = (user: ReturnType<typeof userEvent.setup>, name: string) =>
  user.click(screen.getByRole("button", { name }))

describe("usePolicyWriter", () => {
  it("builds the second body from the first write's answer, not the query", async () => {
    // The lost update this exists for: each control has its own save state, so
    // two fields can be in flight at once over one row.
    const bodies: Body[] = []
    let release: (() => void) | undefined
    const put = async (body: Body) => {
      bodies.push(body)
      if (bodies.length === 1) {
        await new Promise<void>((resolve) => {
          release = resolve
        })
      }
      return { row: "first", readOnly: "server", ...body }
    }
    const user = userEvent.setup()
    render(
      <Harness
        server={{ row: "first", a: "a", b: "b", readOnly: "server" }}
        put={put}
      />,
    )

    await press(user, "save a")
    await press(user, "save b")
    release?.()

    await waitFor(() => expect(bodies).toHaveLength(2))
    expect(bodies[1]).toEqual({ a: "A!", b: "B!" })
  })

  it("sends only what `toBody` names", async () => {
    const bodies: Body[] = []
    const user = userEvent.setup()
    render(
      <Harness
        server={{ row: "first", a: "a", b: "b", readOnly: "server" }}
        put={async (body) => {
          bodies.push(body)
          return { row: "first", readOnly: "server", ...body }
        }}
      />,
    )

    await press(user, "save a")

    await waitFor(() => expect(bodies).toHaveLength(1))
    expect(Object.keys(bodies[0]).sort()).toEqual(["a", "b"])
  })

  it("does not carry a write that resolves after the row changed", async () => {
    // The old row's write is still in flight when the operator switches. Its
    // answer must not become the new row's base, or the next commit builds a
    // body out of the old row's values and PUTs it to the new one.
    const bodies: Body[] = []
    let releaseFirst: (() => void) | undefined
    const put = async (body: Body) => {
      bodies.push(body)
      if (bodies.length === 1) {
        await new Promise<void>((resolve) => {
          releaseFirst = resolve
        })
        return { row: "first", readOnly: "server", a: "STALE", b: "STALE" }
      }
      return { row: "second", readOnly: "server", ...body }
    }
    const user = userEvent.setup()
    render(
      <Harness
        server={{ row: "first", a: "a", b: "b", readOnly: "server" }}
        put={put}
      />,
    )

    await press(user, "save a")
    await press(user, "switch row")
    // The first row's write answers only now, after the switch.
    releaseFirst?.()
    await waitFor(() => expect(bodies).toHaveLength(1))

    await press(user, "save b")

    await waitFor(() => expect(bodies).toHaveLength(2))
    expect(bodies[1]).toEqual({ a: "a", b: "B!" })
    expect(bodies[1]).not.toMatchObject({ a: "STALE" })
  })

  it("does not send a commit that was superseded before its turn", async () => {
    // The escape the store-side guard alone leaves open: A2 is queued behind
    // an in-flight A1, the row switches, B writes and stores its values as the
    // carried base, and only then does A1 settle and let A2 run. A2 would read
    // B's values and PUT them through A's own `put`.
    const putsByRow: { row: string; body: Body }[] = []
    let releaseA1: (() => void) | undefined
    let releaseB: (() => void) | undefined

    function TwoRows() {
      const [row, setRow] = useState("first")
      const write = usePolicyWriter({
        server: {
          row,
          a: row === "first" ? "a" : "B-a",
          b: "b",
          readOnly: "s",
        },
        resetKey: row,
        toBody: (stored: Stored) => ({ a: stored.a, b: stored.b }),
        put: async (body: Body) => {
          const forRow = row
          putsByRow.push({ row: forRow, body })
          if (putsByRow.length === 1) {
            await new Promise<void>((r) => {
              releaseA1 = r
            })
          }
          if (forRow === "second") {
            await new Promise<void>((r) => {
              releaseB = r
            })
            return { row: "second", readOnly: "s", a: "B!", b: "B!" }
          }
          return { row: forRow, readOnly: "s", ...body }
        },
      })
      return (
        <>
          <button
            type="button"
            onClick={() => void write({ a: "A1" }).catch(noop)}
          >
            a1
          </button>
          <button
            type="button"
            onClick={() => void write({ b: "A2" }).catch(noop)}
          >
            a2
          </button>
          <button type="button" onClick={() => setRow("second")}>
            switch
          </button>
        </>
      )
    }

    const user = userEvent.setup()
    render(<TwoRows />)

    await press(user, "a1")
    await press(user, "a2")
    await press(user, "switch")
    await press(user, "a1")
    releaseB?.()
    releaseA1?.()

    await waitFor(() => expect(putsByRow.length).toBeGreaterThanOrEqual(2))
    // A2 never went out at all, so nothing carrying the second row's values
    // was ever sent through the first row's `put`.
    expect(
      putsByRow.filter((p) => p.row === "first" && p.body.b === "A2"),
    ).toHaveLength(0)
  })

  it("keeps writing after one is refused", async () => {
    const bodies: Body[] = []
    const put = async (body: Body) => {
      bodies.push(body)
      if (bodies.length === 1) throw new Error("refused")
      return { row: "first", readOnly: "server", ...body }
    }
    const user = userEvent.setup()
    render(
      <Harness
        server={{ row: "first", a: "a", b: "b", readOnly: "server" }}
        put={put}
      />,
    )

    await press(user, "save a")
    await waitFor(() => expect(bodies).toHaveLength(1))
    await press(user, "save b")

    // The chain is not poisoned, and the refused write left no base behind.
    await waitFor(() => expect(bodies).toHaveLength(2))
    expect(bodies[1]).toEqual({ a: "a", b: "B!" })
  })
})
