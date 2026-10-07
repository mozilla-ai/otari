import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { SpanContentSection } from "@/features/traces/SpanContentSection"
import { API_ROOT } from "@/shared/api/client"
import { jsonResponse, mockApi } from "@/tests/activity"
import { traceSpan } from "@/tests/traceFixtures"

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

interface Call {
  url: string
  method: string
  body: string | undefined
}

// Serves the organization context, which picks the trace scope, and answers
// every content route with `content`.
function serve({
  operator,
  content,
}: {
  operator: boolean
  content: () => Response
}) {
  const calls: Call[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    calls.push({
      url,
      method: (init?.method ?? "GET").toUpperCase(),
      body: typeof init?.body === "string" ? init.body : undefined,
    })
    if (url.includes("/content")) return content()
    return jsonResponse({
      organization: { id: "o" },
      deployment_operator: operator,
      workspace_memberships: [],
    })
  })
  return calls
}

function contentCalls(calls: Call[]): Call[] {
  return calls.filter((call) => call.url.includes("/content"))
}

const ARGUMENTS = () =>
  jsonResponse({
    fields: { arguments: '{"command": "ls"}', result: "README.md" },
  })

afterEach(() => {
  vi.restoreAllMocks()
})

describe("SpanContentSection", () => {
  it("says the content was not captured, and fetches nothing", () => {
    const { calls } = mockApi()

    renderWithClient(
      <SpanContentSection
        traceId="s-1"
        span={traceSpan({ has_content: false })}
      />,
    )

    expect(
      screen.getByText(/Content was not captured for this span/),
    ).toBeInTheDocument()
    expect(contentCalls(calls)).toEqual([])
  })

  it("reads the session owner's content through the organization route", async () => {
    const calls = serve({ operator: false, content: ARGUMENTS })

    renderWithClient(
      <SpanContentSection
        traceId="s-1"
        span={traceSpan({ span_id: "call-1", has_content: true })}
      />,
    )

    expect(await screen.findByText("README.md")).toBeInTheDocument()
    expect(screen.getByRole("region", { name: "Arguments" })).toHaveTextContent(
      '{"command": "ls"}',
    )
    expect(contentCalls(calls).map((call) => call.url)).toEqual([
      `${API_ROOT}/organizations/me/traces/s-1/spans/call-1/content`,
    ])
  })

  it("shows a step's input, the model's output and the prior output, in that order", async () => {
    serve({
      operator: false,
      content: () =>
        jsonResponse({
          fields: {
            prior_output: "Running ls.",
            output: "The repo has a README.",
            input: "README.md",
          },
        }),
    })

    renderWithClient(
      <SpanContentSection
        traceId="s-1"
        span={traceSpan({ span_id: "req-1", has_content: true })}
      />,
    )

    expect(
      await screen.findByRole("region", { name: "Model output" }),
    ).toHaveTextContent("The repo has a README.")
    expect(
      screen.getAllByRole("region").map((region) => region.ariaLabel),
    ).toEqual(["Input", "Model output", "Model output before this request"])
  })

  it("says only the person who ran the session may read it when refused", async () => {
    serve({
      operator: false,
      content: () =>
        jsonResponse(
          {
            detail: "Only the person who ran this session can read its content",
          },
          403,
        ),
    })

    renderWithClient(
      <SpanContentSection
        traceId="s-1"
        span={traceSpan({ span_id: "call-1", has_content: true })}
      />,
    )

    expect(
      await screen.findByText(
        /Only the person who ran this session can read its content/,
      ),
    ).toBeInTheDocument()
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("says so when the content has expired or was purged", async () => {
    serve({
      operator: false,
      content: () => jsonResponse({ detail: "No content" }, 404),
    })

    renderWithClient(
      <SpanContentSection
        traceId="s-1"
        span={traceSpan({ span_id: "call-1", has_content: true })}
      />,
    )

    expect(await screen.findByText(/expired, was purged/)).toBeInTheDocument()
  })

  it("says the content cannot be read right now when the key store is unavailable", async () => {
    serve({
      operator: false,
      content: () =>
        jsonResponse({ detail: "Trace content is unavailable" }, 503),
    })

    renderWithClient(
      <SpanContentSection
        traceId="s-1"
        span={traceSpan({ span_id: "call-1", has_content: true })}
      />,
    )

    expect(
      await screen.findByText(/cannot be read right now/),
    ).toBeInTheDocument()
  })

  describe("in the operator's deployment-wide view", () => {
    it("reads nothing until the operator breaks glass", async () => {
      const calls = serve({ operator: true, content: ARGUMENTS })

      renderWithClient(
        <SpanContentSection
          traceId="s-1"
          span={traceSpan({ span_id: "call-1", has_content: true })}
        />,
      )

      expect(
        await screen.findByRole("button", { name: "Break glass" }),
      ).toBeInTheDocument()
      expect(contentCalls(calls)).toEqual([])
    })

    it("requires a reason, then posts it and shows the content", async () => {
      const calls = serve({ operator: true, content: ARGUMENTS })

      renderWithClient(
        <SpanContentSection
          traceId="s-1"
          span={traceSpan({ span_id: "call-1", has_content: true })}
        />,
      )

      await userEvent.click(
        await screen.findByRole("button", { name: "Break glass" }),
      )
      const dialog = await screen.findByRole("dialog")
      expect(dialog).toHaveTextContent(/legal or safety request/)
      expect(dialog).toHaveTextContent(/shown to the workspace's admins/)
      const submit = screen.getByRole("button", { name: "Read content" })
      expect(submit).toBeDisabled()

      const reason = screen.getByRole("textbox", { name: /Reason/ })
      await userEvent.type(reason, "too short")
      expect(submit).toBeDisabled()
      await userEvent.type(reason, " for a court order")
      expect(submit).toBeEnabled()
      await userEvent.click(submit)

      expect(await screen.findByText("README.md")).toBeInTheDocument()
      const posted = contentCalls(calls)
      expect(posted).toHaveLength(1)
      expect(posted[0]).toMatchObject({
        url: `${API_ROOT}/traces/s-1/spans/call-1/content/break-glass`,
        method: "POST",
      })
      expect(JSON.parse(posted[0].body ?? "{}")).toEqual({
        reason: "too short for a court order",
      })
    })
  })
})
