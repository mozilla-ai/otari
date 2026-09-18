import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ProvidersPage } from "@/features/providers/ProvidersPage"
import { API_ROOT } from "@/shared/api/client"
import { mockApi, renderPage, storedProvider } from "@/tests/providersPage"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ProvidersPage add form", () => {
  it("adds a custom provider and posts a write-only key, never rendering it", async () => {
    const fetchMock = mockApi({ meta: [], stored: [] })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add your first provider" }),
    )
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.type(screen.getByLabelText("Name"), "my-llm")
    await user.type(screen.getByLabelText("API base"), "http://box:8000/v1")
    const apiKey = screen.getByLabelText("API key (optional)")
    expect(apiKey).toHaveAttribute("type", "password")
    await user.type(apiKey, "sk-live-9999")
    await user.click(screen.getByRole("button", { name: "Add provider" }))

    const post = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).endsWith(`${API_ROOT}/provider-credentials`) &&
        (init?.method ?? "") === "POST",
    )
    expect(post).toBeDefined()
    expect(JSON.parse(String(post?.[1]?.body))).toMatchObject({
      instance: "my-llm",
      provider_type: "openai-compatible",
      api_base: "http://box:8000/v1",
      api_key: "sk-live-9999",
    })

    // After the round trip the row shows the redacted key, never the plaintext.
    expect(await screen.findByText("••••9999")).toBeInTheDocument()
    expect(document.body.textContent).not.toContain("sk-live-9999")
  })

  it("offers the known-provider picker with an Advanced disclosure", async () => {
    mockApi({
      stored: [storedProvider("anthropic", "0000")],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••0000")
    await user.click(screen.getByRole("button", { name: "Add provider" }))

    // Known provider is the default tab: a provider picker plus a collapsed Advanced section.
    expect(screen.getByPlaceholderText("Search providers…")).toBeInTheDocument()
    expect(
      screen.getByText("Advanced (API base, rename, client options)"),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText("API base")).not.toBeInTheDocument()
    expect(
      screen.queryByLabelText("Client options (JSON)"),
    ).not.toBeInTheDocument()
  })

  it("asks a known provider for its own fields and sends them in client_args", async () => {
    const fetchMock = mockApi({
      stored: [storedProvider("anthropic", "0000")],
      catalog: [
        {
          id: "bedrock",
          name: "Bedrock",
          env_key: "AWS_BEARER_TOKEN_BEDROCK",
          default_api_base: null,
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••0000")
    await user.click(screen.getByRole("button", { name: "Add provider" }))
    // Typed rather than clicked: the picker is this dialog's autofocused first
    // field, so it opens its list on input (`menuTrigger`), and the chevron
    // beside it is the other way to the full catalog.
    await user.type(screen.getByPlaceholderText("Search providers…"), "Bed")
    await user.click(await screen.findByRole("option", { name: "Bedrock" }))

    const add = within(screen.getByRole("dialog")).getByRole("button", {
      name: "Add provider",
    })
    await user.type(screen.getByLabelText(/Bedrock API key/), "bearer-token")
    // The region is required and outside Advanced, so nothing that blocks the
    // submit is hidden behind a collapsed section.
    expect(add).toBeDisabled()
    await user.type(
      screen.getByRole("textbox", { name: /AWS region/ }),
      "eu-central-1",
    )
    await waitFor(() => expect(add).toBeEnabled())
    await user.click(add)

    const post = await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([u, init]) =>
          String(u).endsWith(`${API_ROOT}/provider-credentials`) &&
          (init?.method ?? "") === "POST",
      )
      expect(call).toBeDefined()
      return call!
    })
    expect(JSON.parse(String(post[1]?.body))).toMatchObject({
      instance: "bedrock",
      api_key: "bearer-token",
      client_args: { region_name: "eu-central-1" },
    })
  })

  it("fetches provider autofill hints lazily, only after one is selected", async () => {
    const fetchMock = mockApi({
      stored: [storedProvider("anthropic", "0000")],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••0000")
    await user.click(screen.getByRole("button", { name: "Add provider" }))

    const detailCalls = () =>
      fetchMock.mock.calls.filter(([u]) =>
        String(u).includes(`${API_ROOT}/providers/catalog/openai`),
      )

    // Opening the picker lists providers (id + name) but must not import any
    // provider SDK: no per-provider detail call until one is chosen (issue #365).
    expect(detailCalls()).toHaveLength(0)

    await user.type(screen.getByPlaceholderText("Search providers…"), "OpenAI")
    await user.click(await screen.findByRole("option", { name: /OpenAI/ }))

    // Selecting the provider triggers exactly the one detail fetch it needs.
    await screen.findByText(/OpenAI's endpoint is built in/)
    expect(detailCalls().length).toBeGreaterThan(0)
  })

  it("opens with the provider picker focused and its list closed", async () => {
    // feedback.md: the first field takes `autoFocus`. The picker opens its list
    // on focus everywhere else, which for an autofocused instance means the
    // catalog is down over the form before anything has been asked (measured:
    // `aria-expanded="true"` and a rendered listbox on mount), so this instance
    // opens on input instead.
    mockApi({ meta: [], stored: [] })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider" }),
    )

    const picker = screen.getByPlaceholderText("Search providers…")
    expect(picker).toHaveFocus()
    expect(picker).toHaveAttribute("aria-expanded", "false")
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument()
  })

  it("offers a fresh draft from each of the two openers", async () => {
    // Nothing unmounts the form, so the remount on the way in is the only thing
    // that clears it, and what it clears here includes a pasted provider key: a
    // still-enabled submit over a surviving secret re-POSTs the credential.
    // Both openers have to bump the counter, and this page is the only one in
    // the stack with two.
    mockApi({
      meta: [],
      stored: [],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    // Opener one: the first-run panel.
    await user.click(
      await screen.findByRole("button", { name: "Add your first provider" }),
    )
    await user.type(screen.getByPlaceholderText("Search providers…"), "OpenAI")
    await user.click(await screen.findByRole("option", { name: /OpenAI/ }))
    await user.type(screen.getByLabelText("API key"), "sk-live-aaaa")

    // Out through the guard, which is the only way out of a dirty form.
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Discard" }))

    // Opener two: the heading's action.
    await user.click(screen.getByRole("button", { name: "Add provider" }))
    expect(screen.getByPlaceholderText("Search providers…")).toHaveValue("")
    expect(screen.getByLabelText("API key")).toHaveValue("")

    // And back through the first opener, so neither is green on the other's
    // counter bump.
    await user.type(screen.getByLabelText("API key"), "sk-live-bbbb")
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Discard" }))
    await user.click(
      screen.getByRole("button", { name: "Add your first provider" }),
    )
    expect(screen.getByLabelText("API key")).toHaveValue("")
  })

  it("keeps the primary pressable while a create is in flight", async () => {
    // feedback.md: a submit in flight is working rather than refused, so it
    // keeps its fill and blocks its own press. `isSubmitDisabled` is the
    // product's one disabled treatment, and drawing it under the spinner says
    // the form rejected the input.
    let release = () => {}
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    const fetchMock = mockApi({ meta: [], stored: [], createGate: gate })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider" }),
    )
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.type(screen.getByLabelText(/^Name/), "my-local-llm")
    await user.type(
      screen.getByLabelText("API base"),
      "http://localhost:8000/v1",
    )

    const dialog = screen.getByRole("dialog")
    const submit = within(dialog).getByRole("button", { name: "Add provider" })
    await user.click(submit)

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.filter(
          ([url, init]) =>
            String(url).endsWith(`${API_ROOT}/provider-credentials`) &&
            (init as RequestInit | undefined)?.method === "POST",
        ).length,
      ).toBe(1),
    )
    expect(submit).toBeEnabled()
    // Pressing again while it runs sends nothing: the guard inside `submit` is
    // where `isPending` belongs.
    await user.click(submit)
    expect(
      fetchMock.mock.calls.filter(
        ([url, init]) =>
          String(url).endsWith(`${API_ROOT}/provider-credentials`) &&
          (init as RequestInit | undefined)?.method === "POST",
      ).length,
    ).toBe(1)

    release()
  })

  it("guards an Advanced rename on the known tab, with no provider chosen", async () => {
    // The guard reads one snapshot of the whole draft. It used to read the
    // provider and the key only, so a rename, an API base, a client_args blob
    // and every typed credential went on Escape with nothing asked.
    mockApi({ meta: [], stored: [] })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider" }),
    )
    await user.click(screen.getByRole("button", { name: /^Advanced/ }))
    await user.type(screen.getByLabelText(/^Name/), "openai-eu")

    await user.keyboard("{Escape}")

    expect(
      await screen.findByRole("button", { name: "Discard" }),
    ).toBeInTheDocument()
  })

  it("keeps a tab's draft, its guard and a hand-edited API base across a switch", async () => {
    // The draft is lifted above the tab components so it survives a switch, but
    // the two things scoped to it stayed below: useDirtySnapshot seeds on mount,
    // so it reseeded against filled values and Escape closed with nothing asked,
    // and the api_base effect refired from cache and overwrote a hand-edited
    // value with the provider's default.
    mockApi({
      stored: [],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider" }),
    )
    await user.type(screen.getByPlaceholderText("Search providers…"), "Open")
    await user.click(await screen.findByRole("option", { name: "OpenAI" }))
    await user.type(screen.getByLabelText(/API key/), "sk-live-aaaa")

    // An operator routing through a proxy replaces the seeded default. The
    // disclosure appears once the provider's detail lands, which is also what
    // seeds the base, so wait for it rather than racing it.
    await user.click(await screen.findByRole("button", { name: /^Advanced/ }))
    const apiBase = screen.getByLabelText("API base")
    await user.clear(apiBase)
    await user.type(apiBase, "https://proxy.internal/v1")

    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.click(screen.getByRole("button", { name: "Known provider" }))

    expect(screen.getByLabelText(/API key/)).toHaveValue("sk-live-aaaa")
    // Advanced stays open: its disclosure is lifted with the rest of the draft,
    // so the base is on screen without reopening it.
    expect(await screen.findByLabelText("API base")).toHaveValue(
      "https://proxy.internal/v1",
    )

    // And the guard still arms: the draft survived, so leaving must ask.
    await user.keyboard("{Escape}")
    expect(
      await screen.findByRole("button", { name: "Discard" }),
    ).toBeInTheDocument()
  })

  it("reseeds the API base when the same provider is picked again after clearing", async () => {
    // Emptying the picker clears the base, and it clears the marker the seeding
    // effect keys on: the marker records which provider the base in the field
    // came from, so one left behind reads as already seeded and the field stays
    // blank on picking that provider back.
    mockApi({
      stored: [],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider" }),
    )
    const picker = screen.getByPlaceholderText("Search providers…")
    await user.type(picker, "Open")
    await user.click(await screen.findByRole("option", { name: "OpenAI" }))

    await user.click(await screen.findByRole("button", { name: /^Advanced/ }))
    expect(await screen.findByLabelText("API base")).toHaveValue(
      "https://api.openai.com/v1",
    )

    // Emptying the field and leaving it is what drops the selection: the picker
    // takes no custom value, so react-aria clears it rather than keeping a
    // provider the input no longer names.
    await user.clear(picker)
    await user.tab()
    await user.type(picker, "Open")
    await user.click(await screen.findByRole("option", { name: "OpenAI" }))

    expect(await screen.findByLabelText("API base")).toHaveValue(
      "https://api.openai.com/v1",
    )
  })

  it("keeps an API base typed before the provider's hints land", async () => {
    // The seeding effect fires when the detail arrives, which can be after the
    // operator has already opened Advanced and typed. Choosing the provider is
    // what blanks the base, so anything in the field by then was typed here and
    // outranks the built-in default.
    let landHints = () => {}
    const detailGate = new Promise<void>((resolve) => {
      landHints = resolve
    })
    mockApi({
      stored: [],
      detailGate,
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider" }),
    )
    await user.type(screen.getByPlaceholderText("Search providers…"), "Open")
    await user.click(await screen.findByRole("option", { name: "OpenAI" }))

    // The disclosure does not wait on the detail, so Advanced opens and takes a
    // value while the request is still in flight.
    await user.click(screen.getByRole("button", { name: /^Advanced/ }))
    await user.type(
      screen.getByLabelText("API base"),
      "https://proxy.internal/v1",
    )

    landHints()
    // The hints land and drive the rest of the form, which is how the test knows
    // the seeding effect has had its chance to run.
    expect(await screen.findByText(/just add your key/)).toBeInTheDocument()
    expect(screen.getByLabelText("API base")).toHaveValue(
      "https://proxy.internal/v1",
    )
  })

  it("guards client options typed on the custom tab, with nothing else filled", async () => {
    mockApi({ meta: [], stored: [] })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add provider" }),
    )
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.type(screen.getByLabelText(/Client options/), '{{"timeout": 30}')

    await user.keyboard("{Escape}")

    expect(
      await screen.findByRole("button", { name: "Discard" }),
    ).toBeInTheDocument()
  })

  it("keeps Add disabled for a key-requiring provider until a key is entered", async () => {
    mockApi({
      stored: [storedProvider("anthropic", "0000")],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••0000")
    await user.click(screen.getByRole("button", { name: "Add provider" }))

    await user.type(screen.getByPlaceholderText("Search providers…"), "OpenAI")
    await user.click(await screen.findByRole("option", { name: /OpenAI/ }))
    // No Escape here any more. Picking an option closes the combo box's own
    // popover, so the keystroke would reach the dialog instead and arm its
    // unsaved-changes guard, which swaps the submit out of the footer.

    const submit = within(screen.getByRole("dialog")).getByRole("button", {
      name: "Add provider",
    })
    expect(submit).toBeDisabled()

    await user.type(screen.getByLabelText("API key"), "sk-live-xxxx")
    expect(submit).toBeEnabled()
  })

  it("lets a key-requiring provider submit without a key when its env var is already set", async () => {
    const fetchMock = mockApi({
      stored: [storedProvider("anthropic", "0000")],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: true,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••0000")
    await user.click(screen.getByRole("button", { name: "Add provider" }))

    await user.type(screen.getByPlaceholderText("Search providers…"), "OpenAI")
    await user.click(await screen.findByRole("option", { name: /OpenAI/ }))
    // No Escape here any more. Picking an option closes the combo box's own
    // popover, so the keystroke would reach the dialog instead and arm its
    // unsaved-changes guard, which swaps the submit out of the footer.

    // The field is optional and the copy explains the env fallback. The hint
    // arrives once the selected provider's detail loads, so wait for it.
    await screen.findByText(/OPENAI_API_KEY is set on the server/)
    expect(screen.getByLabelText("API key (optional)")).toBeInTheDocument()

    // Submit with no key: the server stores none and any-llm reads OPENAI_API_KEY.
    const submit = within(screen.getByRole("dialog")).getByRole("button", {
      name: "Add provider",
    })
    expect(submit).toBeEnabled()
    await user.click(submit)

    const post = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).endsWith(`${API_ROOT}/provider-credentials`) &&
        (init?.method ?? "") === "POST",
    )
    expect(post).toBeDefined()
    expect(JSON.parse(String(post?.[1]?.body))).toMatchObject({
      instance: "openai",
      api_key: null,
    })
  })
})
