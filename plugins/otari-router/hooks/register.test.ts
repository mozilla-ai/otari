import { expect, mock, test } from 'claude-code/testing'

// The two options the manifest requires, as a user gave them at install time.
const OPTIONS = { options: { otari_url: 'http://localhost:8010/', otari_api_key: 'tk-test' } }
// The same two options pointing at a gateway on another machine, over each transport.
const REMOTE_HTTP = { options: { otari_url: 'http://otari.example.com', otari_api_key: 'tk-test' } }
const REMOTE_HTTPS = { options: { otari_url: 'https://otari.example.com', otari_api_key: 'tk-test' } }
// A gateway on this machine by its loopback address, and a remote host whose name merely begins with 127.
const LOOPBACK_HTTP = { options: { otari_url: 'http://127.0.0.1:8010', otari_api_key: 'tk-test' } }
const REMOTE_HTTP_127_NAME = { options: { otari_url: 'http://127.example.com', otari_api_key: 'tk-test' } }

// The fields every spawn carries; each test adds `subagentType` and, where it
// matters, the caller's `model`.
const spawn = {
  tool_use_id: 'toolu_01',
  prompt: 'List the files under src/',
  description: 'List files',
  provider: { plugin: 'engine', tier: 'core' },
  parentModel: 'claude-opus-5',
} as const

// The engine's own resolution, standing in beneath the mod: the model it was
// handed, or the parent's when none was, and an id for the subagent it started.
const resolve = ($: unknown, e: { model?: string; parentModel: string }) => ({
  model: e.model ?? e.parentModel,
  agentId: 'agent-1',
})

const modelOf = (spawned: unknown) => (spawned as { model?: string }).model

type Question = { url: string; init?: { method?: string; headers?: Record<string, string>; body?: string } }
type Answer = { status: number; ok: boolean; text: string }

const HAIKU: Answer = { status: 200, ok: true, text: JSON.stringify({ model: 'haiku', reason: 'mock' }) }

// The world beneath the mod: a held clock, a session id, a transcript, and an
// Otari that gives the same answer to every question. A stand-in for a mods
// API call answers `{ value }`, as the kit requires.
const world = (on: Parameters<Parameters<typeof test>[2]>[1], answer: Answer = HAIKU) => {
  const questions: Question[] = []
  const logs: string[] = []
  mock.clock(on)
  on('session.id', () => ({ value: 'session-1' }))
  on('ui.log', ($, e) => {
    logs.push(e.text)
    return { value: undefined }
  })
  on('http.fetch', ($, e) => {
    questions.push(e)
    return { value: { headers: {}, ...answer } }
  })
  on('agent.spawn', resolve)
  return { questions, logs }
}

test("Otari's model is used, even when the caller asked for another", OPTIONS, async ($, on) => {
  const { logs } = world(on)
  const spawned = await $.agent.spawn({ ...spawn, subagentType: 'Explore', model: 'sonnet' })
  expect(modelOf(spawned)).toBe('haiku')
  expect(logs).toEqual(["Explore agent-1 starts on haiku, Otari's choice (mock)"])
})

test('the question goes to the configured gateway with the configured key', OPTIONS, async ($, on) => {
  const { questions } = world(on)
  await $.agent.spawn({ ...spawn, subagentType: 'general-purpose' })
  expect(questions.length).toBe(1)
  const [q] = questions
  expect(q.url).toBe('http://localhost:8010/api/v1/routing/recommend')
  expect(q.init?.method).toBe('POST')
  expect(q.init?.headers?.authorization).toBe('Bearer tk-test')
  expect(JSON.parse(q.init?.body ?? '{}')).toEqual({
    harness: 'claude-code',
    session_id: 'session-1',
    tool_use_id: 'toolu_01',
    agent_type: 'general-purpose',
    description: 'List files',
    prompt: 'List the files under src/',
    parent_model: 'claude-opus-5',
    requested_model: null,
  })
})

test('when Otari fails, the subagent still starts on its usual model', OPTIONS, async ($, on) => {
  const { logs } = world(on, { status: 503, ok: false, text: '' })
  const spawned = await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  expect(modelOf(spawned)).toBe('claude-opus-5')
  expect(logs).toEqual(['otari-router: Otari did not decide (HTTP 503); Explore starts on its usual model'])
})

test('when Otari names no model, the subagent still starts on its usual model', OPTIONS, async ($, on) => {
  const { logs } = world(on, { status: 200, ok: true, text: JSON.stringify({ reason: 'no idea' }) })
  const spawned = await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  expect(modelOf(spawned)).toBe('claude-opus-5')
  expect(logs).toEqual(['otari-router: Otari did not decide (the answer names no model); Explore starts on its usual model'])
})

test('a fork is not asked about and keeps the parent model', OPTIONS, async ($, on) => {
  const { questions } = world(on)
  const spawned = await $.agent.spawn({ ...spawn, subagentType: 'fork' })
  expect(questions.length).toBe(0)
  expect(modelOf(spawned)).toBe('claude-opus-5')
})

test('a remote gateway over plain http is never sent the key', REMOTE_HTTP, async ($, on) => {
  const { questions, logs } = world(on)
  const spawned = await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  expect(questions.length).toBe(0)
  expect(modelOf(spawned)).toBe('claude-opus-5')
  expect(logs).toEqual([
    'otari-router: Otari did not decide (otari_url must use https unless it points at this machine); Explore starts on its usual model',
  ])
})

test('a remote host whose name begins with 127. is never sent the key', REMOTE_HTTP_127_NAME, async ($, on) => {
  const { questions, logs } = world(on)
  const spawned = await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  expect(questions.length).toBe(0)
  expect(modelOf(spawned)).toBe('claude-opus-5')
  expect(logs).toEqual([
    'otari-router: Otari did not decide (otari_url must use https unless it points at this machine); Explore starts on its usual model',
  ])
})

test('a gateway at the loopback address over plain http is asked', LOOPBACK_HTTP, async ($, on) => {
  const { questions } = world(on)
  const spawned = await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  expect(questions.length).toBe(1)
  expect(questions[0].url).toBe('http://127.0.0.1:8010/api/v1/routing/recommend')
  expect(modelOf(spawned)).toBe('haiku')
})

test('a remote gateway over https is asked', REMOTE_HTTPS, async ($, on) => {
  const { questions } = world(on)
  const spawned = await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  expect(questions.length).toBe(1)
  expect(questions[0].url).toBe('https://otari.example.com/api/v1/routing/recommend')
  expect(modelOf(spawned)).toBe('haiku')
})
