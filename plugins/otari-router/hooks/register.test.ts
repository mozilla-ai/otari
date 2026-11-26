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
// The answer of a gateway that records its decisions.
const RECORDED: Answer = {
  status: 200,
  ok: true,
  text: JSON.stringify({ model: 'haiku', reason: 'mock', decision_id: 'decision-1' }),
}

const usageOn = (model: string, output_tokens: number) => ({
  model,
  input_tokens: 10,
  output_tokens,
  cache_read_input_tokens: 300,
  cache_creation_input_tokens: 40,
})

// The end of one turn of the subagent `resolve` started.
const turnEnd = {
  answer: 'done',
  durationMs: 2000,
  isAborted: false,
  turnId: 'turn-1',
  reason: 'answer',
  agentId: 'agent-1',
  usage: usageOn('claude-haiku-5', 50),
} as const

// The world beneath the mod: a held clock, a session id, a transcript, and an
// Otari that gives the same answer to every question. A stand-in for a mods
// API call answers `{ value }`, as the kit requires.
const world = (on: Parameters<Parameters<typeof test>[2]>[1], answer: Answer = HAIKU, outcome?: () => Promise<Answer>) => {
  const questions: Question[] = []
  const logs: string[] = []
  const clock = mock.clock(on)
  on('session.id', () => ({ value: 'session-1' }))
  on('ui.log', ($, e) => {
    logs.push(e.text)
    return { value: undefined }
  })
  on('http.fetch', ($, e) => {
    questions.push(e)
    if (outcome !== undefined && e.url.endsWith('/outcome')) return outcome().then((value) => ({ value: { headers: {}, ...value } }))
    return { value: { headers: {}, ...answer } }
  })
  on('agent.spawn', resolve)
  on('turn.complete', ($, e) => ({ text: e.answer }))
  return { questions, logs, clock }
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

test("a routed subagent's token counts go back to Otari under its decision", OPTIONS, async ($, on) => {
  const { questions } = world(on, RECORDED)
  // The engine's totals for the subagent: an earlier turn on another model, and this one.
  on('agent.list', () => ({
    value: [{ id: 'agent-1', usage: [usageOn('claude-haiku-5', 50), usageOn('claude-sonnet-5', 7)] }],
  }))
  await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  await $.turn.complete(turnEnd)
  expect(questions.length).toBe(2)
  const report = questions[1]
  expect(report.url).toBe('http://localhost:8010/api/v1/routing/decisions/decision-1/outcome')
  expect(report.init?.method).toBe('PUT')
  expect(report.init?.headers?.authorization).toBe('Bearer tk-test')
  expect(JSON.parse(report.init?.body ?? '{}')).toEqual({
    usage: [
      { model: 'claude-haiku-5', input_tokens: 10, output_tokens: 50, cache_read_tokens: 300, cache_write_tokens: 40 },
      { model: 'claude-sonnet-5', input_tokens: 10, output_tokens: 7, cache_read_tokens: 300, cache_write_tokens: 40 },
    ],
  })
})

test("without the engine's totals, the turn's own counts are sent", OPTIONS, async ($, on) => {
  const { questions } = world(on, RECORDED)
  on('agent.list', () => ({ value: [] }))
  await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  await $.turn.complete(turnEnd)
  expect(JSON.parse(questions[1].init?.body ?? '{}')).toEqual({
    usage: [{ model: 'claude-haiku-5', input_tokens: 10, output_tokens: 50, cache_read_tokens: 300, cache_write_tokens: 40 }],
  })
})

test('a subagent Otari kept no decision for is not reported', OPTIONS, async ($, on) => {
  const { questions } = world(on)
  on('agent.list', () => ({ value: [] }))
  await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  await $.turn.complete(turnEnd)
  expect(questions.length).toBe(1)
})

test("the main loop's turn is not reported", OPTIONS, async ($, on) => {
  const { questions } = world(on, RECORDED)
  on('agent.list', () => ({ value: [] }))
  await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  const { agentId: _, ...mainTurn } = turnEnd
  await $.turn.complete(mainTurn)
  expect(questions.length).toBe(1)
})

test('a report Otari does not take leaves the turn as it was', OPTIONS, async ($, on) => {
  const { questions } = world(on, RECORDED)
  on('agent.list', () => {
    throw new Error('no roster')
  })
  await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  const ended = await $.turn.complete(turnEnd)
  expect((ended as { text?: string }).text).toBe('done')
  expect(questions.length).toBe(1)
})

test("a missing roster entry sends the totals of every turn, not the last turn's", OPTIONS, async ($, on) => {
  const { questions } = world(on, RECORDED)
  on('agent.list', () => ({ value: [] }))
  await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  await $.turn.complete(turnEnd)
  await $.turn.complete({ ...turnEnd, turnId: 'turn-2', usage: usageOn('claude-haiku-5', 7) })
  await $.turn.complete({ ...turnEnd, turnId: 'turn-3', usage: usageOn('claude-sonnet-5', 9) })
  expect(JSON.parse(questions[1].init?.body ?? '{}')).toEqual({
    usage: [{ model: 'claude-haiku-5', input_tokens: 10, output_tokens: 50, cache_read_tokens: 300, cache_write_tokens: 40 }],
  })
  expect(JSON.parse(questions[2].init?.body ?? '{}')).toEqual({
    usage: [{ model: 'claude-haiku-5', input_tokens: 20, output_tokens: 57, cache_read_tokens: 600, cache_write_tokens: 80 }],
  })
  expect(JSON.parse(questions[3].init?.body ?? '{}')).toEqual({
    usage: [
      { model: 'claude-haiku-5', input_tokens: 20, output_tokens: 57, cache_read_tokens: 600, cache_write_tokens: 80 },
      { model: 'claude-sonnet-5', input_tokens: 10, output_tokens: 9, cache_read_tokens: 300, cache_write_tokens: 40 },
    ],
  })
})

test('a roster entry with no usage falls back to the totals kept here', OPTIONS, async ($, on) => {
  const { questions } = world(on, RECORDED)
  on('agent.list', () => ({ value: [{ id: 'agent-1', usage: [] }] }))
  await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  await $.turn.complete(turnEnd)
  expect(questions.length).toBe(2)
  expect(JSON.parse(questions[1].init?.body ?? '{}')).toEqual({
    usage: [{ model: 'claude-haiku-5', input_tokens: 10, output_tokens: 50, cache_read_tokens: 300, cache_write_tokens: 40 }],
  })
})

test('a report that fails after the deadline surfaces nowhere', OPTIONS, async ($, on) => {
  let refuse: (reason: Error) => void = () => {}
  const slow = new Promise<Answer>((_, reject) => {
    refuse = reject
  })
  const { clock } = world(on, RECORDED, () => slow)
  on('agent.list', () => ({ value: [] }))
  await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  const ending = $.turn.complete(turnEnd)
  await clock.settle()
  await clock.advance(2000)
  expect(((await ending) as { text?: string }).text).toBe('done')
  // An unhandled rejection fails the run, so reaching the end is the assertion.
  refuse(new Error('connection reset'))
  await new Promise((resolve) => setTimeout(resolve, 10))
})

test('a roster that never answers does not hold the turn past the deadline', OPTIONS, async ($, on) => {
  const { questions, clock } = world(on, RECORDED)
  on('agent.list', () => new Promise(() => {}))
  await $.agent.spawn({ ...spawn, subagentType: 'Explore' })
  const ending = $.turn.complete(turnEnd)
  await clock.settle()
  await clock.advance(2000)
  expect(((await ending) as { text?: string }).text).toBe('done')
  expect(questions.length).toBe(1)
})
