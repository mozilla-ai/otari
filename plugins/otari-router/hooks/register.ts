import type { Register } from 'claude-code'

// Otari decides every subagent's model. This module only carries the question
// there and the answer back. Where Otari listens and the key the request
// carries are the plugin's two options; the manifest requires both, so
// Claude Code collects them at install time and loads nothing without them.
const RECOMMEND_PATH = '/api/v1/routing/recommend'
const DEADLINE_MS = 3000

type Recommendation = { model: string; reason?: string }

// The key travels in a header, so plain http is for a gateway on this machine
// only. A remote http URL is refused at spawn time rather than at load: a
// module that fails to load says so in the debug log alone, while a refused
// spawn leaves a line in the transcript until the option is fixed.
const plainHttpToRemoteHost = (base: string): boolean => {
  let url: URL
  try {
    url = new URL(base)
  } catch {
    return false // not a URL at all; the fetch reports that itself
  }
  const host = url.hostname
  const local = host === 'localhost' || host.endsWith('.localhost') || host === '[::1]' || isLoopbackV4(host)
  return url.protocol === 'http:' && !local
}

// The URL parser has already turned an IPv4 host into its dotted quad, so a
// loopback address is four numeric labels led by 127. A name such as
// 127.example.com keeps its last label and is a remote host.
const isLoopbackV4 = (host: string): boolean => {
  const labels = host.split('.')
  return labels.length === 4 && labels[0] === '127' && labels.every((label) => /^\d{1,3}$/.test(label) && Number(label) <= 255)
}

export const register: Register = (on, options) => {
  const base = String(options.otari_url).replace(/\/$/, '')
  const key = String(options.otari_api_key)
  const transportProblem = plainHttpToRemoteHost(base) ? 'otari_url must use https unless it points at this machine' : undefined

  on('agent.spawn', async ($, e, next) => {
    // Claude Code ignores `model` for a fork, which always inherits the
    // parent's model along with its context. Asking would change nothing.
    if (e.subagentType === 'fork') return next(e)
    if (transportProblem !== undefined) throw new Error(transportProblem)

    const question = $.http.fetch(`${base}${RECOMMEND_PATH}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${key}` },
      body: JSON.stringify({
        harness: 'claude-code',
        session_id: await $.session.id(),
        tool_use_id: e.tool_use_id,
        agent_type: e.subagentType,
        description: e.description,
        prompt: e.prompt,
        parent_model: e.parentModel,
        requested_model: e.model ?? null,
      }),
    })
    const deadline = $.clock.sleep(DEADLINE_MS).then(() => {
      throw new Error(`no answer within ${DEADLINE_MS} ms`)
    })
    const response = await Promise.race([question, deadline])
    if (!response.ok) throw new Error(`HTTP ${response.status}`)

    const recommended = JSON.parse(response.text) as Partial<Recommendation>
    if (typeof recommended.model !== 'string' || recommended.model === '') throw new Error('the answer names no model')

    const spawned = await next({ ...e, model: recommended.model })
    if (spawned.deny !== undefined) return spawned
    const why = recommended.reason === undefined ? '' : ` (${recommended.reason})`
    $.ui.log(`${e.subagentType} ${spawned.agentId ?? ''} starts on ${spawned.model}, Otari's choice${why}`)
    return spawned
  }).catch(($, e, next) => {
    // Otari did not decide: unreachable, slow, an error, or a bad answer. The
    // subagent still starts, on the model it would have had without this mod,
    // and the transcript says so. To refuse the spawn instead, return
    // `{ deny: reason }` here when `next.called` is false.
    if (!next.called) {
      $.ui.log(`otari-router: Otari did not decide (${next.error.message}); ${e.subagentType} starts on its usual model`)
    }
    return next(e)
  })

  on('turn.complete', async ($, e, next) => {
    if (e.agentId !== undefined && e.usage !== undefined) {
      const u = e.usage
      const seconds = Math.round(e.durationMs / 1000)
      $.ui.log(
        `subagent ${e.agentId} ran on ${u.model}: in ${u.input_tokens}, out ${u.output_tokens}, ` +
          `cache read ${u.cache_read_input_tokens}, cache write ${u.cache_creation_input_tokens}, ${seconds}s` +
          (e.isAborted ? ', aborted' : ''),
      )
    }
    return next(e)
  })
}
