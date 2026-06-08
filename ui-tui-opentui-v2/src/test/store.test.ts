/**
 * Store test (spec v4 §5 Layer 3). Pure data behavior of the reducer: skin →
 * theme, LRU dedup, hydrate-while-buffering (Phase 1); and the Phase 2b ordered
 * `parts[]` model — text/tool interleave in one turn, tool start↔complete matched
 * by id and updated IN PLACE, `{output,exit_code}` envelope stripped.
 */
import { describe, expect, test } from 'bun:test'

import { DEFAULT_THEME } from '../logic/theme.ts'
import { createSessionStore, type Message } from '../logic/store.ts'

describe('session store — theming / dedup / hydrate (Phase 1)', () => {
  test('gateway.ready{skin} re-themes; default before', () => {
    const store = createSessionStore()
    expect(store.state.theme.brand.name).toBe(DEFAULT_THEME.brand.name)
    store.apply({
      type: 'gateway.ready',
      payload: { skin: { branding: { agent_name: 'Zephyr' }, colors: { ui_primary: '#123456' } } }
    })
    expect(store.state.ready).toBe(true)
    expect(store.state.theme.brand.name).toBe('Zephyr')
    expect(store.state.theme.color.primary).toBe('#123456')
  })

  test('skin.changed updates the theme live', () => {
    const store = createSessionStore()
    store.apply({ type: 'skin.changed', payload: { branding: { agent_name: 'Aurora' } } })
    expect(store.state.theme.brand.name).toBe('Aurora')
  })

  test('LRU dedup: duplicate(id) returns false once, true after', () => {
    const store = createSessionStore()
    expect(store.duplicate('evt-1')).toBe(false)
    expect(store.duplicate('evt-1')).toBe(true)
    expect(store.duplicate(undefined)).toBe(false) // no id → never deduped
  })

  test('hydrate replaces history, then replays events buffered mid-hydrate', () => {
    const store = createSessionStore()
    const snapshot: Message[] = [
      { role: 'user', text: 'old q' },
      { role: 'assistant', text: 'old a' }
    ]
    // Simulate a live event arriving DURING hydrate by emitting inside loadSnapshot.
    let emittedDuring = false
    store.hydrate(() => {
      if (!emittedDuring) {
        emittedDuring = true
        store.apply({ type: 'message.start' })
        store.apply({ type: 'message.delta', payload: { text: 'live!' } })
      }
      return snapshot
    })
    // snapshot (2) + the buffered live assistant turn (1) replayed after
    expect(store.state.messages.length).toBe(3)
    expect(store.state.messages[0]!.text).toBe('old q')
    // the streamed assistant text now lives in an ordered text part
    expect(store.state.messages[2]!.parts?.[0]).toMatchObject({ type: 'text', text: 'live!' })
  })
})

describe('session store — ordered parts (Phase 2b)', () => {
  test('interleaves text → tool → text as ordered parts in one assistant turn', () => {
    const store = createSessionStore()
    store.apply({ type: 'message.start' })
    store.apply({ type: 'message.delta', payload: { text: 'before ' } })
    store.apply({ type: 'tool.start', payload: { tool_id: 't1', name: 'terminal' } })
    // result_text is the {output,exit_code} JSON envelope — the store strips it.
    store.apply({
      type: 'tool.complete',
      payload: { tool_id: 't1', result_text: '{"output":"hello\\nworld","exit_code":0}' }
    })
    store.apply({ type: 'message.delta', payload: { text: 'after' } })
    store.apply({ type: 'message.complete' })

    const msg = store.state.messages.at(-1)!
    expect(msg.role).toBe('assistant')
    expect(msg.streaming).toBe(false)
    const parts = msg.parts ?? []
    expect(parts.map(p => p.type)).toEqual(['text', 'tool', 'text'])
    expect(parts[0]).toMatchObject({ type: 'text', text: 'before ' })
    expect(parts[2]).toMatchObject({ type: 'text', text: 'after' })
    const tool = parts[1]!
    if (tool.type === 'tool') {
      expect(tool.state).toBe('complete')
      expect(tool.name).toBe('terminal')
      expect(tool.resultText).toBe('hello\nworld') // envelope stripped
      expect(tool.lineCount).toBe(2)
    } else {
      throw new Error('expected a tool part at index 1')
    }
  })

  test('tool.complete updates the running tool part IN PLACE (not a new row)', () => {
    const store = createSessionStore()
    store.apply({ type: 'message.start' })
    store.apply({ type: 'tool.start', payload: { tool_id: 'x', name: 'read_file' } })
    expect(store.state.messages.at(-1)!.parts).toHaveLength(1)
    expect(store.state.messages.at(-1)!.parts![0]).toMatchObject({ type: 'tool', state: 'running', name: 'read_file' })

    store.apply({ type: 'tool.complete', payload: { tool_id: 'x', summary: 'read 42 lines' } })
    const parts = store.state.messages.at(-1)!.parts!
    expect(parts).toHaveLength(1) // updated in place — NOT appended as a separate row
    const tool = parts[0]!
    if (tool.type === 'tool') {
      expect(tool.state).toBe('complete')
      expect(tool.summary).toBe('read 42 lines')
    } else {
      throw new Error('expected a tool part')
    }
  })

  test('reasoning.delta accumulates into a reasoning part', () => {
    const store = createSessionStore()
    store.apply({ type: 'message.start' })
    store.apply({ type: 'reasoning.delta', payload: { text: 'thinking ' } })
    store.apply({ type: 'reasoning.delta', payload: { text: 'hard' } })
    const parts = store.state.messages.at(-1)!.parts ?? []
    expect(parts[0]).toMatchObject({ type: 'reasoning', text: 'thinking hard' })
  })

  test('thinking.delta (kaomoji face) → transient status, NOT a transcript part; complete clears it', () => {
    const store = createSessionStore()
    store.apply({ type: 'message.start' })
    store.apply({ type: 'thinking.delta', payload: { text: '(´･_･`) formulating...' } })
    expect(store.state.status).toBe('(´･_･`) formulating...')
    expect(store.state.messages.at(-1)!.parts ?? []).toHaveLength(0) // no reasoning row from the face
    store.apply({ type: 'message.delta', payload: { text: 'Hi!' } })
    store.apply({ type: 'message.complete' })
    expect(store.state.status).toBeUndefined() // cleared when the turn ends
    // only the real reply text part remains — the face never entered the transcript
    expect((store.state.messages.at(-1)!.parts ?? []).map(p => p.type)).toEqual(['text'])
  })

  test('status.update also drives the transient status line', () => {
    const store = createSessionStore()
    store.apply({ type: 'status.update', payload: { kind: 'tool', text: 'running terminal…' } })
    expect(store.state.status).toBe('running terminal…')
  })
})

describe('session store — blocking prompts (Phase 3)', () => {
  test('approval.request sets an approval prompt; clearPrompt clears it', () => {
    const store = createSessionStore()
    expect(store.state.prompt).toBeUndefined()
    store.apply({ type: 'approval.request', payload: { command: 'rm -rf /tmp/x', description: 'delete temp' } })
    expect(store.state.prompt).toMatchObject({ kind: 'approval', command: 'rm -rf /tmp/x', description: 'delete temp' })
    store.clearPrompt()
    expect(store.state.prompt).toBeUndefined()
  })

  test('clarify.request carries question + choices + request_id', () => {
    const store = createSessionStore()
    store.apply({ type: 'clarify.request', payload: { question: 'Which?', choices: ['a', 'b'], request_id: 'r1' } })
    const p = store.state.prompt
    expect(p).toMatchObject({ kind: 'clarify', question: 'Which?', requestId: 'r1' })
    if (p?.kind === 'clarify') expect(p.choices).toEqual(['a', 'b'])
  })

  test('clarify.request with null choices → free-text only', () => {
    const store = createSessionStore()
    store.apply({ type: 'clarify.request', payload: { question: 'Name?', choices: null, request_id: 'r2' } })
    const p = store.state.prompt
    if (p?.kind === 'clarify') expect(p.choices).toBeNull()
  })

  test('sudo.request + secret.request set masked prompts', () => {
    const store = createSessionStore()
    store.apply({ type: 'sudo.request', payload: { request_id: 's1' } })
    expect(store.state.prompt).toMatchObject({ kind: 'sudo', requestId: 's1' })
    store.apply({ type: 'secret.request', payload: { env_var: 'API_KEY', prompt: 'Enter key', request_id: 's2' } })
    expect(store.state.prompt).toMatchObject({ kind: 'secret', envVar: 'API_KEY', requestId: 's2' })
  })
})

describe('session store — subagents (Phase 5e agents dashboard)', () => {
  test('subagent.* events build + update a subagent by id', () => {
    const store = createSessionStore()
    store.apply({
      type: 'subagent.start',
      payload: { subagent_id: 'a1', goal: 'research X', model: 'haiku', depth: 1 }
    })
    expect(store.state.subagents).toHaveLength(1)
    expect(store.state.subagents[0]).toMatchObject({ id: 'a1', goal: 'research X', status: 'running', depth: 1 })

    store.apply({ type: 'subagent.tool', payload: { subagent_id: 'a1', tool_name: 'web_search' } })
    expect(store.state.subagents[0]).toMatchObject({ status: 'tool', lastTool: 'web_search' })

    store.apply({ type: 'subagent.complete', payload: { subagent_id: 'a1', summary: 'found it' } })
    expect(store.state.subagents).toHaveLength(1) // updated in place
    expect(store.state.subagents[0]).toMatchObject({ status: 'complete', summary: 'found it' })
  })

  test('clearTranscript also clears subagents', () => {
    const store = createSessionStore()
    store.apply({ type: 'subagent.start', payload: { subagent_id: 'a1', goal: 'g' } })
    store.clearTranscript()
    expect(store.state.subagents).toHaveLength(0)
  })
})

describe('session store — session chrome / status bar (item 14)', () => {
  test('session.info populates model/effort/cwd/branch and nested usage context', () => {
    const store = createSessionStore()
    store.apply({
      type: 'session.info',
      payload: {
        model: 'anthropic/claude-opus-4-8',
        reasoning_effort: 'high',
        fast: true,
        cwd: '/home/x/proj',
        branch: 'main',
        running: false,
        usage: { context_used: 42000, context_max: 200000, context_percent: 21 }
      }
    })
    const info = store.state.info
    expect(info.model).toBe('anthropic/claude-opus-4-8')
    expect(info.effort).toBe('high')
    expect(info.fast).toBe(true)
    expect(info.cwd).toBe('/home/x/proj')
    expect(info.branch).toBe('main')
    expect(info.contextPercent).toBe(21)
    expect(info.contextMax).toBe(200000)
  })

  test('message.start sets running, message.complete clears it + refreshes usage', () => {
    const store = createSessionStore()
    store.apply({ type: 'message.start' })
    expect(store.state.info.running).toBe(true)
    store.apply({ type: 'message.delta', payload: { text: 'hi' } })
    store.apply({ type: 'message.complete', payload: { usage: { context_percent: 33 } } })
    expect(store.state.info.running).toBe(false)
    expect(store.state.info.contextPercent).toBe(33)
  })

  test('applyInfo merges a session.create info patch without clobbering prior fields', () => {
    const store = createSessionStore()
    store.applyInfo({ model: 'gpt-5.4', cwd: '/tmp' })
    store.applyInfo({ branch: 'dev' }) // partial patch — model/cwd must survive
    expect(store.state.info).toMatchObject({ model: 'gpt-5.4', cwd: '/tmp', branch: 'dev' })
  })

  test('setHint sets/clears the transient composer hint (Ctrl+C again to quit — item 11)', () => {
    const store = createSessionStore()
    expect(store.state.hint).toBeUndefined()
    store.setHint('Ctrl+C again to quit')
    expect(store.state.hint).toBe('Ctrl+C again to quit')
    store.setHint(undefined)
    expect(store.state.hint).toBeUndefined()
  })
})

describe('session store — resume hydrate (Phase 4b)', () => {
  test('beginBuffer + commitSnapshot replaces history then replays events buffered across the resume', () => {
    const store = createSessionStore()
    store.beginBuffer()
    // a live event arrives DURING the (async) session.resume RPC
    store.apply({ type: 'message.start' })
    store.apply({ type: 'message.delta', payload: { text: 'live during resume' } })
    // the snapshot commits afterwards
    store.commitSnapshot([{ role: 'user', text: 'old question' }])
    expect(store.state.messages).toHaveLength(2) // snapshot(1) + the replayed assistant turn(1)
    expect(store.state.messages[0]).toMatchObject({ role: 'user', text: 'old question' })
    expect(store.state.messages[1]!.parts?.[0]).toMatchObject({ type: 'text', text: 'live during resume' })
  })
})
