/**
 * Session/message store — the SOLID side (spec v4 §1, §7). Plain `createStore`
 * + an `apply(event)` reducer, à la opencode `context/sync-v2.tsx`. NOT Effect.
 * The boundary calls `apply` with already-decoded `GatewayEvent`s via
 * GatewayService.subscribe.
 *
 * Phase 2b: an assistant turn is ONE ordered `parts[]` of a discriminated union
 * (text / reasoning / tool), so tool calls render INLINE between text blocks
 * instead of dumped as separate rows below (§7 — the "dump-below" bug). Tools are
 * matched start↔complete by `tool_id`; `tool.complete` updates that part IN PLACE.
 * User/system rows stay flat `text` (no parts). Carried from Phase 1: streaming
 * concat (prefer `payload.text`), skin→theme, LRU dedup, hydrate-while-buffering.
 */
import { createStore, produce } from 'solid-js/store'

import type { GatewayEvent, GatewaySkinDecoded } from '../boundary/schema/GatewayEvent.ts'
import { stripToolEnvelope } from './toolOutput.ts'
import { DEFAULT_THEME, type Theme, themeFromSkin } from './theme.ts'

/** A tool call inside an assistant turn (matched start↔complete by `id`=tool_id). */
export interface ToolPartState {
  type: 'tool'
  id: string
  name: string
  state: 'running' | 'complete'
  /** Envelope-stripped output (multi-line → block render; the view caps it). */
  resultText?: string
  /** Short one-line status when there's no substantial output. */
  summary?: string
  error?: string
  lineCount?: number
}

/** One ordered piece of an assistant turn (§7). */
export type Part =
  | { type: 'text'; id: string; text: string }
  | { type: 'reasoning'; id: string; text: string }
  | ToolPartState

export interface Message {
  readonly role: 'user' | 'assistant' | 'system'
  /** Flat body for user/system rows (and settled/resumed assistant rows). */
  text: string
  /** Ordered parts for a live assistant turn; absent for user/system. */
  parts?: Part[]
  streaming?: boolean
}

/**
 * A BLOCKING interactive request from the agent (spec §8 #6 — unhandled = deadlock).
 * Each is answered via the matching `*.respond` RPC; Esc/Ctrl+C sends deny/empty.
 */
export type ActivePrompt =
  | { kind: 'clarify'; question: string; choices: string[] | null; requestId: string }
  | { kind: 'approval'; command: string; description: string }
  | { kind: 'sudo'; requestId: string }
  | { kind: 'secret'; envVar: string; prompt: string; requestId: string }
  // local (non-gateway) Y/N confirm — e.g. /clear, /new (spec §2a)
  | { kind: 'confirm'; message: string; onConfirm: () => void }

/** A full-screen scrollable text viewer (long slash output: /status, /logs, …). */
export interface PagerState {
  title: string
  text: string
}

/** One row in the session switcher (from `session.list`). */
export interface SessionItem {
  id: string
  title: string
  preview: string
  messageCount: number
}

/** A row in a generic `<select>` picker (model picker, skills hub, …). */
export interface PickerItem {
  label: string
  description?: string
  value: string
}

/** An open generic picker overlay: a titled list whose pick runs `onPick(value)`. */
export interface PickerState {
  title: string
  items: PickerItem[]
  onPick: (value: string) => void
}

/** A slash-completion candidate (from `complete.slash`). */
export interface CompletionItem {
  text: string
  display: string
  meta: string
}

/** A delegated subagent, tracked from the `subagent.*` event stream (agents dashboard). */
export interface SubagentInfo {
  id: string
  goal: string
  status: string
  depth: number
  model?: string
  parentId?: string
  summary?: string
  lastTool?: string
}

/**
 * Live session chrome (the status bar — item 14). Sourced from the `session.info`
 * event (and the `session.create`/`resume` result's `info`), refreshed whenever
 * the gateway's agent/config state changes. `running` is the turn-active flag the
 * Ctrl-C interrupt (item 11) reads; we also flip it locally on message.start/
 * complete so the bar reacts instantly even if a `session.info` lags.
 */
export interface SessionInfo {
  model?: string
  effort?: string
  fast?: boolean
  cwd?: string
  branch?: string
  running?: boolean
  contextUsed?: number
  contextMax?: number
  contextPercent?: number
  compressions?: number
}

export interface StoreState {
  ready: boolean
  messages: Message[]
  theme: Theme
  /** The active blocking prompt (composer is hidden while set); undefined when none. */
  prompt: ActivePrompt | undefined
  /** The open pager overlay (replaces the transcript while set); undefined when none. */
  pager: PagerState | undefined
  /** The open session switcher (replaces the composer while set); undefined when none. */
  switcher: SessionItem[] | undefined
  /** The open generic picker (model/skills/…); undefined when none. */
  picker: PickerState | undefined
  /** Live slash-completion candidates shown above the composer; undefined/empty when none. */
  completions: CompletionItem[] | undefined
  /** Delegated subagents (from `subagent.*`), shown in the agents dashboard. */
  subagents: SubagentInfo[]
  /** Whether the agents dashboard overlay is open (/agents). */
  dashboard: boolean
  /** Transient busy indicator (the kaomoji face/verb from `thinking.delta`/`status.update`);
   *  shown above the composer WHILE a turn runs, cleared on `message.complete`. NOT transcript. */
  status: string | undefined
  /** Live session chrome for the status bar (model/effort/cwd/branch/context/running). */
  info: SessionInfo
  /** Transient hint shown above the composer (e.g. "Ctrl+C again to quit" — item 11);
   *  takes visual priority over the busy `status` face. Undefined when none. */
  hint: string | undefined
}

const LRU_LIMIT = 1000

/** Read a string field off an unknown payload record (no `any`, no cast). */
function readStr(payload: { readonly [k: string]: unknown }, key: string): string | undefined {
  const v = payload[key]
  return typeof v === 'string' ? v : undefined
}

/** Read a number field off an unknown payload record. */
function readNum(payload: { readonly [k: string]: unknown }, key: string): number {
  const v = payload[key]
  return typeof v === 'number' ? v : 0
}

/** Read an optional number (undefined when absent) — distinguishes "0" from "missing". */
function readOptNum(payload: { readonly [k: string]: unknown }, key: string): number | undefined {
  const v = payload[key]
  return typeof v === 'number' ? v : undefined
}

/** Read a boolean field (undefined when absent). */
function readOptBool(payload: { readonly [k: string]: unknown }, key: string): boolean | undefined {
  const v = payload[key]
  return typeof v === 'boolean' ? v : undefined
}

/**
 * Fold a `session.info` / `session.create.info` / `session.usage` payload into a
 * partial SessionInfo, reading context/usage fields from either a nested `usage`
 * object or the top level (the gateway shapes vary by RPC vs event — §gateway map).
 */
function readInfoPatch(payload: { readonly [k: string]: unknown }): Partial<SessionInfo> {
  const usageRaw = payload['usage']
  const usage: { readonly [k: string]: unknown } =
    usageRaw && typeof usageRaw === 'object' ? (usageRaw as { readonly [k: string]: unknown }) : {}
  const patch: Partial<SessionInfo> = {}
  const model = readStr(payload, 'model')
  if (model) patch.model = model
  const effort = readStr(payload, 'reasoning_effort')
  if (effort) patch.effort = effort
  const fast = readOptBool(payload, 'fast')
  if (fast !== undefined) patch.fast = fast
  const cwd = readStr(payload, 'cwd')
  if (cwd) patch.cwd = cwd
  const branch = readStr(payload, 'branch')
  if (branch) patch.branch = branch
  const running = readOptBool(payload, 'running')
  if (running !== undefined) patch.running = running
  const used = readOptNum(usage, 'context_used') ?? readOptNum(payload, 'context_used')
  if (used !== undefined) patch.contextUsed = used
  const max = readOptNum(usage, 'context_max') ?? readOptNum(payload, 'context_max')
  if (max !== undefined) patch.contextMax = max
  const pct = readOptNum(usage, 'context_percent') ?? readOptNum(payload, 'context_percent')
  if (pct !== undefined) patch.contextPercent = pct
  const comp = readOptNum(usage, 'compressions') ?? readOptNum(payload, 'compressions')
  if (comp !== undefined) patch.compressions = comp
  return patch
}

/** The subagent status implied by an event type (an explicit payload `status` wins). */
function subagentStatusFor(type: string): string {
  if (type === 'subagent.complete') return 'complete'
  if (type === 'subagent.thinking') return 'thinking'
  if (type === 'subagent.tool') return 'tool'
  if (type === 'subagent.progress') return 'working'
  return 'running'
}

export function createSessionStore() {
  const [state, setState] = createStore<StoreState>({
    ready: false,
    messages: [],
    theme: DEFAULT_THEME,
    prompt: undefined,
    pager: undefined,
    switcher: undefined,
    picker: undefined,
    completions: undefined,
    subagents: [],
    dashboard: false,
    status: undefined,
    info: {},
    hint: undefined
  })

  // Monotonic part id (stable `key` per part so a new tool part below a streaming
  // text part doesn't remount/re-tokenize it).
  let partSeq = 0
  const nextId = () => `p${++partSeq}`

  // LRU id-dedup: events that carry a stable id are applied at most once.
  const applied = new Set<string>()
  function duplicate(id: string | undefined): boolean {
    if (!id) return false
    if (applied.has(id)) return true
    applied.add(id)
    if (applied.size > LRU_LIMIT) {
      const oldest = applied.values().next()
      if (!oldest.done) applied.delete(oldest.value)
    }
    return false
  }

  // Hydrate-while-buffering (resume): while a snapshot is loading, live events
  // queue here and replay after the snapshot is reconciled (opencode sync-v2).
  let buffering: GatewayEvent[] | null = null

  function setSkin(skin: GatewaySkinDecoded | undefined): void {
    setState('theme', themeFromSkin(skin))
  }

  // ── parts helpers (operate on a draft message inside produce) ───────────
  function appendPart(m: Message, type: 'text' | 'reasoning', text: string): void {
    const parts = (m.parts ??= [])
    const last = parts[parts.length - 1]
    if (last && last.type === type) last.text += text
    else parts.push({ type, id: nextId(), text })
  }

  /** The live (last) assistant message, optionally only when still streaming. */
  function liveAssistant(draft: StoreState, streamingOnly = false): Message | undefined {
    const last = draft.messages[draft.messages.length - 1]
    if (last && last.role === 'assistant' && (!streamingOnly || last.streaming)) return last
    return undefined
  }

  /** Ensure there's an open assistant turn to attach parts to (tool/reasoning). */
  function ensureAssistant(draft: StoreState): Message {
    const live = liveAssistant(draft, true)
    if (live) return live
    const created: Message = { role: 'assistant', text: '', parts: [], streaming: true }
    draft.messages.push(created)
    return created
  }

  /** Find a tool part by id, scanning recent assistant turns (complete may land late). */
  function findToolPart(draft: StoreState, id: string): ToolPartState | undefined {
    for (let i = draft.messages.length - 1; i >= 0; i--) {
      const parts = draft.messages[i]?.parts
      if (!parts) continue
      for (let j = parts.length - 1; j >= 0; j--) {
        const p = parts[j]
        if (p && p.type === 'tool' && p.id === id) return p
      }
    }
    return undefined
  }

  /** Push a user message (composer submit). */
  function pushUser(text: string) {
    setState(
      produce(draft => {
        draft.messages.push({ role: 'user', text })
      })
    )
  }

  /** Push a system line (slash output, errors, notices). */
  function pushSystem(text: string) {
    setState(
      produce(draft => {
        draft.messages.push({ role: 'system', text })
      })
    )
  }

  /** Clear the transcript (e.g. /clear, /new) and any tracked subagents. */
  function clearTranscript() {
    setState('messages', [])
    setState('subagents', [])
  }

  /** Open / close the agents dashboard overlay (/agents). */
  function openDashboard() {
    setState('dashboard', true)
  }
  function closeDashboard() {
    setState('dashboard', false)
  }

  /** Open a local Y/N confirm dialog (non-gateway; e.g. /clear). */
  function setConfirm(message: string, onConfirm: () => void) {
    setState('prompt', { kind: 'confirm', message, onConfirm })
  }

  /** Open the pager overlay (long slash output: /status, /logs, …). */
  function openPager(title: string, text: string) {
    setState('pager', { title, text })
  }

  /** Close the pager overlay. */
  function closePager() {
    setState('pager', undefined)
  }

  /** Open the session switcher with the given session rows (/sessions, /resume). */
  function openSwitcher(sessions: SessionItem[]) {
    setState('switcher', sessions)
  }

  /** Close the session switcher. */
  function closeSwitcher() {
    setState('switcher', undefined)
  }

  /** Open the generic picker (model picker, skills hub, …). */
  function openPicker(picker: PickerState) {
    setState('picker', picker)
  }

  /** Close the generic picker. */
  function closePicker() {
    setState('picker', undefined)
  }

  /** Set / clear the transient composer hint ("Ctrl+C again to quit" — item 11). */
  function setHint(text: string | undefined): void {
    setState('hint', text)
  }

  /** Merge a session-info patch into the chrome state (status bar — item 14). */
  function applyInfo(raw: { readonly [k: string]: unknown }): void {
    const patch = readInfoPatch(raw)
    if (Object.keys(patch).length) setState('info', prev => ({ ...prev, ...patch }))
  }

  /** Set / clear the live slash-completion candidates (composer dropdown). */
  function setCompletions(items: CompletionItem[]) {
    setState('completions', items.length ? items : undefined)
  }
  function clearCompletions() {
    setState('completions', undefined)
  }

  /** Reduce a decoded gateway event into the store. The sole boundary->Solid sink. */
  function apply(event: GatewayEvent): void {
    if (buffering) {
      buffering.push(event)
      return
    }
    applyNow(event)
  }

  function applyNow(event: GatewayEvent): void {
    switch (event.type) {
      case 'gateway.ready':
        setState('ready', true)
        setSkin(event.payload?.skin)
        break
      case 'skin.changed':
        setSkin(event.payload)
        break
      case 'session.info':
        applyInfo(event.payload)
        break
      case 'message.start':
        setState('status', undefined)
        setState('info', prev => ({ ...prev, running: true }))
        setState(
          produce(draft => {
            draft.messages.push({ role: 'assistant', text: '', parts: [], streaming: true })
          })
        )
        break
      case 'message.delta': {
        // prefer `text` over `rendered` (gotcha §8 #4 — rendered is incremental Rich-ANSI).
        const text = event.payload?.text ?? ''
        if (!text) break
        setState(
          produce(draft => {
            const live = liveAssistant(draft, true)
            if (live) appendPart(live, 'text', text)
          })
        )
        break
      }
      case 'message.complete':
        setState(
          produce(draft => {
            const live = liveAssistant(draft, true)
            if (!live) return
            // If no deltas arrived (complete-only gateways), seed the full text once.
            const finalText = event.payload?.text
            const hasText = (live.parts ?? []).some(p => p.type === 'text' && p.text.length > 0)
            if (finalText && !hasText) appendPart(live, 'text', finalText)
            live.streaming = false
          })
        )
        setState('status', undefined)
        setState('info', prev => ({ ...prev, running: false }))
        // message.complete carries the latest usage/context — refresh the bar.
        if (event.payload) applyInfo(event.payload)
        break
      // thinking.delta / status.update are the TRANSIENT busy indicator (kaomoji
      // face/verb) — route them to the status line, NOT the transcript (gotcha: Ink
      // shows these as a FaceTicker, not message content).
      case 'thinking.delta':
      case 'status.update': {
        const text = event.payload?.text ?? ''
        if (text) setState('status', text)
        break
      }
      // reasoning.delta is the model's actual reasoning — a (dim) transcript part.
      case 'reasoning.delta': {
        const text = event.payload?.text ?? ''
        if (!text) break
        setState(
          produce(draft => {
            appendPart(ensureAssistant(draft), 'reasoning', text)
          })
        )
        break
      }
      case 'tool.start': {
        const id = readStr(event.payload, 'tool_id')
        if (!id) break
        const name = readStr(event.payload, 'name') ?? 'tool'
        setState(
          produce(draft => {
            const live = ensureAssistant(draft)
            ;(live.parts ??= []).push({ type: 'tool', id, name, state: 'running' })
          })
        )
        break
      }
      case 'tool.complete': {
        const id = readStr(event.payload, 'tool_id')
        if (!id) break
        const name = readStr(event.payload, 'name')
        const error = readStr(event.payload, 'error')
        const summary = readStr(event.payload, 'summary')
        const resultText = stripToolEnvelope(readStr(event.payload, 'result_text') ?? summary ?? '')
        const lineCount = resultText ? resultText.replace(/\s+$/, '').split('\n').length : 0
        setState(
          produce(draft => {
            let part = findToolPart(draft, id)
            if (!part) {
              // complete without a matching start — append a settled tool part.
              part = { type: 'tool', id, name: name ?? 'tool', state: 'running' }
              ;(ensureAssistant(draft).parts ??= []).push(part)
            }
            part.state = 'complete'
            part.lineCount = lineCount
            if (name) part.name = name
            if (resultText) part.resultText = resultText
            if (summary) part.summary = summary
            if (error) part.error = error
          })
        )
        break
      }
      // ── blocking prompts (spec §8 #6 — unhandled = the agent deadlocks) ──
      case 'clarify.request':
        setState('prompt', {
          kind: 'clarify',
          question: event.payload.question ?? '',
          // decoded choices are readonly — copy to the store's mutable string[]
          choices: event.payload.choices ? [...event.payload.choices] : null,
          requestId: event.payload.request_id
        })
        break
      case 'approval.request':
        setState('prompt', { kind: 'approval', command: event.payload.command, description: event.payload.description })
        break
      case 'sudo.request':
        setState('prompt', { kind: 'sudo', requestId: event.payload.request_id })
        break
      case 'secret.request':
        setState('prompt', {
          kind: 'secret',
          envVar: event.payload.env_var,
          prompt: event.payload.prompt,
          requestId: event.payload.request_id
        })
        break
      // ── subagents (agents dashboard) — track the delegation tree by id ──
      case 'subagent.spawn_requested':
      case 'subagent.start':
      case 'subagent.thinking':
      case 'subagent.tool':
      case 'subagent.progress':
      case 'subagent.complete': {
        const id = readStr(event.payload, 'subagent_id')
        if (!id) break
        setState(
          produce(draft => {
            let sa = draft.subagents.find(s => s.id === id)
            if (!sa) {
              sa = { depth: readNum(event.payload, 'depth'), goal: '', id, status: 'running' }
              draft.subagents.push(sa)
            }
            const goal = readStr(event.payload, 'goal')
            if (goal) sa.goal = goal
            const model = readStr(event.payload, 'model')
            if (model) sa.model = model
            const parent = readStr(event.payload, 'parent_id')
            if (parent) sa.parentId = parent
            const summary = readStr(event.payload, 'summary')
            if (summary) sa.summary = summary
            const tool = readStr(event.payload, 'tool_name')
            if (tool) sa.lastTool = tool
            sa.status = readStr(event.payload, 'status') ?? subagentStatusFor(event.type)
          })
        )
        break
      }
      // Other event types (chrome) are reduced in later phases; unhandled members
      // are intentionally ignored here.
    }
  }

  /** Clear the active blocking prompt (after it's answered/cancelled). */
  function clearPrompt(): void {
    setState('prompt', undefined)
  }

  // ── resume hydrate (opencode sync-v2): buffer live events while the snapshot
  // loads, then replace history + replay the buffer in order. Split into begin/
  // commit so the buffer can span an async `session.resume` RPC.
  /** Start buffering live events (call BEFORE the async resume RPC). Idempotent. */
  function beginBuffer(): void {
    if (!buffering) buffering = []
  }

  /** Replace history with the resume snapshot, then replay events buffered meanwhile. */
  function commitSnapshot(snapshot: Message[]): void {
    setState('messages', snapshot)
    const pending = buffering ?? []
    buffering = null
    for (const event of pending) applyNow(event)
  }

  /** Synchronous convenience: buffer → load → commit (used by tests). */
  function hydrate(loadSnapshot: () => Message[]): void {
    beginBuffer()
    commitSnapshot(loadSnapshot())
  }

  return {
    state,
    apply,
    pushUser,
    pushSystem,
    clearTranscript,
    setConfirm,
    openPager,
    closePager,
    openSwitcher,
    closeSwitcher,
    openPicker,
    closePicker,
    setCompletions,
    clearCompletions,
    applyInfo,
    setHint,
    openDashboard,
    closeDashboard,
    hydrate,
    beginBuffer,
    commitSnapshot,
    duplicate,
    clearPrompt
  } as const
}

export type SessionStore = ReturnType<typeof createSessionStore>
