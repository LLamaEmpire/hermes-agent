/**
 * Entry — the single boundary edge (spec v4 §3.1). This is the ONE place that:
 *   - acquires the renderer (acquireRelease + Deferred-on-destroy),
 *   - creates the Solid store,
 *   - wires GatewayService.subscribe -> store.apply  (Effect->Solid contact #2),
 *   - does the one-line `render(() => <App/>, renderer)` bridge (contact #1),
 *   - (live) bootstraps a session and optionally submits an initial prompt,
 *   - blocks until the renderer is destroyed (user quit),
 * and at the bottom PROVIDES the layers and runs (`Effect.provide(AppLayer)`).
 *
 * Backend selection (import.meta.main):
 *   - default            → the LIVE `liveGatewayLayer` (spawns the real Python
 *     `tui_gateway`); after `gateway.ready` it `session.create`s and, if an
 *     initial prompt is given (HERMES_TUI_PROMPT or argv), `prompt.submit`s it.
 *     The composer lands in Phase 2 — until then the initial prompt is how a
 *     streamed reply is driven into the transcript (spec Phase-1 smoke).
 *   - HERMES_TUI_FAKE=1  → the scripted FakeGateway "hello" (offline dev/CI).
 *
 * The body of `run` does not change when the backend swaps — that's the point of
 * the layer; only `makeAppLayer(...)` differs at the edge.
 */
import { render } from '@opentui/solid'
import { Deferred, Duration, Effect } from 'effect'

import { GatewayService, type GatewayServiceShape } from '../boundary/gateway/GatewayService.ts'
import { liveGatewayLayer } from '../boundary/gateway/liveGateway.ts'
import { getLog } from '../boundary/log.ts'
import { acquireRenderer } from '../boundary/renderer.ts'
import { makeAppLayer } from '../boundary/runtime.ts'
import { mapResumeHistory, mapSessionList } from '../logic/resume.ts'
import { dispatchSlash, mapCompletions, type SlashContext } from '../logic/slash.ts'
import { createSessionStore, type SessionStore } from '../logic/store.ts'
import { App } from '../view/App.tsx'
import { ThemeProvider } from '../view/theme.tsx'
import { makeFakeGatewayLayer, type FakeGatewayController } from './fakeGateway.ts'

export interface TuiInput {
  /** Mouse tracking on/off. */
  readonly mouse: boolean
  /** Skip the live session bootstrap (the fake backend drives the stream itself). */
  readonly fake: boolean
  /** Terminal width passed to `session.create` (Ink uses the live cols; 80 is a fine default). */
  readonly cols: number
  /** Optional initial prompt submitted once the session is ready — the Phase-1 stand-in for the composer. */
  readonly initialPrompt?: string
  /** Resume a session instead of creating one: a session id, or 'recent'/'last' (→ session.most_recent). */
  readonly resumeId?: string
}

const READY_POLL = Duration.millis(100)
const READY_TIMEOUT_MS = 20_000
/** Window after a Ctrl+C in which a second Ctrl+C quits the TUI (item 11). */
const QUIT_WINDOW_MS = 3_000

/**
 * Resume a session INTO the store: buffer live events across the `session.resume`
 * RPC, then replace history + replay (gotcha §8 #5 tool rows handled by
 * mapResumeHistory). Shared by the launch bootstrap and the session switcher.
 * Timed (rpc_ms / hydrate_ms) for the resume profile.
 */
const resumeInto = (gateway: GatewayServiceShape, store: SessionStore, sid: string, cols: number) =>
  Effect.gen(function* () {
    store.beginBuffer()
    const t0 = Date.now()
    const resumed = yield* gateway.request<{ messages?: unknown; info?: Record<string, unknown> }>('session.resume', {
      cols,
      session_id: sid
    })
    const t1 = Date.now()
    const snapshot = mapResumeHistory(resumed?.messages)
    store.commitSnapshot(snapshot)
    if (resumed?.info) store.applyInfo(resumed.info)
    getLog().info('bootstrap', 'session resumed', {
      count: snapshot.length,
      hydrate_ms: Date.now() - t1,
      rpc_ms: t1 - t0,
      sid
    })
  })

/**
 * Live session bootstrap: wait for the unsolicited `gateway.ready` handshake,
 * then either RESUME a session (hydrate its transcript — incl. tool rows — via
 * the snapshot, buffering live events across the RPC) or CREATE a fresh one, and
 * (if given) submit the initial prompt. Forked into the entry scope so it runs
 * concurrently with the render + the quit-await. Any failure is logged and
 * swallowed — a bootstrap hiccup must never tear down the rendered UI.
 */
const bootstrapSession = (gateway: GatewayServiceShape, store: SessionStore, input: TuiInput) =>
  Effect.gen(function* () {
    const log = getLog()
    let waited = 0
    while (!store.state.ready && waited < READY_TIMEOUT_MS) {
      yield* Effect.sleep(READY_POLL)
      waited += 100
    }
    if (!store.state.ready) {
      log.warn('bootstrap', 'no gateway.ready within timeout', { waited })
      return
    }

    let sid: string | undefined
    if (input.resumeId) {
      sid = input.resumeId
      if (sid === 'recent' || sid === 'last') {
        const recent = yield* gateway.request<{ session_id?: string }>('session.most_recent', {})
        sid = recent?.session_id
      }
      if (!sid) {
        log.warn('bootstrap', 'no session to resume', { resumeId: input.resumeId })
        return
      }
      yield* resumeInto(gateway, store, sid, input.cols)
    } else {
      const created = yield* gateway.request<{ session_id?: string; info?: Record<string, unknown> }>(
        'session.create',
        { cols: input.cols }
      )
      sid = created?.session_id ?? gateway.sessionId()
      if (!sid) {
        log.warn('bootstrap', 'session.create returned no session_id')
        return
      }
      if (created?.info) store.applyInfo(created.info)
      log.info('bootstrap', 'session created', { sid })
    }

    const prompt = input.initialPrompt?.trim()
    if (prompt) {
      store.pushUser(prompt)
      yield* gateway.request('prompt.submit', { session_id: sid, text: prompt })
    }
  }).pipe(Effect.catchCause(cause => Effect.sync(() => getLog().warn('bootstrap', 'failed', { cause: String(cause) }))))

/** The entry Effect. Mirrors opencode `app.tsx:177` `run = Effect.fn("Tui.run")`. */
export const run = Effect.fn('Tui.run')(function* (input: TuiInput) {
  yield* Effect.scoped(
    Effect.gen(function* () {
      // Solid side: the store + reducer. Created here, lives in Solid-land.
      const store = createSessionStore()

      // Contact point #2: boundary pushes decoded events into the Solid store.
      const gateway = yield* GatewayService
      yield* gateway.subscribe(event => store.apply(event))

      // ── Ctrl+C state machine (item 11) ──────────────────────────────────
      // While a turn runs, the first Ctrl+C STOPS the agent (session.interrupt);
      // a second Ctrl+C within QUIT_WINDOW_MS (or when idle) KILLS the TUI. The
      // debounce stops a stray Ctrl+C from nuking the session (opencode's
      // double-press model; the user's preferred behaviour).
      let quitArmed = false
      let quitTimer: ReturnType<typeof setTimeout> | undefined
      let doQuit = () => {} // assigned once the renderer exists
      const disarmQuit = () => {
        quitArmed = false
        if (quitTimer) clearTimeout(quitTimer)
        quitTimer = undefined
        store.setHint(undefined)
      }
      const armQuit = (message: string) => {
        quitArmed = true
        store.setHint(message)
        if (quitTimer) clearTimeout(quitTimer)
        quitTimer = setTimeout(disarmQuit, QUIT_WINDOW_MS)
      }
      const interruptTurn = () => {
        const sid = gateway.sessionId()
        if (!sid) return
        Effect.runFork(
          gateway
            .request('session.interrupt', { session_id: sid })
            .pipe(
              Effect.catchCause(cause => Effect.sync(() => getLog().warn('interrupt', 'failed', { cause: String(cause) })))
            )
        )
      }
      const onCtrlC = () => {
        if (quitArmed) {
          disarmQuit()
          doQuit()
          return
        }
        if (store.state.info.running) {
          interruptTurn()
          armQuit('⏹ stopped — Ctrl+C again to quit')
        } else {
          armQuit('Ctrl+C again to quit')
        }
      }

      // A blocking prompt owns Ctrl+C (→ cancel); otherwise the state machine above runs.
      const { renderer, shutdown } = yield* acquireRenderer({
        mouse: input.mouse,
        isBlocked: () => store.state.prompt !== undefined,
        onCtrlC
      })
      doQuit = () => {
        if (!renderer.isDestroyed) renderer.destroy()
      }

      // Submit a user turn: the service value is in hand, so `gateway.request(...)`
      // is Effect<…, never> — fire it detached with runFork; failures are logged.
      const submitPrompt = (text: string) => {
        store.pushUser(text)
        const sid = gateway.sessionId()
        if (!sid) {
          getLog().warn('submit', 'no session yet — dropping prompt', { text })
          return
        }
        Effect.runFork(
          gateway
            .request('prompt.submit', { session_id: sid, text })
            .pipe(
              Effect.catchCause(cause => Effect.sync(() => getLog().warn('submit', 'failed', { cause: String(cause) })))
            )
        )
      }

      // Slash dispatch context (Solid logic; the boundary just hands it a
      // Promise-returning `request` + the host capabilities it needs).
      const slashCtx: SlashContext = {
        clearTranscript: () => store.clearTranscript(),
        confirm: (message, onConfirm) => store.setConfirm(message, onConfirm),
        listSessions: () => Effect.runPromise(gateway.request('session.list', {})).then(mapSessionList),
        logTail: () =>
          getLog()
            .tail(200)
            .map(e => `${e.scope}: ${e.msg}`),
        openDashboard: () => store.openDashboard(),
        openPager: (title, text) => store.openPager(title, text),
        openPicker: picker => store.openPicker(picker),
        openSwitcher: sessions => store.openSwitcher(sessions),
        pushSystem: text => store.pushSystem(text),
        quit: () => {
          if (!renderer.isDestroyed) renderer.destroy()
        },
        request: (method, params) => Effect.runPromise(gateway.request(method, params)),
        sessionId: () => gateway.sessionId(),
        submit: submitPrompt
      }

      // Resume a chosen session (session switcher pick) — same hydrate path as launch.
      const onResume = (resumeSid: string) => {
        Effect.runFork(
          resumeInto(gateway, store, resumeSid, input.cols).pipe(
            Effect.catchCause(cause => Effect.sync(() => getLog().warn('resume', 'failed', { cause: String(cause) })))
          )
        )
      }

      // The composer's submit: route `/command` through the slash ladder, else a prompt.
      const submit = (text: string) => {
        if (text.startsWith('/')) void dispatchSlash(text, slashCtx)
        else submitPrompt(text)
      }

      // Live slash completions: query `complete.slash` while typing a `/command`
      // name (no space yet); clear otherwise. Cheap local completer — fired per
      // keystroke (a debounce is a polish item).
      const onType = (text: string) => {
        if (!text.startsWith('/') || text.includes(' ') || text.includes('\n')) {
          store.clearCompletions()
          return
        }
        Effect.runPromise(gateway.request('complete.slash', { text }))
          .then(result => store.setCompletions(mapCompletions(result)))
          .catch(() => store.clearCompletions())
      }

      // Blocking-prompt replies (clarify/approval/sudo/secret `*.respond`). Same
      // detached-runFork pattern; failures logged, never thrown into the view.
      const respond = (method: string, params: Record<string, unknown>) => {
        Effect.runFork(
          gateway
            .request(method, params)
            .pipe(
              Effect.catchCause(cause =>
                Effect.sync(() => getLog().warn('respond', 'failed', { cause: String(cause), method }))
              )
            )
        )
      }

      // Live backend: drive a session (create + optional initial prompt) concurrently.
      if (!input.fake) yield* Effect.forkScoped(bootstrapSession(gateway, store, input))

      // Contact point #1: the single render bridge. After this, the screen is Solid's.
      // The theme is sourced reactively from the store (skin events update it).
      yield* Effect.promise(() =>
        render(
          () => (
            <ThemeProvider theme={() => store.state.theme}>
              <App
                store={store}
                onSubmit={submit}
                onType={onType}
                onRespond={respond}
                onResume={onResume}
                sessionId={() => gateway.sessionId()}
              />
            </ThemeProvider>
          ),
          renderer
        )
      )

      // Block until the renderer is destroyed (Ctrl+C / quit); finalizers then run.
      yield* Deferred.await(shutdown)
    })
  )
})

/** Scripted "hello" stream so the fake backend paints a non-empty frame offline. */
function streamHello(controller: FakeGatewayController): void {
  controller.emit({ type: 'gateway.ready' })
  controller.emit({ type: 'message.start' })
  for (const chunk of ['Hi ', 'there, ', 'glitch!']) {
    controller.emit({ type: 'message.delta', payload: { text: chunk } })
  }
  controller.emit({ type: 'message.complete' })
}

const TRUE_RE = /^(?:1|true|yes|on)$/i

if (import.meta.main) {
  const fake = TRUE_RE.test(process.env.HERMES_TUI_FAKE?.trim() ?? '')
  const cols = process.stdout.columns || 80
  const initialPrompt = process.env.HERMES_TUI_PROMPT?.trim() || process.argv.slice(2).join(' ').trim()
  const resumeId = process.env.HERMES_TUI_RESUME?.trim()
  // Mouse on by default (opencode parity: wheel-scroll the transcript, drag the
  // scrollbar, click-to-expand tools, text-aware selection). HERMES_TUI_MOUSE=0 opts out.
  const mouse = !/^(?:0|false|no|off)$/i.test(process.env.HERMES_TUI_MOUSE?.trim() ?? '')
  const base = { mouse, fake, cols }
  const withPrompt = initialPrompt ? { ...base, initialPrompt } : base
  const input: TuiInput = resumeId ? { ...withPrompt, resumeId } : withPrompt

  const onFatal = (error: unknown) => {
    getLog().error('entry', 'fatal', { error: String(error) })
    process.exitCode = 1
  }

  if (fake) {
    const { layer, controller } = makeFakeGatewayLayer()
    // Drive the fake stream shortly after mount so the subscription is live.
    setTimeout(() => streamHello(controller), 50)
    Effect.runPromise(run(input).pipe(Effect.provide(makeAppLayer(layer)))).catch(onFatal)
  } else {
    Effect.runPromise(run(input).pipe(Effect.provide(makeAppLayer(liveGatewayLayer)))).catch(onFatal)
  }
}
