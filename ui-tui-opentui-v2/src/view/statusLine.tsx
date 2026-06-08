/**
 * StatusLine — the transient line just below the transcript (spec §3 chrome).
 * Shows EITHER:
 *   - a `hint` (e.g. "Ctrl+C again to quit" — item 11), in the warn colour and
 *     taking priority; or
 *   - the kaomoji busy face/verb from `thinking.delta`/`status.update` WHILE a
 *     turn runs (Ink's FaceTicker), dim, cleared on `message.complete`.
 * This keeps those transient indicators OUT of the transcript. Renders nothing
 * when both are idle.
 */
import { Show } from 'solid-js'

import type { SessionStore } from '../logic/store.ts'
import { useTheme } from './theme.tsx'

export function StatusLine(props: { store: SessionStore }) {
  const theme = useTheme()
  const line = () => props.store.state.hint ?? props.store.state.status
  const isHint = () => props.store.state.hint !== undefined
  return (
    <Show when={line()}>
      {text => (
        <box style={{ flexShrink: 0 }}>
          <text>
            <span style={{ fg: isHint() ? theme().color.warn : theme().color.muted }}>{text()}</span>
          </text>
        </box>
      )}
    </Show>
  )
}
