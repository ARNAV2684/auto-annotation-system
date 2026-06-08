import { useEffect } from 'react'

/**
 * A single keyboard binding. The same array drives both the global keydown
 * listener (in this hook) and the help overlay (so they cannot drift).
 *
 * `combo` strings use the canonical form parsed by `eventMatchesCombo` below.
 * Examples: 'Shift+A', 'Shift+?', 'Escape'. Modifier required for safety —
 * unmodified letter keys (except Escape) intentionally do nothing, so that a
 * stray keystroke in an empty area never mutates state.
 */
export interface Shortcut {
  combo: string
  label: string
  group: string
  handler: () => void
  /** If true, the shortcut also fires while a text input is focused. Default: false. */
  allowInInputs?: boolean
}

const EDITABLE_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT'])

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (EDITABLE_TAGS.has(target.tagName)) return true
  if (target.isContentEditable) return true
  return false
}

function normaliseKey(event: KeyboardEvent): string {
  // For letter keys, event.key reflects the modifier-applied character ('A' vs 'a'),
  // which is exactly what we want for matching 'Shift+A'. For '?' the same is true —
  // pressing Shift+/ on US layouts gives event.key === '?'.
  const k = event.key
  if (k === ' ') return 'Space'
  // Escape, Enter, Tab, ArrowUp etc. arrive with their canonical names already.
  if (k.length === 1) return k.toUpperCase()
  return k
}

function eventMatchesCombo(event: KeyboardEvent, combo: string): boolean {
  const parts = combo.split('+').map((p) => p.trim())
  const key = parts[parts.length - 1]
  const wantShift = parts.slice(0, -1).some((p) => p.toLowerCase() === 'shift')
  const wantCtrl = parts.slice(0, -1).some((p) => p.toLowerCase() === 'ctrl')
  const wantAlt = parts.slice(0, -1).some((p) => p.toLowerCase() === 'alt')
  if (event.shiftKey !== wantShift) return false
  if (event.ctrlKey !== wantCtrl) return false
  if (event.altKey !== wantAlt) return false
  if (event.metaKey) return false
  return normaliseKey(event) === key.toUpperCase()
}

/**
 * Attach a global keydown listener while `enabled` is true.
 *
 * Skips the lookup when focus is inside a text input / textarea / select /
 * contenteditable, unless the matched shortcut sets `allowInInputs: true`
 * (used for Escape so it can always dismiss a modal).
 */
export function useKeyboardShortcuts(shortcuts: Shortcut[], enabled: boolean): void {
  useEffect(() => {
    if (!enabled) return
    const handler = (event: KeyboardEvent) => {
      const editable = isEditableTarget(event.target)
      for (const shortcut of shortcuts) {
        if (!eventMatchesCombo(event, shortcut.combo)) continue
        if (editable && !shortcut.allowInInputs) return
        event.preventDefault()
        shortcut.handler()
        return
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [shortcuts, enabled])
}
