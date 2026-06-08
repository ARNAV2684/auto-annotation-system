import { useEffect } from 'react'
import type { Shortcut } from './useKeyboardShortcuts'

interface ShortcutHelpOverlayProps {
  open: boolean
  shortcuts: Shortcut[]
  onClose: () => void
}

/**
 * Centered modal showing every keyboard shortcut, grouped by `Shortcut.group`.
 *
 * Closes on Escape OR any other key press (so the user can flick it open with
 * Shift+? and dismiss with any keystroke). Mouse-click on the backdrop also
 * closes it. The component owns its own keydown listener while open instead
 * of going through `useKeyboardShortcuts`, because while the overlay is up
 * the page-wide shortcut bindings should NOT fire — pressing Shift+A to close
 * the overlay must not also approve an annotation underneath.
 */
export function ShortcutHelpOverlay({
  open,
  shortcuts,
  onClose,
}: ShortcutHelpOverlayProps) {
  useEffect(() => {
    if (!open) return
    const handler = (event: KeyboardEvent) => {
      event.preventDefault()
      event.stopPropagation()
      onClose()
    }
    // Capture phase so this listener wins over the page shortcut handler.
    window.addEventListener('keydown', handler, true)
    return () => window.removeEventListener('keydown', handler, true)
  }, [open, onClose])

  if (!open) return null

  const groups = new Map<string, Shortcut[]>()
  for (const s of shortcuts) {
    const existing = groups.get(s.group)
    if (existing) existing.push(s)
    else groups.set(s.group, [s])
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Keyboard shortcuts"
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="max-h-[80vh] w-full max-w-xl overflow-auto rounded-lg bg-white p-6 shadow-2xl"
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-800">Keyboard shortcuts</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded px-2 py-0.5 text-sm text-slate-500 hover:bg-slate-100 hover:text-slate-800"
          >
            Esc
          </button>
        </div>
        <div className="space-y-5">
          {Array.from(groups.entries()).map(([group, items]) => (
            <section key={group}>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                {group}
              </h3>
              <ul className="space-y-1.5">
                {items.map((s) => (
                  <li
                    key={s.combo + s.label}
                    className="flex items-center justify-between gap-3 text-sm"
                  >
                    <span className="text-slate-700">{s.label}</span>
                    <ComboBadge combo={s.combo} />
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
        <p className="mt-5 text-xs text-slate-500">
          Most shortcuts require <kbd className="font-mono text-slate-700">Shift</kbd> so an
          accidental keystroke can&apos;t change your work. Shortcuts are ignored while a
          text field is focused.
        </p>
      </div>
    </div>
  )
}

function ComboBadge({ combo }: { combo: string }) {
  const parts = combo.split('+').map((p) => p.trim())
  return (
    <span className="flex items-center gap-1">
      {parts.map((part, i) => (
        <kbd
          key={i}
          className="rounded border border-slate-300 bg-slate-50 px-1.5 py-0.5 font-mono text-xs text-slate-700 shadow-sm"
        >
          {part}
        </kbd>
      ))}
    </span>
  )
}
