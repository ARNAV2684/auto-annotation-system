import { StatsWidget } from './StatsWidget'

interface HeaderProps {
  onClearSession: () => void
}

/** Top bar with the app title, live stats, and the Clear Session button. */
export function Header({ onClearSession }: HeaderProps) {
  return (
    <header className="flex flex-shrink-0 items-center justify-between gap-4 border-b border-slate-200 bg-white px-6 py-3 shadow-sm">
      <div className="flex items-center gap-3">
        <span className="text-lg font-semibold text-slate-900">Auto Annotation System</span>
        <span className="rounded bg-slate-100 px-2 py-0.5 text-xs font-mono text-slate-600">
          v0.5-dev
        </span>
      </div>
      <div className="flex items-center gap-6">
        <StatsWidget />
        <button
          type="button"
          onClick={onClearSession}
          className="rounded border border-red-200 px-3 py-1 text-sm text-red-700 transition hover:bg-red-50"
        >
          Clear Session
        </button>
      </div>
    </header>
  )
}
