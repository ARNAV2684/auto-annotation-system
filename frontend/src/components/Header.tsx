import type { Project } from '../api/types'
import { ProjectSelector } from './ProjectSelector'
import { StatsWidget } from './StatsWidget'

interface HeaderProps {
  currentProject: Project | null
  onProjectChange: (projectId: string) => void
  onClearSession: () => void
}

/** Top bar: title + project selector + live stats + Clear Session button. */
export function Header({ currentProject, onProjectChange, onClearSession }: HeaderProps) {
  return (
    <header className="flex flex-shrink-0 items-center justify-between gap-4 border-b border-slate-200 bg-white px-6 py-3 shadow-sm">
      <div className="flex min-w-0 items-center gap-3">
        <span className="truncate text-lg font-semibold text-slate-900">
          Auto Annotation System
        </span>
        <span className="rounded bg-slate-100 px-2 py-0.5 font-mono text-xs text-slate-600">
          v0.6.0-dev
        </span>
      </div>
      <div className="flex items-center gap-6">
        <ProjectSelector currentProject={currentProject} onChange={onProjectChange} />
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
