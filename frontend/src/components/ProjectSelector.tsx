import { useProjects } from '../api/hooks'
import type { Project } from '../api/types'

interface ProjectSelectorProps {
  currentProject: Project | null
  onChange: (projectId: string) => void
}

/** Header dropdown to switch between configured projects. */
export function ProjectSelector({ currentProject, onChange }: ProjectSelectorProps) {
  const { data: projects, isLoading } = useProjects()

  if (isLoading) {
    return <span className="text-sm text-slate-400">loading projects…</span>
  }
  if (!projects || projects.length === 0) {
    return <span className="text-sm text-slate-400">no projects configured</span>
  }

  return (
    <label className="flex items-center gap-2 text-sm text-slate-600">
      <span className="font-medium">Project:</span>
      <select
        value={currentProject?.id ?? projects[0].id}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border border-slate-300 bg-white px-2 py-1 text-sm focus:border-sky-500 focus:outline-none"
      >
        {projects.map((p) => (
          <option key={p.id} value={p.id}>
            {p.label}
          </option>
        ))}
      </select>
    </label>
  )
}
