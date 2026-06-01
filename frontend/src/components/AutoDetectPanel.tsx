import { useState } from 'react'
import { useAutoDetect } from '../api/hooks'
import type { Project } from '../api/types'

interface AutoDetectPanelProps {
  imageId: string
  /** Active project; default_prompts seed the input. ``null`` while loading. */
  project: Project | null
}

const DEFAULT_PROMPT = 'tree . palm tree'

/** Text input + Run button -> POST /api/auto-detect. Shows last-detection count. */
export function AutoDetectPanel({ imageId, project }: AutoDetectPanelProps) {
  const projectDefault = project?.default_prompts?.length
    ? project.default_prompts.join(' . ')
    : DEFAULT_PROMPT

  // The input is a free-text field; it gets re-seeded whenever the active
  // project changes (derive-during-render so no useEffect needed).
  const [prompt, setPrompt] = useState(projectDefault)
  const [prevProjectId, setPrevProjectId] = useState<string | null>(project?.id ?? null)
  if (prevProjectId !== (project?.id ?? null)) {
    setPrevProjectId(project?.id ?? null)
    setPrompt(projectDefault)
  }

  const autoDetect = useAutoDetect(imageId)

  const run = () => {
    // Backend expects an array. We split on "." so users can enter multiple phrases.
    const prompts = prompt
      .split('.')
      .map((p) => p.trim())
      .filter(Boolean)
    if (prompts.length === 0) return
    autoDetect.mutate({ prompts })
  }

  return (
    <div className="rounded border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-sm font-semibold text-slate-700">Auto-detect</span>
        {project && <span className="text-xs text-slate-400">project: {project.label}</span>}
      </div>
      <input
        type="text"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder={projectDefault}
        className="w-full rounded border border-slate-300 px-2 py-1 text-sm focus:border-sky-500 focus:outline-none"
      />
      <div className="mt-2 flex items-center justify-between">
        <span className="text-xs text-slate-500">
          {autoDetect.isPending
            ? 'Running…'
            : autoDetect.data
              ? `${autoDetect.data.length} found last run`
              : 'Phrases separated by "."'}
        </span>
        <button
          type="button"
          onClick={run}
          disabled={autoDetect.isPending || !prompt.trim()}
          className="rounded bg-sky-600 px-3 py-1 text-sm font-medium text-white shadow transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          Run
        </button>
      </div>
      {autoDetect.isError && (
        <div className="mt-2 text-xs text-red-600">{autoDetect.error?.message}</div>
      )}
    </div>
  )
}
