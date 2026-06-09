import { useState } from 'react'
import type { Annotation } from '../api/types'
import { colorForLabel, rgbHex } from '../utils/classColors'

function ColorDot({ label, muted }: { label: string; muted?: boolean }) {
  return (
    <span
      className="inline-block h-3 w-3 flex-shrink-0 rounded-full"
      style={{ backgroundColor: rgbHex(colorForLabel(label)), opacity: muted ? 0.4 : 1 }}
    />
  )
}

/** Classes tab: every label in use with a live count, plus an "unused" section. */
export function ClassesPanel({
  annotations,
  projectClasses,
}: {
  annotations: Annotation[]
  projectClasses: string[]
}) {
  const counts = new Map<string, number>()
  for (const a of annotations) counts.set(a.label, (counts.get(a.label) ?? 0) + 1)
  const used = [...counts.entries()].sort((a, b) => b[1] - a[1])
  const unused = projectClasses.filter((c) => !counts.has(c))

  return (
    <div className="space-y-3 p-3 text-sm">
      {used.length === 0 ? (
        <p className="text-slate-500">No labels on this image yet.</p>
      ) : (
        <ul className="space-y-1">
          {used.map(([label, n]) => (
            <li key={label} className="flex items-center gap-2">
              <ColorDot label={label} />
              <span className="min-w-0 flex-1 truncate text-slate-700">{label}</span>
              <span className="font-mono text-xs text-slate-500">{n}</span>
            </li>
          ))}
        </ul>
      )}
      {unused.length > 0 && (
        <div>
          <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
            Unused
          </h4>
          <ul className="space-y-1">
            {unused.map((c) => (
              <li key={c} className="flex items-center gap-2">
                <ColorDot label={c} muted />
                <span className="min-w-0 flex-1 truncate text-slate-400">{c}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

/** Layers tab: every annotation instance on the image; click to select. */
export function LayersPanel({
  annotations,
  selectedId,
  onSelect,
}: {
  annotations: Annotation[]
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  if (annotations.length === 0) {
    return <p className="p-3 text-sm text-slate-500">No annotations on this image yet.</p>
  }
  return (
    <ul className="space-y-1 p-2 text-sm">
      {annotations.map((a) => (
        <li key={a.annotation_id}>
          <button
            type="button"
            onClick={() => onSelect(a.annotation_id)}
            className={`flex w-full items-center gap-2 rounded px-2 py-1 text-left transition ${
              a.annotation_id === selectedId
                ? 'bg-sky-100 ring-1 ring-sky-300'
                : 'hover:bg-slate-100'
            }`}
          >
            <ColorDot label={a.label} />
            <span className="min-w-0 flex-1 truncate text-slate-700">{a.label}</span>
            <span className="font-mono text-xs text-slate-400">{(a.score * 100).toFixed(0)}%</span>
          </button>
        </li>
      ))}
    </ul>
  )
}

/** Tags tab: image-level tags with add / remove. */
export function TagsPanel({
  tags,
  onChange,
}: {
  tags: string[]
  onChange: (tags: string[]) => void
}) {
  const [draft, setDraft] = useState('')
  const add = () => {
    const cleaned = draft.trim()
    if (!cleaned || tags.includes(cleaned)) {
      setDraft('')
      return
    }
    onChange([...tags, cleaned])
    setDraft('')
  }
  return (
    <div className="space-y-3 p-3 text-sm">
      <div className="flex gap-1">
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') add()
          }}
          placeholder="Add tag…"
          className="min-w-0 flex-1 rounded border border-slate-300 px-2 py-1 text-sm"
        />
        <button
          type="button"
          onClick={add}
          className="rounded bg-slate-800 px-2 py-1 text-xs font-medium text-white hover:bg-slate-700"
        >
          Add
        </button>
      </div>
      {tags.length === 0 ? (
        <p className="text-slate-500">No tags on this image.</p>
      ) : (
        <ul className="flex flex-wrap gap-1">
          {tags.map((tag) => (
            <li
              key={tag}
              className="flex items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-700"
            >
              {tag}
              <button
                type="button"
                onClick={() => onChange(tags.filter((t) => t !== tag))}
                className="text-slate-400 hover:text-slate-700"
                title="Remove tag"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/** Raw Data tab: the annotations JSON for the current image. */
export function RawDataPanel({ annotations }: { annotations: Annotation[] }) {
  return (
    <pre className="overflow-auto p-3 font-mono text-xs leading-relaxed text-slate-700">
      {JSON.stringify(annotations, null, 2)}
    </pre>
  )
}
