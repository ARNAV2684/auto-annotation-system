import { useState } from 'react'
import { useAddClass, useCreateProject, useProjects } from '../api/hooks'
import { useCurrentProject } from './useCurrentProject'

const NEW_GROUP = '__new__'

interface InlineClassPopupProps {
  open: boolean
  onClose: () => void
  /** Called with the raw class name once it is registered, so the caller can
   *  apply it as the annotation's label (the PATCH route normalises it the
   *  same way the store did, so the two always agree). */
  onRegistered: (className: string) => void
}

/**
 * Small modal for custom labeling: name a class (e.g. "house") and assign it to
 * a group (project) — an existing one, or a brand-new group created on the spot.
 * The class is registered server-side so it persists and becomes exportable.
 */
export function InlineClassPopup({ open, onClose, onRegistered }: InlineClassPopupProps) {
  const { data: projects } = useProjects()
  const { current } = useCurrentProject()
  const createProject = useCreateProject()

  const [className, setClassName] = useState('')
  const [target, setTarget] = useState('') // project id, or NEW_GROUP
  const [newGroup, setNewGroup] = useState('')

  // Seed the target group to the active project once it is known.
  const [seeded, setSeeded] = useState(false)
  if (!seeded && current) {
    setSeeded(true)
    setTarget(current.id)
  }

  // Bound to the selected existing group; harmless when target is NEW_GROUP/''
  // because we only fire it on the existing-group path.
  const addClass = useAddClass(target === NEW_GROUP ? '' : target)

  if (!open) return null

  const isNew = target === NEW_GROUP
  const busy = createProject.isPending || addClass.isPending
  const validTarget = isNew ? newGroup.trim().length > 0 : target.length > 0
  const canSubmit = className.trim().length > 0 && validTarget && !busy
  const error = createProject.error?.message ?? addClass.error?.message

  const reset = () => {
    setClassName('')
    setNewGroup('')
  }

  const handleClose = () => {
    reset()
    onClose()
  }

  const submit = async () => {
    const name = className.trim()
    if (!name || !validTarget) return
    try {
      if (isNew) {
        await createProject.mutateAsync({ label: newGroup.trim(), classes: [name] })
      } else {
        await addClass.mutateAsync({ name })
      }
      onRegistered(name)
      reset()
      onClose()
    } catch {
      // Error surfaced below; keep the dialog open.
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4"
      onClick={handleClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-[24rem] rounded-lg bg-white p-5 shadow-xl"
      >
        <h2 className="text-base font-semibold text-slate-900">New class</h2>
        <p className="mt-1 text-xs text-slate-500">
          Name the object and assign it to a group. It&apos;s saved for reuse and export.
        </p>

        <label className="mt-4 block text-sm font-medium text-slate-700">Class name</label>
        <input
          autoFocus
          type="text"
          value={className}
          onChange={(e) => setClassName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && canSubmit) submit()
          }}
          placeholder="e.g. house"
          className="mt-1 w-full rounded border border-slate-300 px-2 py-1.5 text-sm text-slate-800"
        />

        <label className="mt-4 block text-sm font-medium text-slate-700">Assign to group</label>
        <select
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          className="mt-1 w-full rounded border border-slate-300 px-2 py-1.5 text-sm text-slate-800"
        >
          {(projects ?? []).map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
          <option value={NEW_GROUP}>+ new group…</option>
        </select>

        {isNew && (
          <input
            type="text"
            value={newGroup}
            onChange={(e) => setNewGroup(e.target.value)}
            placeholder="New group name, e.g. Buildings"
            className="mt-2 w-full rounded border border-slate-300 px-2 py-1.5 text-sm text-slate-800"
          />
        )}

        {error && (
          <div className="mt-3 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        <div className="mt-5 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={handleClose}
            disabled={busy}
            className="rounded px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={submit}
            disabled={!canSubmit}
            className="rounded bg-sky-600 px-4 py-1.5 text-sm font-medium text-white shadow transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {busy ? 'Saving…' : 'Add class'}
          </button>
        </div>
      </div>
    </div>
  )
}
