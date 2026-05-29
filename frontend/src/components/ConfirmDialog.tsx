interface ConfirmDialogProps {
  open: boolean
  title: string
  message: string
  confirmLabel?: string
  confirmDanger?: boolean
  onConfirm: () => void
  onClose: () => void
}

/** Generic confirm modal. Render conditionally - it returns null when closed. */
export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = 'Confirm',
  confirmDanger = false,
  onConfirm,
  onClose,
}: ConfirmDialogProps) {
  if (!open) return null

  const confirmClasses = confirmDanger
    ? 'bg-red-600 hover:bg-red-500 focus:ring-red-500'
    : 'bg-sky-600 hover:bg-sky-500 focus:ring-sky-500'

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4"
      onClick={onClose}
    >
      <div
        className="w-[26rem] rounded-lg bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
        <p className="mt-2 text-sm text-slate-600">{message}</p>
        <div className="mt-6 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => {
              onConfirm()
              onClose()
            }}
            className={`rounded px-4 py-1.5 text-sm font-medium text-white shadow transition focus:ring-2 focus:ring-offset-1 focus:outline-none ${confirmClasses}`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
