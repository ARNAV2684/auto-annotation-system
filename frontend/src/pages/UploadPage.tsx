import { useState } from 'react'
import { useUpload } from '../api/hooks'
import { ApiError } from '../api/client'
import { Dropzone } from '../components/Dropzone'

interface UploadPageProps {
  imageCount: number
  onStartReview: () => void
}

interface UploadProgress {
  fileName: string
  status: 'pending' | 'done' | 'failed'
  error?: string
}

/** Drop zone + per-file progress + "Start Review" once any image is uploaded. */
export function UploadPage({ imageCount, onStartReview }: UploadPageProps) {
  const upload = useUpload()
  const [pending, setPending] = useState<UploadProgress[]>([])

  const handleFiles = async (files: File[]) => {
    setPending(files.map((f) => ({ fileName: f.name, status: 'pending' })))
    // Fire all uploads in parallel; record results as they resolve.
    await Promise.all(
      files.map((file) =>
        upload
          .mutateAsync(file)
          .then(() =>
            setPending((rows) =>
              rows.map((r) => (r.fileName === file.name ? { ...r, status: 'done' } : r)),
            ),
          )
          .catch((err: unknown) => {
            const message = err instanceof ApiError ? err.detail : 'Upload failed'
            setPending((rows) =>
              rows.map((r) =>
                r.fileName === file.name ? { ...r, status: 'failed', error: message } : r,
              ),
            )
          }),
      ),
    )
  }

  const inFlight = pending.some((r) => r.status === 'pending')
  const numDone = pending.filter((r) => r.status === 'done').length
  const numFailed = pending.filter((r) => r.status === 'failed').length

  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <Dropzone onFiles={handleFiles} disabled={inFlight} />

      {pending.length > 0 && (
        <div className="mt-8 rounded border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-2 flex items-baseline justify-between">
            <span className="text-sm font-semibold text-slate-700">
              Uploading {numDone} / {pending.length}
              {numFailed > 0 && <span className="ml-2 text-red-600">({numFailed} failed)</span>}
            </span>
            <span className="text-xs text-slate-500">{inFlight ? 'in progress…' : 'done'}</span>
          </div>
          <ul className="max-h-48 overflow-auto text-sm">
            {pending.map((row) => (
              <li
                key={row.fileName}
                className="flex items-center justify-between border-t border-slate-100 px-1 py-1.5"
              >
                <span className="truncate text-slate-700">{row.fileName}</span>
                <StatusBadge status={row.status} error={row.error} />
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="mt-8 flex items-center justify-between">
        <p className="text-sm text-slate-600">
          {imageCount === 0
            ? 'No images uploaded yet.'
            : `${imageCount} image${imageCount === 1 ? '' : 's'} in this session.`}
        </p>
        <button
          type="button"
          onClick={onStartReview}
          disabled={imageCount === 0 || inFlight}
          className="rounded bg-sky-600 px-4 py-2 text-sm font-medium text-white shadow transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          Start Review →
        </button>
      </div>
    </div>
  )
}

function StatusBadge({ status, error }: { status: UploadProgress['status']; error?: string }) {
  if (status === 'done') {
    return <span className="text-xs font-medium text-emerald-600">✓ uploaded</span>
  }
  if (status === 'failed') {
    return (
      <span className="text-xs font-medium text-red-600" title={error}>
        ✗ {error ?? 'failed'}
      </span>
    )
  }
  return <span className="text-xs text-slate-500">…</span>
}
