import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import {
  qk,
  useDeployModelVersion,
  useModels,
  useStartTraining,
  useTrainStatus,
} from '../api/hooks'
import type { ModelVersion, Project, TrainStatus } from '../api/types'

interface TrainModalProps {
  /** Active project; training is offered when it has a training track. */
  project: Project | null
  onClose: () => void
}

const PHASE_LABELS: Record<string, string> = {
  preparing: 'Preparing the dataset',
  training: 'Training',
  evaluating: 'Evaluating on the test split',
  gating: 'Comparing against the live model',
}

/**
 * The TRAIN dialog: start a fine-tuning job, watch its progress, and browse
 * the version registry ("Your models") with one-click rollback.
 *
 * The eval gate lives server-side — a finished job reports whether the new
 * version beat the live model and was deployed, or was recorded but rejected.
 */
export function TrainModal({ project, onClose }: TrainModalProps) {
  const track = project?.trainable ? project.id : null
  const status = useTrainStatus(track)
  const models = useModels(track)
  const start = useStartTraining(track ?? '')
  const deploy = useDeployModelVersion(track ?? '')

  // When a job finishes, the registry + the deployed model may have changed.
  const queryClient = useQueryClient()
  const state = status.data?.state
  useEffect(() => {
    if (track && (state === 'done' || state === 'failed')) {
      queryClient.invalidateQueries({ queryKey: qk.models(track) })
      queryClient.invalidateQueries({ queryKey: qk.projects })
    }
  }, [state, track, queryClient])

  const running = state === 'running' || start.isPending

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="max-h-[85vh] w-[32rem] overflow-auto rounded-lg bg-white p-6 shadow-xl"
      >
        <div className="flex items-baseline justify-between">
          <h2 className="text-lg font-semibold text-slate-900">Train a model</h2>
          {project && <span className="text-xs text-slate-400">project: {project.label}</span>}
        </div>

        {!track ? (
          <p className="mt-3 text-sm text-slate-600">
            This project has no training dataset configured yet — only Road Defects can be
            custom-trained today. Keep labeling and approving; that work becomes training data once
            a dataset exists for this project.
          </p>
        ) : (
          <>
            <p className="mt-2 text-sm text-slate-600">
              Fine-tunes on the base dataset <span className="font-medium">plus</span> every image
              you marked <span className="font-medium">done</span> that has approved labels. The new
              model is deployed <span className="font-medium">only if it beats the live one</span>{' '}
              on the held-out test split.
            </p>
            <p className="mt-2 rounded border border-amber-200 bg-amber-50 px-2 py-1.5 text-xs text-amber-800">
              Training shares the GPU with segmentation — expect auto-detect and clicks to be slow
              (or fail) while a job runs. It takes roughly 20 minutes.
            </p>

            <div className="mt-4">
              <StatusSection status={status.data} startError={start.error} />
            </div>

            {start.data && (
              <p className="mt-2 text-xs text-slate-500">
                Snapshot: {start.data.num_images} reviewed image(s) / {start.data.num_annotations}{' '}
                approved annotation(s) added to the training set.
              </p>
            )}

            <button
              type="button"
              onClick={() => start.mutate()}
              disabled={running}
              className="mt-3 w-full rounded bg-sky-600 px-4 py-2 text-sm font-medium text-white shadow transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {running ? 'Training in progress…' : 'Start training'}
            </button>

            <div className="mt-5">
              <h3 className="text-sm font-semibold text-slate-700">Your models</h3>
              {!models.data?.versions.length ? (
                <p className="mt-1 text-xs text-slate-500">
                  No versions yet — the first training run records the current model as v1 and the
                  new one as v2.
                </p>
              ) : (
                <ul className="mt-2 space-y-1.5">
                  {[...models.data.versions].reverse().map((v) => (
                    <VersionRow
                      key={v.version}
                      version={v}
                      onDeploy={() => deploy.mutate(v.version)}
                      deploying={deploy.isPending}
                    />
                  ))}
                </ul>
              )}
              {deploy.isError && (
                <p className="mt-1 text-xs text-red-600">{deploy.error?.message}</p>
              )}
            </div>
          </>
        )}

        <div className="mt-5 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="rounded bg-slate-900 px-4 py-1.5 text-sm font-medium text-white hover:bg-slate-700"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

function StatusSection({
  status,
  startError,
}: {
  status: TrainStatus | undefined
  startError: Error | null
}) {
  if (startError) {
    return <p className="text-sm text-red-600">{startError.message}</p>
  }
  if (!status || status.state === 'idle') {
    return <p className="text-sm text-slate-500">No training job has run yet.</p>
  }
  if (status.state === 'running') {
    const phase = status.phase ? (PHASE_LABELS[status.phase] ?? status.phase) : '…'
    const total = status.epochs_total ?? 0
    const done = status.epochs_done ?? 0
    const pct = total > 0 ? Math.round((done / total) * 100) : 0
    return (
      <div>
        <p className="text-sm font-medium text-slate-700">
          {phase}
          <span className="ml-2 inline-block h-3 w-3 animate-spin rounded-full border-2 border-sky-500 border-t-transparent align-middle" />
        </p>
        {status.message && <p className="mt-0.5 text-xs text-slate-500">{status.message}</p>}
        {status.phase === 'training' && total > 0 && (
          <div className="mt-2">
            <div className="h-2 w-full overflow-hidden rounded bg-slate-200">
              <div className="h-full bg-sky-500 transition-all" style={{ width: `${pct}%` }} />
            </div>
            <p className="mt-0.5 text-xs text-slate-500">
              Epoch {done} / {total}
            </p>
          </div>
        )}
      </div>
    )
  }
  if (status.state === 'failed') {
    return (
      <p className="rounded border border-red-200 bg-red-50 px-2 py-1.5 text-sm text-red-700">
        Training failed: {status.message ?? 'unknown error'}
      </p>
    )
  }
  // done
  const result = status.result
  const candidate = result?.candidate?.seg_map50
  const baseline = result?.baseline?.seg_map50
  return (
    <div
      className={`rounded border px-2 py-1.5 text-sm ${
        result?.deployed
          ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
          : 'border-amber-200 bg-amber-50 text-amber-800'
      }`}
    >
      <p className="font-medium">{status.message}</p>
      {candidate !== undefined && (
        <p className="mt-0.5 text-xs">
          New model mask mAP50: {candidate.toFixed(3)}
          {baseline !== undefined && ` — live model: ${baseline.toFixed(3)}`}
        </p>
      )}
    </div>
  )
}

function VersionRow({
  version,
  onDeploy,
  deploying,
}: {
  version: ModelVersion
  onDeploy: () => void
  deploying: boolean
}) {
  const map50 = version.metrics.seg_map50
  return (
    <li className="flex items-center justify-between rounded border border-slate-200 bg-slate-50 px-2 py-1.5 text-xs">
      <div>
        <span className="font-semibold text-slate-700">v{version.version}</span>
        <span className="ml-2 text-slate-500">
          mask mAP50 {map50 !== undefined ? map50.toFixed(3) : '—'}
        </span>
        <span className="ml-2 text-slate-400">
          {version.created_at.slice(0, 16).replace('T', ' ')}
        </span>
        {version.note && <div className="text-slate-400">{version.note}</div>}
      </div>
      {version.deployed ? (
        <span className="rounded bg-emerald-100 px-2 py-0.5 font-medium text-emerald-700">
          live
        </span>
      ) : (
        <button
          type="button"
          onClick={onDeploy}
          disabled={deploying}
          title="Make this version the live model (rollback / promote)"
          className="rounded border border-slate-300 bg-white px-2 py-0.5 font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-50"
        >
          Make live
        </button>
      )}
    </li>
  )
}
