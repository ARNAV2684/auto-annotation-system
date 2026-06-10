// TypeScript mirror of `app/api/schemas.py`. The backend's Pydantic models are the
// source of truth; if anything drifts here, the API contract doc + Pydantic win.
//
// A polygon is a flat list of pixel coordinates: [x1, y1, x2, y2, ...].
// A mask may have one or more disjoint polygons.

export type Polygon = number[]
export type Box = [number, number, number, number]

export interface HealthResponse {
  status: 'ok' | 'degraded'
  device: string
  models_loaded: boolean
}

export interface ImageRecord {
  image_id: string
  file_name: string
  width: number
  height: number
  num_detections: number
  num_approved: number
  done: boolean
  tags: string[]
}

export interface ImagePatch {
  done?: boolean
  tags?: string[]
}

export interface EmbedResponse {
  ok: boolean
  cached: boolean
}

export interface Annotation {
  annotation_id: string
  image_id: string
  polygon: Polygon[]
  box: Box
  label: string
  score: number
  is_crowd: boolean
  approved: boolean | null
  source: 'auto' | 'point' | 'box' | 'polygon'
}

export interface PointRequest {
  x: number
  y: number
  label?: 0 | 1
  multimask?: boolean
}

export interface BoxRequest {
  x1: number
  y1: number
  x2: number
  y2: number
}

/** One include/exclude point for Smart Select. */
export interface PointSpec {
  x: number
  y: number
  label: 0 | 1
}

/** A Smart-Select object collected (but not yet committed): its click point
 *  plus the previewed outline shown on the canvas. */
export interface SmartObject {
  x: number
  y: number
  polygon: Polygon[]
}

export interface MultiPointRequest {
  points: PointSpec[]
  multimask?: boolean
  simplify?: number | null
}

export interface PreviewResponse {
  polygon: Polygon[]
  box: Box
  score: number
}

export interface PolygonRequest {
  points: [number, number][]
  label?: string | null
}

export interface AutoDetectRequest {
  prompts: string[]
  /** Active project id; routes detection to its trained model if deployed. */
  project_id?: string
}

export interface AnnotationPatch {
  approved?: boolean | null
  label?: string
  is_crowd?: boolean
}

export type ExportFormat = 'yolo' | 'coco' | 'voc' | 'all'

export interface ExportRequest {
  formats: ExportFormat[]
  only_approved: boolean
  /** Scope export to one project's class list. null/omitted = union of all projects. */
  project_id?: string | null
}

export interface ExportResponse {
  run_id: string
  num_images: number
  num_annotations_exported: number
  download_url: string
  num_dropped: number
  dropped_labels: string[]
}

export interface StatsResponse {
  num_images: number
  num_annotations: number
  num_approved: number
  num_rejected: number
  num_pending: number
  per_class_counts: Record<string, number>
}

/** A UI preset: named class list + default detection prompts. */
export interface Project {
  id: string
  label: string
  classes: string[]
  default_prompts: string[]
  /** True when auto-detect runs a deployed fine-tuned model (prompts unused). */
  has_trained_model?: boolean
  /** True when this project has a training track (TRAIN can launch a job). */
  trainable?: boolean
}

// ─── Training jobs + model registry (Sprint 7) ────────────────────────────

export interface TrainStartResponse {
  job_id: string
  state: string
  num_images: number
  num_annotations: number
}

export interface TrainResult {
  version: number
  deployed: boolean
  candidate: Record<string, number>
  baseline: Record<string, number> | null
  counts?: Record<string, number> | null
}

export interface TrainStatus {
  state: 'idle' | 'running' | 'done' | 'failed'
  phase?: string | null
  message?: string | null
  job_id?: string | null
  started_at?: string | null
  finished_at?: string | null
  epochs_total?: number | null
  epochs_done?: number | null
  result?: TrainResult | null
}

export interface ModelVersion {
  version: number
  created_at: string
  note: string
  metrics: Record<string, number>
  deployed: boolean
}

export interface ModelsResponse {
  track: string
  deployed_version: number | null
  versions: ModelVersion[]
}

export interface CreateProjectRequest {
  label: string
  classes?: string[]
  default_prompts?: string[]
}

export interface AddClassRequest {
  name: string
}

export type BulkAction = 'approve' | 'reject' | 'delete'

export interface BulkAnnotationRequest {
  action: BulkAction
  annotation_ids: string[]
}

export interface BulkAnnotationError {
  annotation_id: string
  reason: string
}

export interface BulkAnnotationResponse {
  updated: number
  errors: BulkAnnotationError[]
}
