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
  source: 'auto' | 'point' | 'box'
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

export interface AutoDetectRequest {
  prompts: string[]
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
}

export interface ExportResponse {
  run_id: string
  num_images: number
  num_annotations_exported: number
  download_url: string
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
