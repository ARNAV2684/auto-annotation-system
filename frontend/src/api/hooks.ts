// TanStack Query hooks around every backend route.
//
// Read hooks (`useQuery`) cache the response in browser memory and refetch when stale.
// Write hooks (`useMutation`) call the backend, then invalidate the read caches that
// depend on the data they changed — that's what makes the UI auto-refresh after
// upload / inference / approve.

import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationResult,
  type UseQueryResult,
} from '@tanstack/react-query'
import { deleteJson, getJson, patchJson, postFile, postJson } from './client'
import type {
  AddClassRequest,
  Annotation,
  AnnotationPatch,
  AutoDetectRequest,
  BoxRequest,
  BulkAnnotationRequest,
  BulkAnnotationResponse,
  CreateProjectRequest,
  EmbedResponse,
  ExportRequest,
  ExportResponse,
  HealthResponse,
  ImagePatch,
  ImageRecord,
  ModelsResponse,
  MultiPointRequest,
  PointRequest,
  PolygonRequest,
  Project,
  StatsResponse,
  TrainStartResponse,
  TrainStatus,
} from './types'

// ─── query keys (centralised so invalidation is type-safe) ───────────────

export const qk = {
  health: ['health'] as const,
  images: ['images'] as const,
  annotationsFor: (imageId: string) => ['annotations', imageId] as const,
  stats: ['stats'] as const,
  projects: ['projects'] as const,
  trainStatus: (track: string) => ['train-status', track] as const,
  models: (track: string) => ['models', track] as const,
}

// ─── reads ───────────────────────────────────────────────────────────────

export function useHealth(): UseQueryResult<HealthResponse> {
  return useQuery({
    queryKey: qk.health,
    queryFn: () => getJson<HealthResponse>('/api/health'),
    refetchInterval: 10_000,
  })
}

export function useImages(): UseQueryResult<ImageRecord[]> {
  return useQuery({
    queryKey: qk.images,
    queryFn: () => getJson<ImageRecord[]>('/api/images'),
  })
}

export function useAnnotations(imageId: string | null): UseQueryResult<Annotation[]> {
  return useQuery({
    queryKey: imageId ? qk.annotationsFor(imageId) : ['annotations', 'none'],
    queryFn: () => getJson<Annotation[]>(`/api/images/${imageId}/annotations`),
    enabled: !!imageId,
  })
}

export function useStats(): UseQueryResult<StatsResponse> {
  return useQuery({
    queryKey: qk.stats,
    queryFn: () => getJson<StatsResponse>('/api/stats'),
  })
}

export function useProjects(): UseQueryResult<Project[]> {
  return useQuery({
    queryKey: qk.projects,
    queryFn: () => getJson<Project[]>('/api/projects'),
    // Projects come from config and rarely change while the app runs.
    staleTime: 5 * 60 * 1000,
  })
}

export function useCreateProject(): UseMutationResult<Project, Error, CreateProjectRequest> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body) => postJson<Project>('/api/projects', body),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.projects }),
  })
}

export function useAddClass(projectId: string): UseMutationResult<Project, Error, AddClassRequest> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body) => postJson<Project>(`/api/projects/${projectId}/classes`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.projects }),
  })
}

// ─── writes ──────────────────────────────────────────────────────────────

export function useUpload(): UseMutationResult<ImageRecord, Error, File> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (file: File) => postFile<ImageRecord>('/api/upload', file),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.images })
      qc.invalidateQueries({ queryKey: qk.stats })
    },
  })
}

export function useUploadZip(): UseMutationResult<ImageRecord[], Error, File> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (file: File) => postFile<ImageRecord[]>('/api/upload-zip', file),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.images })
      qc.invalidateQueries({ queryKey: qk.stats })
    },
  })
}

export function usePatchImage(imageId: string): UseMutationResult<ImageRecord, Error, ImagePatch> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body) => patchJson<ImageRecord>(`/api/images/${imageId}`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.images }),
  })
}

export function useEmbed(): UseMutationResult<EmbedResponse, Error, string> {
  return useMutation({
    mutationFn: (imageId: string) => postJson<EmbedResponse>(`/api/images/${imageId}/embed`, {}),
  })
}

export function useInferPoint(
  imageId: string,
): UseMutationResult<Annotation[], Error, PointRequest> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body) => postJson<Annotation[]>(`/api/images/${imageId}/infer/point`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.annotationsFor(imageId) })
      qc.invalidateQueries({ queryKey: qk.images })
      qc.invalidateQueries({ queryKey: qk.stats })
    },
  })
}

export function useInferPoints(
  imageId: string,
): UseMutationResult<Annotation[], Error, MultiPointRequest> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body) => postJson<Annotation[]>(`/api/images/${imageId}/infer/points`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.annotationsFor(imageId) })
      qc.invalidateQueries({ queryKey: qk.images })
      qc.invalidateQueries({ queryKey: qk.stats })
    },
  })
}

export function useCreatePolygon(
  imageId: string,
): UseMutationResult<Annotation, Error, PolygonRequest> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body) => postJson<Annotation>(`/api/images/${imageId}/annotations/polygon`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.annotationsFor(imageId) })
      qc.invalidateQueries({ queryKey: qk.images })
      qc.invalidateQueries({ queryKey: qk.stats })
    },
  })
}

export function useCopyAnnotations(
  imageId: string,
): UseMutationResult<Annotation[], Error, string> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (sourceId) =>
      postJson<Annotation[]>(`/api/images/${imageId}/annotations/copy-from/${sourceId}`, {}),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.annotationsFor(imageId) })
      qc.invalidateQueries({ queryKey: qk.images })
      qc.invalidateQueries({ queryKey: qk.stats })
    },
  })
}

export function useInferBox(imageId: string): UseMutationResult<Annotation, Error, BoxRequest> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body) => postJson<Annotation>(`/api/images/${imageId}/infer/box`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.annotationsFor(imageId) })
      qc.invalidateQueries({ queryKey: qk.images })
      qc.invalidateQueries({ queryKey: qk.stats })
    },
  })
}

export function useAutoDetect(
  imageId: string,
): UseMutationResult<Annotation[], Error, AutoDetectRequest> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body) => postJson<Annotation[]>(`/api/images/${imageId}/auto-detect`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.annotationsFor(imageId) })
      qc.invalidateQueries({ queryKey: qk.images })
      qc.invalidateQueries({ queryKey: qk.stats })
    },
  })
}

export function usePatchAnnotation(): UseMutationResult<
  Annotation,
  Error,
  { annotationId: string; patch: AnnotationPatch }
> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ annotationId, patch }) =>
      patchJson<Annotation>(`/api/annotations/${annotationId}`, patch),
    onSuccess: (annotation) => {
      qc.invalidateQueries({ queryKey: qk.annotationsFor(annotation.image_id) })
      qc.invalidateQueries({ queryKey: qk.images })
      qc.invalidateQueries({ queryKey: qk.stats })
    },
  })
}

export function useDeleteAnnotation(): UseMutationResult<
  void,
  Error,
  { annotationId: string; imageId: string }
> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ annotationId }) => deleteJson<void>(`/api/annotations/${annotationId}`),
    onSuccess: (_void, { imageId }) => {
      qc.invalidateQueries({ queryKey: qk.annotationsFor(imageId) })
      qc.invalidateQueries({ queryKey: qk.images })
      qc.invalidateQueries({ queryKey: qk.stats })
    },
  })
}

export function useBulkAnnotations(
  imageId: string,
): UseMutationResult<BulkAnnotationResponse, Error, BulkAnnotationRequest> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body) =>
      postJson<BulkAnnotationResponse>(`/api/images/${imageId}/annotations/bulk`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.annotationsFor(imageId) })
      qc.invalidateQueries({ queryKey: qk.images })
      qc.invalidateQueries({ queryKey: qk.stats })
    },
  })
}

export function useExport(): UseMutationResult<ExportResponse, Error, ExportRequest> {
  return useMutation({
    mutationFn: (body) => postJson<ExportResponse>('/api/export', body),
  })
}

export function useDeleteSession(): UseMutationResult<void, Error, void> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => deleteJson<void>('/api/session'),
    onSuccess: () => {
      qc.invalidateQueries() // session wipe nukes everything
    },
  })
}

// ─── Training jobs + model registry (Sprint 7) ────────────────────────────

export function useTrainStatus(track: string | null): UseQueryResult<TrainStatus> {
  return useQuery({
    queryKey: track ? qk.trainStatus(track) : ['train-status', 'none'],
    queryFn: () => getJson<TrainStatus>(`/api/train/${track}/status`),
    enabled: !!track,
    // Poll while a job is live so the modal shows phase + epoch progress.
    refetchInterval: (query) => (query.state.data?.state === 'running' ? 3000 : false),
  })
}

export function useModels(track: string | null): UseQueryResult<ModelsResponse> {
  return useQuery({
    queryKey: track ? qk.models(track) : ['models', 'none'],
    queryFn: () => getJson<ModelsResponse>(`/api/models/${track}`),
    enabled: !!track,
  })
}

export function useStartTraining(
  track: string,
): UseMutationResult<TrainStartResponse, Error, void> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => postJson<TrainStartResponse>(`/api/train/${track}`, {}),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.trainStatus(track) })
      qc.invalidateQueries({ queryKey: qk.models(track) })
    },
  })
}

export function useDeployModelVersion(
  track: string,
): UseMutationResult<ModelsResponse, Error, number> {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (version) => postJson<ModelsResponse>(`/api/models/${track}/deploy/${version}`, {}),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.models(track) })
      // deployed.pt changed -> has_trained_model may flip; refresh projects.
      qc.invalidateQueries({ queryKey: qk.projects })
    },
  })
}
