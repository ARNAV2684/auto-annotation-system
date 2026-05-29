// Tiny fetch wrapper with consistent error handling. Every backend call goes through
// here so we can swap base URL, add auth headers, or log centrally in one place.
//
// All paths are relative ("/api/..."); Vite's dev-server proxy forwards them to the
// FastAPI backend (see vite.config.ts). In Docker the same paths are forwarded by
// nginx, so this code never needs to know the host.

export class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(`API ${status}: ${detail}`)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function unwrap<T>(response: Response): Promise<T> {
  if (response.ok) {
    if (response.status === 204) return undefined as T
    return (await response.json()) as T
  }
  let detail = response.statusText
  try {
    const payload = await response.json()
    detail = payload?.detail ?? detail
  } catch {
    // body was not JSON; keep the status text
  }
  throw new ApiError(response.status, String(detail))
}

export async function getJson<T>(path: string): Promise<T> {
  return unwrap<T>(await fetch(path, { method: 'GET' }))
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  return unwrap<T>(
    await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  )
}

export async function patchJson<T>(path: string, body: unknown): Promise<T> {
  return unwrap<T>(
    await fetch(path, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  )
}

export async function deleteJson<T>(path: string): Promise<T> {
  return unwrap<T>(await fetch(path, { method: 'DELETE' }))
}

export async function postFile<T>(path: string, file: File): Promise<T> {
  const form = new FormData()
  form.append('file', file)
  return unwrap<T>(await fetch(path, { method: 'POST', body: form }))
}

// Absolute URL for resources the browser fetches directly (e.g. <img src=…>).
export function fileUrl(path: string): string {
  return path
}
