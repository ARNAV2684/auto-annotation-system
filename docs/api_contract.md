# API Contract — Auto Annotation System (v1)

> This is the **single source of truth** for the backend (Sprint 4) and the React
> frontend (Sprint 5). The Pydantic models in `app/api/schemas.py` are the runtime
> implementation of this contract; frontend TypeScript types mirror them.

- **Base URL:** `http://localhost:8000` (dev). Production base configurable via
  `API_HOST` / `API_PORT` env vars.
- **Auth:** none for v1 (single user on localhost).
- **CORS:** allows the Vite dev origin (`http://localhost:5173` by default).
- **Error shape:** `{"detail": "human-readable message"}` (FastAPI default). A
  status code is sent on every response; 4xx for client mistakes, 5xx for server
  bugs. Validation errors come as 422 with FastAPI's automatic detail.
- **Session model:** there is one server-wide session named `"default"`. All
  uploaded images and annotations live in `outputs/sessions/default/`. Multi-user
  is out of scope for v1.
- **Coordinates:** image pixels, origin at top-left, `x` to the right, `y` down.
  Polygons are flat lists `[x1, y1, x2, y2, ...]` in absolute image pixels.
- **Mask wire format:** masks are sent as **polygons**, never raw pixel arrays
  (a 1280×1280 bool mask is 1.6 MB; the same shape as a polygon is ~200 bytes).

---

## Health

### `GET /api/health`
Cheap liveness probe used by the frontend and Docker healthcheck.

**Response 200**
```json
{ "status": "ok", "device": "cuda", "models_loaded": true }
```

---

## Images

### `POST /api/upload`
Upload one image. Sent as `multipart/form-data` with field name `file`.

**Response 201**
```json
{
  "image_id": "img_2026052915xxx",
  "file_name": "drone_001.JPG",
  "width": 1280,
  "height": 1280
}
```

**Errors**
- `400` — unsupported format (only `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif/.tiff`, `.webp`).
- `413` — file too large (configurable, default 20 MB).

### `GET /api/images`
List every image in the current session.

**Response 200**
```json
[
  {
    "image_id": "img_2026052915xxx",
    "file_name": "drone_001.JPG",
    "width": 1280,
    "height": 1280,
    "num_detections": 9,
    "num_approved": 7
  }
]
```

### `GET /api/images/{image_id}/file`
Return the raw image bytes (so the canvas can draw it).

**Response 200** — `Content-Type: image/jpeg` (or whatever was uploaded).

**Errors**
- `404` — unknown `image_id`.

### `POST /api/images/{image_id}/embed`
Tell SAM to compute the image embedding now (so subsequent prompt requests are
near-instant). The frontend may call this when the user opens an image; calling
it is optional — `infer/point` and `infer/box` embed implicitly if needed.

**Response 200**
```json
{ "ok": true, "cached": false }
```

`cached: true` means the embedding was already loaded (no work done).

**Errors**
- `404` — unknown `image_id`.

---

## Inference

> The three inference endpoints below all return one or more **`Annotation`** objects
> and **persist them** to the session. Returned annotations are also added to the
> image's annotation list.

### `Annotation` (response shape, shared by every inference endpoint)
```json
{
  "annotation_id": "ann_2026052915yyy",
  "image_id": "img_2026052915xxx",
  "polygon": [[124, 78, 130, 80, 132, 88, 122, 90]],
  "box": [122, 78, 132, 90],
  "label": "tree",
  "score": 0.834,
  "is_crowd": false,
  "approved": null,
  "source": "auto"
}
```

- `polygon` — list of polygons (a single mask may be multiple disjoint pieces).
  Each polygon is `[x1, y1, x2, y2, ...]` in absolute image pixels.
- `box` — `[x1, y1, x2, y2]` absolute pixels.
- `score` — confidence in `[0, 1]`.
- `approved` — `true` / `false` / `null` (null = unset / pending review).
- `source` — `"auto"` (GroundingDINO+SAM), `"point"` (user click), `"box"` (user drag).

### `POST /api/images/{image_id}/infer/point`
Segment the object at a single click.

**Request body**
```json
{ "x": 450, "y": 320, "label": 1, "multimask": false }
```
- `label` — `1` for foreground (the thing you want), `0` for background (to refine).
- `multimask` — if `true`, returns up to 3 candidate masks ranked by score; if
  `false`, returns the single best one. Default `false`.

**Response 201** — array of one or more `Annotation`s (sorted best-first).

### `POST /api/images/{image_id}/infer/box`
Segment the object inside a bounding box.

**Request body**
```json
{ "x1": 100, "y1": 60, "x2": 280, "y2": 200 }
```

**Response 201** — a single `Annotation`.

### `POST /api/images/{image_id}/auto-detect`
Run GroundingDINO + SAM 2: text → boxes → masks. Replaces any previous
`source: "auto"` annotations for this image (call it again to re-run).

**Request body**
```json
{ "prompts": ["tree", "palm tree"] }
```

**Response 201** — array of `Annotation`s (one per detected object).

---

## Annotations CRUD

### `GET /api/images/{image_id}/annotations`
List every annotation for an image.

**Response 200** — `Annotation[]`.

### `PATCH /api/annotations/{annotation_id}`
Approve, reject, or relabel one annotation. All fields optional; only the ones
sent are updated.

**Request body**
```json
{ "approved": true, "label": "palm_tree", "is_crowd": false }
```

**Response 200** — the updated `Annotation`.

**Errors**
- `404` — unknown `annotation_id`.
- `422` — label not in `config.yaml#classes`.

### `DELETE /api/annotations/{annotation_id}`
Remove an annotation entirely.

**Response 204** — empty.

---

## Export

### `POST /api/export`
Build an export bundle (YOLO / COCO / VOC) and a `.zip`. Blocks for a few
seconds while writing — frontend shows a spinner.

**Request body**
```json
{ "formats": ["yolo", "coco", "voc"], "only_approved": true }
```

- `formats` — subset of `["yolo", "coco", "voc"]`; or `["all"]`.
- `only_approved` — if `true`, exports only `approved: true` annotations. If
  `false`, exports everything (including pending and rejected).

**Response 201**
```json
{
  "run_id": "run_20260530_104215",
  "num_images": 24,
  "num_annotations_exported": 215,
  "download_url": "/api/exports/run_20260530_104215.zip"
}
```

### `GET /api/exports/{run_id}.zip`
Download the bundled zip. `Content-Type: application/zip`.

**Errors**
- `404` — unknown `run_id`.

---

## Stats

### `GET /api/stats`
Overview numbers for the sidebar widget.

**Response 200**
```json
{
  "num_images": 24,
  "num_annotations": 329,
  "num_approved": 215,
  "num_rejected": 12,
  "num_pending": 102,
  "per_class_counts": { "tree": 196, "palm_tree": 129, "shrub": 0 }
}
```

---

## Session admin (v1 nice-to-have)

### `DELETE /api/session`
Wipe the entire current session: deletes all uploaded images, embeddings,
annotations, and exports. Useful between runs.

**Response 204** — empty.

---

## Notes for the frontend (Sprint 5)

- All inference endpoints persist the returned annotation on the server. The UI
  does NOT need to store them locally beyond the in-memory React state.
- Approve / reject / relabel = `PATCH /api/annotations/{id}`. No bulk endpoint
  for v1; the UI sends one PATCH per click. Cheap.
- Re-running auto-detect on an image replaces only its previous `source: "auto"`
  annotations; user-created (`point`, `box`) annotations survive.
- The frontend should poll `GET /api/stats` lazily (on focus, after mutating
  actions). No push for v1.
