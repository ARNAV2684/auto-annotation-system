import { useImages } from '../api/hooks'
import type { ImageRecord } from '../api/types'

interface ImageListProps {
  selectedId: string | null
  onSelect: (imageId: string) => void
}

/** Left sidebar of clickable thumbnails. Re-uses the cached useImages() query. */
export function ImageList({ selectedId, onSelect }: ImageListProps) {
  const { data: images, isLoading } = useImages()

  if (isLoading) {
    return <div className="p-4 text-sm text-slate-500">Loading images…</div>
  }
  if (!images || images.length === 0) {
    return <div className="p-4 text-sm text-slate-500">No images.</div>
  }

  return (
    <ul className="divide-y divide-slate-200">
      {images.map((image) => (
        <ImageRow
          key={image.image_id}
          image={image}
          selected={image.image_id === selectedId}
          onClick={() => onSelect(image.image_id)}
        />
      ))}
    </ul>
  )
}

function ImageRow({
  image,
  selected,
  onClick,
}: {
  image: ImageRecord
  selected: boolean
  onClick: () => void
}) {
  const stateClasses = selected
    ? 'bg-sky-50 border-l-4 border-l-sky-500'
    : 'hover:bg-slate-50 border-l-4 border-l-transparent'

  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        className={`flex w-full items-center gap-3 px-3 py-2 text-left transition ${stateClasses}`}
      >
        <img
          src={`/api/images/${image.image_id}/file`}
          alt={image.file_name}
          loading="lazy"
          className="h-12 w-12 flex-shrink-0 rounded border border-slate-200 object-cover"
        />
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium text-slate-800" title={image.file_name}>
            {image.file_name}
          </div>
          <div className="mt-0.5 flex items-center gap-2 text-xs text-slate-500">
            <span>{image.num_detections} det</span>
            {image.num_approved > 0 && (
              <span className="text-emerald-600">{image.num_approved} ✓</span>
            )}
          </div>
        </div>
      </button>
    </li>
  )
}
