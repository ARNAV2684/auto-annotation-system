import { useRef, useState, type ChangeEvent, type DragEvent } from 'react'

interface DropzoneProps {
  onFiles: (files: File[]) => void
  disabled?: boolean
  accept?: string
}

const SUPPORTED_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']

/** Drag-and-drop + click-to-browse area for image uploads. */
export function Dropzone({
  onFiles,
  disabled = false,
  accept = SUPPORTED_EXTS.join(','),
}: DropzoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handle = (files: FileList | null) => {
    if (!files) return
    const valid: File[] = []
    for (const file of files) {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase()
      if (SUPPORTED_EXTS.includes(ext)) {
        valid.push(file)
      }
    }
    if (valid.length) {
      onFiles(valid)
    }
  }

  const onDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    setIsDragging(false)
    if (!disabled) handle(e.dataTransfer.files)
  }

  const onDragOver = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    if (!disabled) setIsDragging(true)
  }

  const onDragLeave = () => setIsDragging(false)

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    handle(e.target.files)
    e.target.value = ''
  }

  const stateClasses = isDragging
    ? 'border-sky-500 bg-sky-50'
    : disabled
      ? 'border-slate-200 bg-slate-50 cursor-not-allowed'
      : 'border-slate-300 bg-white hover:border-sky-400 hover:bg-sky-50 cursor-pointer'

  return (
    <label
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed px-12 py-20 text-center transition ${stateClasses}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple
        disabled={disabled}
        onChange={onChange}
        className="hidden"
      />
      <svg
        className="mb-3 h-12 w-12 text-slate-400"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 7.5m0 0L7.5 12M12 7.5v9"
        />
      </svg>
      <span className="text-base font-medium text-slate-800">
        Drop drone images here, or click to browse
      </span>
      <span className="mt-1 text-sm text-slate-500">JPG · PNG · TIFF · WebP</span>
    </label>
  )
}
