// Deterministic class colours.
//
// A label's colour is derived from its text via a stable hash into a fixed
// palette, so "tree" is always the same colour — on the canvas, on the cards,
// and (later) on the Classes panel — with zero per-project setup. The review
// state (pending / approved / rejected) is carried on a *separate* visual
// channel (opacity + dashing), so class and state never compete for hue.

export interface RGB {
  r: number
  g: number
  b: number
}

// 15 distinct, saturated hues (Tailwind 500s). Order is irrelevant — the hash
// spreads labels across the whole palette.
const PALETTE: RGB[] = [
  { r: 239, g: 68, b: 68 }, // red
  { r: 249, g: 115, b: 22 }, // orange
  { r: 234, g: 179, b: 8 }, // yellow
  { r: 132, g: 204, b: 22 }, // lime
  { r: 34, g: 197, b: 94 }, // green
  { r: 16, g: 185, b: 129 }, // emerald
  { r: 20, g: 184, b: 166 }, // teal
  { r: 6, g: 182, b: 212 }, // cyan
  { r: 14, g: 165, b: 233 }, // sky
  { r: 59, g: 130, b: 246 }, // blue
  { r: 99, g: 102, b: 241 }, // indigo
  { r: 139, g: 92, b: 246 }, // violet
  { r: 168, g: 85, b: 247 }, // purple
  { r: 217, g: 70, b: 239 }, // fuchsia
  { r: 236, g: 72, b: 153 }, // pink
]

// Unlabeled / empty objects render in a neutral slate so they read as "needs a
// label" rather than as a real class.
const NEUTRAL: RGB = { r: 148, g: 163, b: 184 } // slate-400

/** djb2 string hash, wrapped to a non-negative 32-bit int. */
function hashString(value: string): number {
  let hash = 5381
  for (let i = 0; i < value.length; i++) {
    hash = ((hash << 5) + hash + value.charCodeAt(i)) | 0
  }
  return Math.abs(hash)
}

/** Stable colour for a class label. Empty / "unlabeled" -> neutral slate. */
export function colorForLabel(label: string): RGB {
  const norm = label.trim().toLowerCase()
  if (!norm || norm === 'unlabeled') return NEUTRAL
  return PALETTE[hashString(norm) % PALETTE.length]
}

export function rgba({ r, g, b }: RGB, alpha: number): string {
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

export function rgbHex({ r, g, b }: RGB): string {
  const hex = (n: number) => n.toString(16).padStart(2, '0')
  return `#${hex(r)}${hex(g)}${hex(b)}`
}
