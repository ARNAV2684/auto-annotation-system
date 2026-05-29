import { useEffect, useState } from 'react'

/**
 * Load an image URL into an ``HTMLImageElement``; returns ``null`` until it's loaded.
 *
 * This is a standard async-resource-loading effect: a side effect (creating an
 * Image, attaching listeners) followed by setState when the browser reports
 * load/error. The React-19 ``react-hooks/set-state-in-effect`` rule flags these
 * setState calls — we disable it here because the pattern is correct and there
 * is no derive-during-render alternative for resource loading.
 */
export function useImageElement(src: string | null): HTMLImageElement | null {
  const [image, setImage] = useState<HTMLImageElement | null>(null)

  useEffect(() => {
    if (!src) {
      // Sync reset when src goes null; the rule otherwise flags this.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setImage(null)
      return
    }
    const element = new Image()
    element.src = src
    const handleLoad = () => setImage(element)
    const handleError = () => setImage(null)
    element.addEventListener('load', handleLoad)
    element.addEventListener('error', handleError)
    return () => {
      element.removeEventListener('load', handleLoad)
      element.removeEventListener('error', handleError)
    }
  }, [src])

  return image
}
