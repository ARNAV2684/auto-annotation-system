// Non-standard <input> attributes used for the folder picker. React's built-in
// typings don't include them, so we augment InputHTMLAttributes here.
import 'react'

declare module 'react' {
  // The generic must match React's declaration for merging, even though we
  // don't use it here.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface InputHTMLAttributes<T> {
    webkitdirectory?: string
    directory?: string
  }
}
