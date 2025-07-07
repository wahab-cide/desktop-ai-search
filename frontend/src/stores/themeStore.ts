import { createSignal } from 'solid-js'

export type Theme = 'light' | 'dark'

const [theme, setTheme] = createSignal<Theme>('light') // Start with light mode

export { theme, setTheme }

export const toggleTheme = () => {
  setTheme(theme() === 'light' ? 'dark' : 'light')
}

// Theme-aware class helpers
export const themeClasses = {
  // Main backgrounds
  bg: () => theme() === 'light' ? 'bg-white' : 'bg-black',
  bgSecondary: () => theme() === 'light' ? 'bg-gray-50' : 'bg-zinc-900/80',
  bgTertiary: () => theme() === 'light' ? 'bg-gray-100' : 'bg-zinc-900/60',
  
  // Text colors
  text: () => theme() === 'light' ? 'text-gray-900' : 'text-white',
  textSecondary: () => theme() === 'light' ? 'text-gray-600' : 'text-zinc-300',
  textMuted: () => theme() === 'light' ? 'text-gray-500' : 'text-zinc-400',
  textSubtle: () => theme() === 'light' ? 'text-gray-400' : 'text-zinc-500',
  
  // Borders
  border: () => theme() === 'light' ? 'border-gray-200' : 'border-zinc-800/60',
  borderSecondary: () => theme() === 'light' ? 'border-gray-300' : 'border-zinc-700/50',
  
  // Interactive elements
  hover: () => theme() === 'light' ? 'hover:bg-gray-100' : 'hover:bg-zinc-800/50',
  hoverStrong: () => theme() === 'light' ? 'hover:bg-gray-200' : 'hover:bg-zinc-700/50',
  
  // Cards and panels
  card: () => theme() === 'light' ? 'bg-white border-gray-200' : 'bg-zinc-900/50 border-zinc-800/60',
  cardHover: () => theme() === 'light' ? 'hover:bg-gray-50 hover:border-gray-300' : 'hover:bg-zinc-900/70 hover:border-zinc-700/60',
  
  // Input fields
  input: () => theme() === 'light' ? 'bg-white border-gray-300 text-gray-900 placeholder-gray-500' : 'bg-zinc-900/90 border-zinc-800/60 text-white placeholder-zinc-500',
  inputFocus: () => theme() === 'light' ? 'focus:border-blue-500 focus:ring-blue-500/20' : 'focus:border-purple-500 focus:ring-purple-500/20',
}