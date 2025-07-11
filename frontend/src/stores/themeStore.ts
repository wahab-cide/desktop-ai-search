import { createSignal, createEffect } from 'solid-js'

export type Theme = 'light' | 'dark' | 'system'

// Get initial theme from localStorage or system preference
const getInitialTheme = (): Theme => {
  const stored = localStorage.getItem('desktop-ai-search-theme') as Theme
  if (stored && ['light', 'dark', 'system'].includes(stored)) {
    return stored
  }
  return 'system'
}

// Get effective theme (resolve 'system' to actual theme)
const getEffectiveTheme = (theme: Theme): 'light' | 'dark' => {
  if (theme === 'system') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }
  return theme
}

const [theme, setTheme] = createSignal<Theme>(getInitialTheme())
const [effectiveTheme, setEffectiveTheme] = createSignal<'light' | 'dark'>(getEffectiveTheme(getInitialTheme()))

// Listen for system theme changes
const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
mediaQuery.addEventListener('change', () => {
  if (theme() === 'system') {
    setEffectiveTheme(getEffectiveTheme('system'))
  }
})

// Update effective theme and persist to localStorage when theme changes
createEffect(() => {
  const currentTheme = theme()
  setEffectiveTheme(getEffectiveTheme(currentTheme))
  localStorage.setItem('desktop-ai-search-theme', currentTheme)
})

export { theme, setTheme, effectiveTheme }

export const toggleTheme = () => {
  const currentTheme = theme()
  if (currentTheme === 'light') {
    setTheme('dark')
  } else if (currentTheme === 'dark') {
    setTheme('system')
  } else {
    setTheme('light')
  }
}

// Theme-aware class helpers
export const themeClasses = {
  // Main backgrounds
  bg: () => effectiveTheme() === 'light' ? 'bg-white' : 'bg-black',
  bgSecondary: () => effectiveTheme() === 'light' ? 'bg-gray-50' : 'bg-zinc-900/80',
  bgTertiary: () => effectiveTheme() === 'light' ? 'bg-gray-100' : 'bg-zinc-900/60',
  
  // Text colors
  text: () => effectiveTheme() === 'light' ? 'text-gray-900' : 'text-white',
  textSecondary: () => effectiveTheme() === 'light' ? 'text-gray-600' : 'text-zinc-300',
  textMuted: () => effectiveTheme() === 'light' ? 'text-gray-500' : 'text-zinc-400',
  textSubtle: () => effectiveTheme() === 'light' ? 'text-gray-400' : 'text-zinc-500',
  
  // Borders
  border: () => effectiveTheme() === 'light' ? 'border-gray-200' : 'border-zinc-800/60',
  borderSecondary: () => effectiveTheme() === 'light' ? 'border-gray-300' : 'border-zinc-700/50',
  
  // Interactive elements
  hover: () => effectiveTheme() === 'light' ? 'hover:bg-gray-100' : 'hover:bg-zinc-800/50',
  hoverStrong: () => effectiveTheme() === 'light' ? 'hover:bg-gray-200' : 'hover:bg-zinc-700/50',
  
  // Cards and panels
  card: () => effectiveTheme() === 'light' ? 'bg-white border-gray-200' : 'bg-zinc-900/50 border-zinc-800/60',
  cardHover: () => effectiveTheme() === 'light' ? 'hover:bg-gray-50 hover:border-gray-300' : 'hover:bg-zinc-900/70 hover:border-zinc-700/60',
  
  // Input fields
  input: () => effectiveTheme() === 'light' ? 'bg-white border-gray-300 text-gray-900 placeholder-gray-500' : 'bg-zinc-900/90 border-zinc-800/60 text-white placeholder-zinc-500',
  inputFocus: () => effectiveTheme() === 'light' ? 'focus:border-blue-500 focus:ring-blue-500/20' : 'focus:border-purple-500 focus:ring-purple-500/20',
}