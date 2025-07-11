import { createSignal, createEffect, onCleanup } from 'solid-js'

export interface KeyboardShortcut {
  key: string
  ctrlKey?: boolean
  metaKey?: boolean
  shiftKey?: boolean
  altKey?: boolean
  action: () => void
  description: string
  category: string
}

const [shortcuts, setShortcuts] = createSignal<KeyboardShortcut[]>([])
const [showShortcutsModal, setShowShortcutsModal] = createSignal(false)

// Global keyboard event handler
const handleKeyboardEvent = (event: KeyboardEvent) => {
  const currentShortcuts = shortcuts()
  
  for (const shortcut of currentShortcuts) {
    const matchKey = shortcut.key.toLowerCase() === event.key.toLowerCase()
    const matchCtrl = (shortcut.ctrlKey || false) === event.ctrlKey
    const matchMeta = (shortcut.metaKey || false) === event.metaKey
    const matchShift = (shortcut.shiftKey || false) === event.shiftKey
    const matchAlt = (shortcut.altKey || false) === event.altKey
    
    if (matchKey && matchCtrl && matchMeta && matchShift && matchAlt) {
      // Don't prevent default for certain system shortcuts
      const isSystemShortcut = (event.ctrlKey || event.metaKey) && 
        ['r', 'f', 'a', 'c', 'v', 'x', 'z', 'y'].includes(event.key.toLowerCase())
      
      if (!isSystemShortcut) {
        event.preventDefault()
      }
      
      shortcut.action()
      break
    }
  }
  
  // Show shortcuts modal with Ctrl/Cmd + ?
  if ((event.ctrlKey || event.metaKey) && event.key === '?') {
    event.preventDefault()
    setShowShortcutsModal(true)
  }
  
  // Close shortcuts modal with Escape
  if (event.key === 'Escape' && showShortcutsModal()) {
    setShowShortcutsModal(false)
  }
}

// Initialize keyboard shortcuts system
createEffect(() => {
  document.addEventListener('keydown', handleKeyboardEvent)
  
  onCleanup(() => {
    document.removeEventListener('keydown', handleKeyboardEvent)
  })
})

export const registerShortcut = (shortcut: KeyboardShortcut) => {
  setShortcuts(prev => [...prev, shortcut])
}

export const unregisterShortcut = (shortcut: KeyboardShortcut) => {
  setShortcuts(prev => prev.filter(s => s !== shortcut))
}

export const clearShortcuts = () => {
  setShortcuts([])
}

export const getShortcutsByCategory = () => {
  const allShortcuts = shortcuts()
  const categories: Record<string, KeyboardShortcut[]> = {}
  
  allShortcuts.forEach(shortcut => {
    if (!categories[shortcut.category]) {
      categories[shortcut.category] = []
    }
    categories[shortcut.category].push(shortcut)
  })
  
  return categories
}

export const formatShortcutKey = (shortcut: KeyboardShortcut) => {
  const parts: string[] = []
  
  if (shortcut.ctrlKey) parts.push('Ctrl')
  if (shortcut.metaKey) parts.push('⌘')
  if (shortcut.altKey) parts.push('Alt')
  if (shortcut.shiftKey) parts.push('⇧')
  
  parts.push(shortcut.key.toUpperCase())
  
  return parts.join(' + ')
}

export { 
  shortcuts, 
  showShortcutsModal, 
  setShowShortcutsModal 
}