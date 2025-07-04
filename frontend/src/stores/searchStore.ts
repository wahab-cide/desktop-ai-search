import { createSignal, createEffect } from 'solid-js'
import { searchAPI } from '../api'
import type { SearchResult } from '../types/api'

export const [searchQuery, setSearchQuery] = createSignal('')
export const [searchResults, setSearchResults] = createSignal<SearchResult[]>([])
export const [isSearching, setIsSearching] = createSignal(false)

export const performSearch = async (query: string) => {
  if (!query.trim()) {
    setSearchResults([])
    return
  }
  
  setIsSearching(true)
  try {
    const results = await searchAPI.search(query)
    setSearchResults(results)
  } catch (error) {
    console.error('Search failed:', error)
    setSearchResults([])
  } finally {
    setIsSearching(false)
  }
}

// Auto-search with debouncing
createEffect(() => {
  const query = searchQuery()
  const timeoutId = setTimeout(() => {
    performSearch(query)
  }, 300)
  
  return () => clearTimeout(timeoutId)
})