import { createSignal, createEffect, batch } from 'solid-js'
import { searchAPI } from '../api'
import type { SearchResult, SearchSuggestion, SearchFilters, SearchMetadata } from '../types/api'

// Core search state
export const [searchQuery, setSearchQuery] = createSignal('')
export const [searchResults, setSearchResults] = createSignal<SearchResult[]>([])
export const [isSearching, setIsSearching] = createSignal(false)
export const [searchMetadata, setSearchMetadata] = createSignal<SearchMetadata | null>(null)

// Suggestions state
export const [searchSuggestions, setSearchSuggestions] = createSignal<SearchSuggestion[]>([])
export const [isLoadingSuggestions, setIsLoadingSuggestions] = createSignal(false)

// Filter state
export const [filters, setFilters] = createSignal<SearchFilters>({
  fileTypes: [],
  dateRange: 'all',
  searchMode: 'hybrid',
  includeContent: true,
  includeOCR: true,
  searchImages: true
})

// Available filter options
export const [availableFileTypes, setAvailableFileTypes] = createSignal([
  { type: 'pdf', label: 'PDF Documents', count: 0 },
  { type: 'docx', label: 'Word Documents', count: 0 },
  { type: 'txt', label: 'Text Files', count: 0 },
  { type: 'md', label: 'Markdown', count: 0 },
  { type: 'image', label: 'Images', count: 0 },
  { type: 'audio', label: 'Audio Files', count: 0 },
  { type: 'email', label: 'Emails', count: 0 }
])

export const dateRanges = [
  { value: 'all', label: 'All Time' },
  { value: 'today', label: 'Today' },
  { value: 'week', label: 'Last 7 Days' },
  { value: 'month', label: 'Last 30 Days' },
  { value: 'year', label: 'Last Year' }
]

// Debounced search function
let searchTimeout: number | null = null
export const performSearch = async (query: string, currentFilters: SearchFilters) => {
  if (!query.trim() && currentFilters.fileTypes.length === 0) {
    batch(() => {
      setSearchResults([])
      setSearchMetadata(null)
    })
    return
  }
  
  setIsSearching(true)
  try {
    const startTime = Date.now()
    const results = await searchAPI.search(query, currentFilters)
    const endTime = Date.now()
    
    batch(() => {
      setSearchResults(results.results || [])
      setSearchMetadata({
        totalResults: results.total || 0,
        searchTime: (endTime - startTime) / 1000,
        queryIntent: results.queryIntent,
        suggestedQuery: results.suggestedQuery
      })
    })
  } catch (error) {
    console.error('Search failed:', error)
    batch(() => {
      setSearchResults([])
      setSearchMetadata(null)
    })
  } finally {
    setIsSearching(false)
  }
}

// Debounced suggestions function
let suggestionsTimeout: number | null = null
export const loadSuggestions = async (query: string) => {
  if (!query.trim()) {
    setSearchSuggestions([])
    return
  }
  
  setIsLoadingSuggestions(true)
  try {
    const suggestions = await searchAPI.getSuggestions(query)
    setSearchSuggestions(suggestions)
  } catch (error) {
    console.error('Failed to load suggestions:', error)
    setSearchSuggestions([])
  } finally {
    setIsLoadingSuggestions(false)
  }
}

// Auto-search with debouncing
createEffect(() => {
  const query = searchQuery()
  const currentFilters = filters()
  
  if (searchTimeout) clearTimeout(searchTimeout)
  searchTimeout = window.setTimeout(() => {
    performSearch(query, currentFilters)
  }, 300)
})

// Auto-suggestions with debouncing
createEffect(() => {
  const query = searchQuery()
  
  if (suggestionsTimeout) clearTimeout(suggestionsTimeout)
  suggestionsTimeout = window.setTimeout(() => {
    loadSuggestions(query)
  }, 150)
})

// Load initial file type counts
export const loadFileTypeCounts = async () => {
  try {
    const counts = await searchAPI.getFileTypeCounts()
    setAvailableFileTypes(prevTypes => 
      prevTypes.map(type => ({
        ...type,
        count: counts[type.type] || 0
      }))
    )
  } catch (error) {
    console.error('Failed to load file type counts:', error)
  }
}

// Initialize
loadFileTypeCounts()