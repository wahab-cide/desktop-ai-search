import { createSignal, createEffect, batch } from 'solid-js'
import { mockSearchAPI, isDevelopment } from '../mock-api'
import { searchAPI } from '../api'
import type { SearchResult, SearchSuggestion, SearchFilters, SearchMetadata, FileType } from '../types/api'

// Use mock API in development, real API in production
// Set to false to test real backend even in development
const USE_MOCK = false
const api = (isDevelopment && USE_MOCK) ? mockSearchAPI : searchAPI

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
  searchImages: true,
  fileSize: 'any'
})

// Available filter options
export const [availableFileTypes, setAvailableFileTypes] = createSignal<FileType[]>([
  { type: 'pdf', label: 'PDF Documents', count: 0 },
  { type: 'docx', label: 'Word Documents', count: 0 },
  { type: 'doc', label: 'Word Documents (Legacy)', count: 0 },
  { type: 'txt', label: 'Text Files', count: 0 },
  { type: 'md', label: 'Markdown', count: 0 },
  { type: 'jpg', label: 'JPEG Images', count: 0 },
  { type: 'png', label: 'PNG Images', count: 0 },
  { type: 'gif', label: 'GIF Images', count: 0 },
  { type: 'mp3', label: 'MP3 Audio', count: 0 },
  { type: 'mp4', label: 'MP4 Video', count: 0 },
  { type: 'email', label: 'Emails', count: 0 }
])

export const dateRanges = [
  { value: 'all', label: 'All Time' },
  { value: 'today', label: 'Today' },
  { value: 'week', label: 'Last 7 Days' },
  { value: 'month', label: 'Last 30 Days' },
  { value: 'year', label: 'Last Year' }
]

export const fileSizes = [
  { value: 'any', label: 'Any size' },
  { value: 'small', label: 'Small (< 1MB)' },
  { value: 'medium', label: 'Medium (1-10MB)' },
  { value: 'large', label: 'Large (> 10MB)' }
]

// Debounced search function
let searchTimeout: number | null = null
export const performSearch = async (query: string, currentFilters: SearchFilters) => {
  // If we have file type filters but no query, browse by file type
  if (!query.trim() && currentFilters.fileTypes.length > 0) {
    setIsSearching(true)
    try {
      const results = []
      const startTime = Date.now()
      
      // Browse each selected file type
      for (const fileType of currentFilters.fileTypes) {
        const response = await api.browseFilesByType(fileType, 50)
        if (response.results) {
          results.push(...response.results)
        }
      }
      
      const endTime = Date.now()
      
      batch(() => {
        setSearchResults(results)
        setSearchMetadata({
          totalResults: results.length,
          searchTime: (endTime - startTime) / 1000,
          queryIntent: `Browsing ${currentFilters.fileTypes.join(', ')} files`,
          suggestedQuery: undefined
        })
      })
    } catch (error) {
      console.error('Browse failed:', error)
      batch(() => {
        setSearchResults([])
        setSearchMetadata(null)
      })
    } finally {
      setIsSearching(false)
    }
    return
  }
  
  // Clear results if no query and no filters
  if (!query.trim() && currentFilters.fileTypes.length === 0) {
    batch(() => {
      setSearchResults([])
      setSearchMetadata(null)
    })
    return
  }
  
  // Regular search with query
  setIsSearching(true)
  try {
    const startTime = Date.now()
    const results = await api.search(query, currentFilters)
    const endTime = Date.now()
    
    batch(() => {
      setSearchResults(results.results || [])
      setSearchMetadata({
        totalResults: results.total || 0,
        searchTime: (endTime - startTime) / 1000,
        queryIntent: results.query_intent,
        suggestedQuery: results.suggested_query
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
    const suggestions = await api.getSuggestions(query)
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
  
  // Trigger search if we have a query OR file type filters
  if (query.trim() || currentFilters.fileTypes.length > 0) {
    searchTimeout = window.setTimeout(() => {
      performSearch(query, currentFilters)
    }, 300)
  } else {
    // Clear results if no query and no filters
    batch(() => {
      setSearchResults([])
      setSearchMetadata(null)
    })
  }
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
    const counts = await api.getFileTypeCounts()
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