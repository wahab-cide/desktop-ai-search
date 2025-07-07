import { invoke } from '@tauri-apps/api/core'
import type { SearchResponse, SearchFilters, SearchSuggestion, FileTypeCount, IndexingStatus } from './types/api'

export const searchAPI = {
  async search(query: string, filters: SearchFilters): Promise<SearchResponse> {
    try {
      // Call the Tauri backend search command
      const results = await invoke<SearchResponse>('search', {
        query,
        filters
      })
      return results
    } catch (error) {
      console.error('Search error:', error)
      return { results: [], total: 0 }
    }
  },

  async getSuggestions(query: string): Promise<SearchSuggestion[]> {
    try {
      // Call the Tauri backend for search suggestions
      const suggestions = await invoke<SearchSuggestion[]>('get_search_suggestions', {
        query,
        limit: 10
      })
      return suggestions
    } catch (error) {
      console.error('Suggestions error:', error)
      return []
    }
  },

  async getFileTypeCounts(): Promise<FileTypeCount> {
    try {
      // Get file type statistics from the backend
      const counts = await invoke<FileTypeCount>('get_file_type_counts')
      return counts
    } catch (error) {
      console.error('File type counts error:', error)
      return {}
    }
  },

  async getIndexingStatus(): Promise<IndexingStatus | null> {
    try {
      const status = await invoke<IndexingStatus>('get_indexing_status')
      return status
    } catch (error) {
      console.error('Indexing status error:', error)
      return null
    }
  },

  async startIndexing(path: string): Promise<void> {
    try {
      await invoke('start_indexing', { path })
    } catch (error) {
      console.error('Start indexing error:', error)
      throw error
    }
  },

  async stopIndexing(): Promise<void> {
    try {
      await invoke('stop_indexing')
    } catch (error) {
      console.error('Stop indexing error:', error)
      throw error
    }
  },

  async getFileContent(path: string): Promise<string> {
    try {
      const content = await invoke<string>('get_file_content', { path })
      return content
    } catch (error) {
      console.error('Get file content error:', error)
      return ''
    }
  }
}