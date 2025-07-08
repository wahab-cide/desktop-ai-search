import { invoke } from '@tauri-apps/api/core'
import type { 
  SearchResponse, 
  SearchFilters, 
  BackendSearchFilters,
  SearchSuggestion, 
  FileTypeCount, 
  IndexingStatus,
  IndexingStatistics,
  AiQueryRequest,
  AiQueryResponse,
  FileType
} from './types/api'

// Add missing SearchResult type
interface SearchResult {
  id: string
  title: string
  content: string
  file_path: string
  file_type: string
  relevance_score: number
  created_at: string
  modified_at: string
}

// Helper function to convert frontend filters to backend format
function transformFilters(filters: SearchFilters): BackendSearchFilters {
  return {
    file_types: filters.fileTypes,
    date_range: filters.dateRange,
    search_mode: filters.searchMode,
    include_content: filters.includeContent,
    include_ocr: filters.includeOCR,
    search_images: filters.searchImages
  }
}

export const searchAPI = {
  // FTS rebuild command
  async rebuildSearchIndex(): Promise<string> {
    try {
      const result = await invoke<string>('rebuild_search_index')
      return result
    } catch (error) {
      console.error('Rebuild search index error:', error)
      throw error
    }
  },

  async search(query: string, filters: SearchFilters): Promise<SearchResponse> {
    try {
      // Use the basic search command first (more stable)
      const results = await invoke<any[]>('search_documents', {
        query
      })
      return {
        results: results || [],
        total: results?.length || 0,
        query_intent: undefined,
        suggested_query: undefined
      }
    } catch (error) {
      console.error('Search error:', error)
      return { results: [], total: 0 }
    }
  },

  async getSuggestions(query: string): Promise<SearchSuggestion[]> {
    try {
      // Use basic suggestions (more stable)
      const suggestions = await invoke<string[]>('get_search_suggestions', {
        partial_query: query
      })
      return suggestions.map(suggestion => ({
        query: suggestion,
        type: 'suggestion' as const,
        category: undefined,
        count: undefined,
        confidence: undefined
      }))
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

  async indexDirectory(path: string, incremental: boolean = false): Promise<void> {
    try {
      const command = incremental ? 'index_directory_incremental' : 'index_directory'
      await invoke(command, { directoryPath: path })
    } catch (error) {
      console.error('Index directory error:', error)
      throw error
    }
  },

  async indexFile(path: string): Promise<void> {
    try {
      await invoke('index_file', { path })
    } catch (error) {
      console.error('Index file error:', error)
      throw error
    }
  },

  async startBackgroundIndexing(): Promise<void> {
    try {
      await invoke('start_background_indexing')
    } catch (error) {
      console.error('Start background indexing error:', error)
      throw error
    }
  },

  async getIndexingStatistics(): Promise<IndexingStatistics | null> {
    try {
      const stats = await invoke<IndexingStatistics>('get_indexing_statistics')
      return stats
    } catch (error) {
      console.error('Indexing statistics error:', error)
      return null
    }
  },

  async resetIndexingState(): Promise<void> {
    try {
      await invoke('reset_indexing_state')
    } catch (error) {
      console.error('Reset indexing state error:', error)
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
  },

  async openFileInDefaultApp(filePath: string): Promise<string> {
    try {
      const result = await invoke<string>('open_file_in_default_app', { filePath })
      return result
    } catch (error) {
      console.error('Open file error:', error)
      throw error
    }
  },

  async showFileInFolder(filePath: string): Promise<string> {
    try {
      const result = await invoke<string>('show_file_in_folder', { filePath })
      return result
    } catch (error) {
      console.error('Show file in folder error:', error)
      throw error
    }
  }
}

// AI API functions
export const aiAPI = {
  async processQuery(request: AiQueryRequest): Promise<AiQueryResponse> {
    try {
      const response = await invoke<AiQueryResponse>('process_ai_query', request)
      return response
    } catch (error) {
      console.error('AI query error:', error)
      throw error
    }
  },

  async generateQuerySuggestions(query: string): Promise<string[]> {
    try {
      const suggestions = await invoke<string[]>('generate_query_suggestions', { query })
      return suggestions
    } catch (error) {
      console.error('AI suggestions error:', error)
      return []
    }
  },

  async analyzeSearchResults(results: any[], query: string): Promise<string> {
    try {
      const analysis = await invoke<string>('analyze_search_results', { results, query })
      return analysis
    } catch (error) {
      console.error('AI analysis error:', error)
      return ''
    }
  },

  async initAiSystem(): Promise<void> {
    try {
      await invoke('init_ai_system')
    } catch (error) {
      console.error('AI init error:', error)
      throw error
    }
  },

  async getAiSystemInfo(): Promise<any> {
    try {
      const info = await invoke('get_ai_system_info')
      return info
    } catch (error) {
      console.error('AI system info error:', error)
      return null
    }
  }
}