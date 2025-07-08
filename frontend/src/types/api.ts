export interface SearchResult {
  id: string
  file_path: string
  content: string
  highlighted_content?: string
  score: number
  match_type?: 'exact' | 'semantic' | 'hybrid' | 'fuzzy'
  file_type?: string
  file_size?: number
  modified_date?: string
  created_date?: string
  author?: string
  tags?: string[]
}

export interface SearchSuggestion {
  query: string
  type: 'history' | 'suggestion' | 'completion' | 'recent' | 'popular'
  category?: string
  count?: number
  confidence?: number
}

export interface SearchFilters {
  fileTypes: string[]
  dateRange: string
  searchMode: string
  includeContent: boolean
  includeOCR: boolean
  searchImages: boolean
  fileSize?: string
}

// Backend format for search filters
export interface BackendSearchFilters {
  file_types: string[]
  date_range: string
  search_mode: string
  include_content: boolean
  include_ocr: boolean
  search_images: boolean
}

export interface SearchMetadata {
  totalResults: number
  searchTime: number
  queryIntent?: string
  suggestedQuery?: string
}

export interface SearchResponse {
  results: SearchResult[]
  total: number
  query_intent?: string
  suggested_query?: string
}

export interface IndexingStatus {
  indexed: number
  total: number
  current_file?: string
}

export interface FileTypeCount {
  [key: string]: number
}

// AI Query Types
export interface AiQueryRequest {
  query: string
  search_context?: string
  preset?: 'creative' | 'precise' | 'balanced'
  max_tokens?: number
}

export interface AiQueryResponse {
  response: string
  tokens_generated: number
  tokens_per_second: number
  time_ms: number
  model_info?: string
}

// Indexing Types
export interface IndexingStatistics {
  total_files: number
  indexed_files: number
  failed_files: number
  total_size_bytes: number
  indexed_size_bytes: number
  start_time?: string
  last_update?: string
  files_per_second?: number
  bytes_per_second?: number
}

// File Type with label and count for UI
export interface FileType {
  type: string
  label: string
  count: number
}