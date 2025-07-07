export interface SearchResult {
  id: string
  file_path: string
  content: string
  highlighted_content?: string
  score: number
  match_type?: 'exact' | 'semantic' | 'fuzzy'
  file_type?: string
  file_size?: number
  modified_date?: string
  created_date?: string
  author?: string
  tags?: string[]
}

export interface SearchSuggestion {
  query: string
  type: 'history' | 'suggestion' | 'completion'
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
  queryIntent?: string
  suggestedQuery?: string
}

export interface IndexingStatus {
  indexed: number
  total: number
  current_file?: string
}

export interface FileTypeCount {
  [key: string]: number
}