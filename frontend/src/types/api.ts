export interface SearchResult {
    id: string
    content: string
    file_path: string
    score: number
  }
  
  export interface IndexingStatus {
    indexed: number
    total: number
    current_file?: string
  }