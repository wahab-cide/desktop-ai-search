import { invoke } from '@tauri-apps/api/core'

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

export const searchAPI = {
  search: (query: string): Promise<SearchResult[]> => 
    invoke('search_documents', { query }),
  
  indexFile: (path: string): Promise<void> => 
    invoke('index_file', { path }),
  
  getIndexingStatus: (): Promise<[number, number]> =>
    invoke('get_indexing_status'),
  
  getFileContent: (path: string): Promise<string> =>
    invoke('get_file_content', { path })
}