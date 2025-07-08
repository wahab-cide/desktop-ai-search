import type { 
  SearchResponse, 
  SearchFilters, 
  SearchSuggestion, 
  FileTypeCount, 
  IndexingStatus,
  IndexingStatistics,
  SearchResult
} from './types/api'

// Mock data for testing
const mockSearchResults: SearchResult[] = [
  {
    id: '1',
    file_path: '/Users/demo/Documents/Machine Learning Fundamentals.pdf',
    content: 'Machine learning is a subset of artificial intelligence that focuses on algorithms which can learn from and make predictions on data...',
    highlighted_content: '<mark>Machine learning</mark> is a subset of artificial intelligence that focuses on algorithms which can learn from and make predictions on data...',
    score: 0.95,
    match_type: 'hybrid',
    file_type: 'pdf',
    file_size: 2048576,
    modified_date: '2024-01-15T10:30:00Z',
    created_date: '2024-01-10T14:20:00Z'
  },
  {
    id: '2', 
    file_path: '/Users/demo/Projects/ai-research/neural-networks.md',
    content: 'Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes or neurons...',
    highlighted_content: '<mark>Neural networks</mark> are computing systems inspired by biological neural networks. They consist of interconnected nodes or neurons...',
    score: 0.87,
    match_type: 'semantic',
    file_type: 'md',
    file_size: 45632,
    modified_date: '2024-01-20T16:45:00Z'
  },
  {
    id: '3',
    file_path: '/Users/demo/Screenshots/Screenshot 2024-01-21 at 3.45.12 PM.png',
    content: 'Image contains text about deep learning algorithms and convolutional neural networks...',
    highlighted_content: 'Image contains text about <mark>deep learning</mark> algorithms and convolutional neural networks...',
    score: 0.72,
    match_type: 'exact',
    file_type: 'png',
    file_size: 1234567,
    modified_date: '2024-01-21T15:45:00Z'
  },
  {
    id: '4',
    file_path: '/Users/demo/Notes/lecture-notes-2024.txt',
    content: 'Today we discussed supervised learning techniques including linear regression, decision trees, and support vector machines...',
    highlighted_content: 'Today we discussed <mark>supervised learning</mark> techniques including linear regression, decision trees, and support vector machines...',
    score: 0.68,
    match_type: 'hybrid',
    file_type: 'txt',
    file_size: 8192,
    modified_date: '2024-01-18T09:15:00Z'
  }
]

const mockSuggestions: SearchSuggestion[] = [
  { query: 'machine learning algorithms', type: 'history', category: 'AI/ML', count: 15 },
  { query: 'neural network architecture', type: 'suggestion', category: 'Deep Learning', count: 8 },
  { query: 'supervised learning examples', type: 'completion', category: 'ML Basics', count: 23 },
  { query: 'tensorflow tutorial', type: 'popular', category: 'Frameworks', count: 45 },
  { query: 'machine learning papers 2024', type: 'recent', category: 'Research', count: 12 }
]

const mockFileTypes: FileTypeCount = {
  pdf: 145,
  md: 67,
  txt: 234,
  docx: 89,
  png: 156,
  jpg: 203,
  mp4: 34,
  mp3: 78
}

const mockIndexingStatus: IndexingStatus = {
  indexed: 1247,
  total: 1850,
  current_file: '/Users/demo/Documents/Research Papers/transformer-attention.pdf'
}

const mockIndexingStats: IndexingStatistics = {
  total_files: 1850,
  indexed_files: 1247,
  failed_files: 15,
  total_size_bytes: 5439872000, // ~5GB
  indexed_size_bytes: 3521456128, // ~3.3GB
  start_time: '2024-01-22T10:00:00Z',
  last_update: new Date().toISOString(),
  files_per_second: 2.3,
  bytes_per_second: 1572864 // ~1.5MB/s
}

// Mock API implementation
export const mockSearchAPI = {
  async search(query: string, filters: SearchFilters): Promise<SearchResponse> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 300))
    
    let results = mockSearchResults
    
    // Apply filters
    if (filters.fileTypes.length > 0) {
      results = results.filter(r => filters.fileTypes.includes(r.file_type || ''))
    }
    
    // Simulate query matching
    if (query.trim()) {
      results = results.filter(r => 
        r.content.toLowerCase().includes(query.toLowerCase()) ||
        r.file_path.toLowerCase().includes(query.toLowerCase())
      )
    }
    
    return {
      results,
      total: results.length,
      query_intent: query.includes('how') ? 'Learn about machine learning concepts' : undefined,
      suggested_query: query.length < 3 ? 'machine learning fundamentals' : undefined
    }
  },

  async getSuggestions(query: string): Promise<SearchSuggestion[]> {
    await new Promise(resolve => setTimeout(resolve, 150))
    
    if (!query.trim()) return []
    
    return mockSuggestions.filter(s => 
      s.query.toLowerCase().includes(query.toLowerCase())
    ).slice(0, 5)
  },

  async getFileTypeCounts(): Promise<FileTypeCount> {
    await new Promise(resolve => setTimeout(resolve, 100))
    return mockFileTypes
  },

  async getIndexingStatus(): Promise<IndexingStatus | null> {
    await new Promise(resolve => setTimeout(resolve, 50))
    return mockIndexingStatus
  },

  async indexDirectory(path: string, incremental: boolean = false): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 500))
    console.log(`Mock: Started ${incremental ? 'incremental' : 'full'} indexing of ${path}`)
  },

  async indexFile(path: string): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 200))
    console.log(`Mock: Indexed file ${path}`)
  },

  async startBackgroundIndexing(): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 300))
    console.log('Mock: Started background indexing')
  },

  async getIndexingStatistics(): Promise<IndexingStatistics | null> {
    await new Promise(resolve => setTimeout(resolve, 100))
    return mockIndexingStats
  },

  async resetIndexingState(): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 400))
    console.log('Mock: Reset indexing state')
  },

  async getFileContent(path: string): Promise<string> {
    await new Promise(resolve => setTimeout(resolve, 200))
    return `Mock content for file: ${path}\n\nThis is sample content that would be extracted from the file.`
  }
}

// Check if we're in development mode
export const isDevelopment = import.meta.env.DEV

// Use mock API in development, real API in production
export const getCurrentAPI = () => {
  if (isDevelopment) {
    console.log('🚀 Using Mock API for development testing')
    return mockSearchAPI
  } else {
    console.log('🔗 Using Real Tauri API')
    // Import the real API when needed
    return import('./api').then(module => module.searchAPI)
  }
}