# API Reference

This document describes the Tauri command API available to the frontend application.

## Table of Contents

- [Search Commands](#search-commands)
- [Indexing Commands](#indexing-commands)
- [File Operations](#file-operations)
- [AI Commands](#ai-commands)
- [Embedding Commands](#embedding-commands)
- [Health & Recovery](#health--recovery)
- [Cache Management](#cache-management)
- [Monitoring Commands](#monitoring-commands)

## Search Commands

### `search_documents`
Basic document search with optional filters.

```typescript
interface SearchDocuments {
  query: string;
  filters?: {
    file_type?: string;
    date_range?: {
      start: string;
      end: string;
    };
    size_range?: {
      min: number;
      max: number;
    };
  };
  limit?: number;
  offset?: number;
}

const results = await invoke<SearchResult[]>('search_documents', {
  query: 'machine learning',
  filters: {
    file_type: 'pdf',
    date_range: {
      start: '2024-01-01',
      end: '2024-12-31'
    }
  },
  limit: 50
});
```

### `search_with_filters`
Advanced search with comprehensive filtering options.

```typescript
interface SearchWithFilters {
  query: string;
  file_types?: string[];
  date_from?: string;
  date_to?: string;
  size_min?: number;
  size_max?: number;
  author?: string;
  tags?: string[];
  limit?: number;
  offset?: number;
}

const results = await invoke<SearchResult[]>('search_with_filters', {
  query: 'quarterly report',
  file_types: ['pdf', 'docx'],
  date_from: '2024-01-01',
  author: 'John Smith',
  limit: 25
});
```

### `get_search_suggestions`
Get search suggestions based on query history and content.

```typescript
const suggestions = await invoke<SearchSuggestion[]>('get_search_suggestions', {
  query: 'mach',
  limit: 10
});

interface SearchSuggestion {
  query: string;
  type: 'history' | 'content' | 'ai_generated';
  category?: string;
  count?: number;
}
```

### `rebuild_search_index`
Rebuild the entire search index from scratch.

```typescript
await invoke('rebuild_search_index');
```

## Indexing Commands

### `index_file`
Index a single file.

```typescript
await invoke('index_file', {
  file_path: '/path/to/document.pdf'
});
```

### `index_directory`
Index all files in a directory recursively.

```typescript
await invoke('index_directory', {
  directory_path: '/path/to/documents',
  recursive: true,
  follow_symlinks: false
});
```

### `index_directory_incremental`
Perform incremental indexing on a directory.

```typescript
await invoke('index_directory_incremental', {
  directory_path: '/path/to/documents',
  force_reindex: false
});
```

### `get_indexing_status`
Get current indexing status and progress.

```typescript
const status = await invoke<IndexingStatus>('get_indexing_status');

interface IndexingStatus {
  is_running: boolean;
  current_file?: string;
  files_processed: number;
  total_files: number;
  errors: number;
  start_time?: string;
  estimated_completion?: string;
}
```

### `get_indexing_statistics`
Get detailed indexing statistics.

```typescript
const stats = await invoke<IndexingStatistics>('get_indexing_statistics');

interface IndexingStatistics {
  total_documents: number;
  total_size: number;
  file_type_breakdown: Record<string, number>;
  indexing_errors: number;
  last_index_time: string;
  average_processing_time: number;
}
```

## File Operations

### `get_file_content`
Get the content of a file.

```typescript
const content = await invoke<string>('get_file_content', {
  file_path: '/path/to/document.txt'
});
```

### `open_file_in_default_app`
Open a file in the default system application.

```typescript
await invoke('open_file_in_default_app', {
  file_path: '/path/to/document.pdf'
});
```

### `show_file_in_folder`
Show a file in the system file manager.

```typescript
await invoke('show_file_in_folder', {
  file_path: '/path/to/document.pdf'
});
```

## AI Commands

### `init_ai_system`
Initialize the AI system with specified configuration.

```typescript
await invoke('init_ai_system', {
  model_path: '/path/to/model',
  device: 'auto', // 'cpu', 'cuda', 'auto'
  max_tokens: 2048
});
```

### `list_ai_models`
List available AI models.

```typescript
const models = await invoke<AIModel[]>('list_ai_models');

interface AIModel {
  name: string;
  type: 'embedding' | 'llm' | 'vision';
  size: number;
  description: string;
  is_downloaded: boolean;
  is_loaded: boolean;
}
```

### `load_ai_model`
Load a specific AI model.

```typescript
await invoke('load_ai_model', {
  model_name: 'sentence-transformers/all-MiniLM-L6-v2',
  device: 'auto'
});
```

### `process_ai_query`
Process a query using the AI system.

```typescript
const response = await invoke<AIResponse>('process_ai_query', {
  query: 'Summarize the key points from the quarterly report',
  context?: string,
  max_tokens?: number
});

interface AIResponse {
  response: string;
  confidence: number;
  processing_time: number;
  model_used: string;
}
```

## Embedding Commands

### `generate_embeddings`
Generate embeddings for a text or file.

```typescript
const embeddings = await invoke<number[]>('generate_embeddings', {
  text: 'This is a sample text for embedding generation',
  model_name?: 'all-MiniLM-L6-v2'
});
```

### `calculate_text_similarity`
Calculate similarity between two texts.

```typescript
const similarity = await invoke<number>('calculate_text_similarity', {
  text1: 'First text',
  text2: 'Second text',
  model_name?: 'all-MiniLM-L6-v2'
});
```

### `get_embedding_status`
Get the status of the embedding system.

```typescript
const status = await invoke<EmbeddingStatus>('get_embedding_status');

interface EmbeddingStatus {
  is_initialized: boolean;
  model_loaded: string;
  embedding_count: number;
  last_update: string;
  memory_usage: number;
}
```

## Health & Recovery

### `get_health_status`
Get overall system health status.

```typescript
const health = await invoke<HealthStatus>('get_health_status');

interface HealthStatus {
  overall_status: 'healthy' | 'warning' | 'error';
  database_status: 'connected' | 'disconnected' | 'error';
  ai_system_status: 'loaded' | 'loading' | 'error';
  cache_status: 'active' | 'inactive' | 'error';
  last_check: string;
  uptime: number;
}
```

### `get_system_metrics`
Get detailed system metrics.

```typescript
const metrics = await invoke<SystemMetrics>('get_system_metrics');

interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_activity: number;
  process_count: number;
  thread_count: number;
}
```

### `trigger_recovery`
Trigger system recovery procedures.

```typescript
await invoke('trigger_recovery', {
  component: 'database' | 'cache' | 'ai_system' | 'all',
  force: false
});
```

## Cache Management

### `get_cache_status`
Get current cache status and statistics.

```typescript
const status = await invoke<CacheStatus>('get_cache_status');

interface CacheStatus {
  total_size: number;
  entry_count: number;
  hit_rate: number;
  miss_rate: number;
  eviction_count: number;
  memory_usage: number;
}
```

### `clear_all_caches`
Clear all caches.

```typescript
await invoke('clear_all_caches');
```

### `clear_cache_type`
Clear specific cache type.

```typescript
await invoke('clear_cache_type', {
  cache_type: 'search' | 'embedding' | 'model' | 'query'
});
```

### `optimize_caches`
Optimize cache performance.

```typescript
await invoke('optimize_caches');
```

## Monitoring Commands

### `get_system_health`
Get comprehensive system health information.

```typescript
const health = await invoke<SystemHealth>('get_system_health');

interface SystemHealth {
  overall_status: 'healthy' | 'warning' | 'critical';
  components: Record<string, ComponentHealth>;
  system_metrics: SystemMetrics;
  performance_metrics: PerformanceMetrics;
  alerts: Alert[];
  uptime: number;
  last_updated: string;
}
```

### `get_performance_metrics`
Get current performance metrics.

```typescript
const metrics = await invoke<PerformanceMetrics>('get_performance_metrics');

interface PerformanceMetrics {
  search_performance: {
    avg_response_time_ms: number;
    queries_per_second: number;
    success_rate: number;
  };
  indexing_performance: {
    files_per_second: number;
    avg_processing_time_ms: number;
    success_rate: number;
  };
  cache_performance: {
    hit_rate: number;
    memory_usage_mb: number;
  };
  model_performance: {
    inference_time_ms: number;
    memory_usage_mb: number;
  };
}
```

### `record_metric`
Record a custom metric.

```typescript
await invoke('record_metric', {
  name: 'custom_metric',
  value: 42.0,
  tags: {
    component: 'search',
    operation: 'query'
  }
});
```

### `get_active_alerts`
Get currently active alerts.

```typescript
const alerts = await invoke<Alert[]>('get_active_alerts');

interface Alert {
  id: string;
  severity: 'info' | 'warning' | 'critical' | 'emergency';
  title: string;
  description: string;
  component: string;
  created_at: string;
  acknowledged: boolean;
  resolved: boolean;
}
```

## Error Handling

All commands may throw errors. Use try-catch blocks for proper error handling:

```typescript
try {
  const results = await invoke<SearchResult[]>('search_documents', {
    query: 'example query'
  });
  // Handle successful response
} catch (error) {
  console.error('Search failed:', error);
  // Handle error appropriately
}
```

## Common Error Types

- `DatabaseError`: Database connection or query issues
- `IndexingError`: File indexing problems
- `AIError`: AI model loading or inference errors
- `FileSystemError`: File access or permission issues
- `ConfigurationError`: Configuration validation errors
- `NetworkError`: Network-related issues
- `ValidationError`: Input validation failures

## Rate Limiting

Some commands may be rate-limited to prevent system overload:

- Search commands: 100 requests per minute
- Indexing commands: 10 requests per minute
- AI commands: 50 requests per minute

## Best Practices

1. **Error Handling**: Always wrap commands in try-catch blocks
2. **Debouncing**: Debounce search queries to avoid excessive API calls
3. **Pagination**: Use offset and limit parameters for large result sets
4. **Caching**: Cache frequently accessed data on the frontend
5. **Progress Tracking**: Use status commands for long-running operations
6. **Resource Management**: Monitor system resources during intensive operations

## TypeScript Types

All TypeScript interfaces are available in the `types/api.ts` file:

```typescript
import type {
  SearchResult,
  SearchSuggestion,
  IndexingStatus,
  SystemHealth,
  PerformanceMetrics
} from '../types/api';
```