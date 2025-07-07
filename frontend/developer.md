# Desktop AI Search - Frontend Developer Guide

## Project Overview

This is the frontend development guide for a **privacy-first, on-device AI search engine** built with Tauri, Solid.js, and TypeScript. The application serves as a personal digital assistant for finding anything across a user's entire digital life while running entirely locally with zero cloud dependencies.

## Tech Stack

### Frontend Architecture
- **Framework**: Tauri (Rust backend + Solid.js frontend)
- **UI Library**: Solid.js with fine-grained reactivity
- **Styling**: Tailwind CSS v4
- **Build Tool**: Vite for frontend bundling
- **Language**: TypeScript for type safety

### Frontend Dependencies
```json
{
  "dependencies": {
    "solid-js": "^1.9.5",
    "@solidjs/router": "^0.32.0",
    "@tauri-apps/api": "^2.6.0",
    "@tauri-apps/plugin-fs": "^2.4.0",
    "@solid-primitives/clipboard": "^1.5.0",
    "@thisbeyond/solid-datepicker": "^0.3.0"
  },
  "devDependencies": {
    "vite": "^6.0.0",
    "vite-plugin-solid": "^2.11.6",
    "tailwindcss": "^4.0.7",
    "@tailwindcss/vite": "^4.0.7",
    "typescript": "^5.7.2"
  }
}
```

### Key Libraries
- **uPlot**: For performance charts and analytics
- **vis-timeline**: For timeline visualization
- **fuse.js**: For client-side fuzzy search
- **Prism.js**: For syntax highlighting
- **leaflet-tiny**: For offline maps

## Development Timeline

### Phase 4: Frontend Development (Weeks 10-12)

#### Week 10: Core UI Components

**Day 1-2: Main Search Interface**
- **Reactive Search Bar**: Implement with streaming suggestions using dual debouncing (requestAnimationFrame + 120ms delay) to prevent >8 suggestion rounds/sec
- **Query History Dropdown**: Use virtualized lists (Solid Virt) for >200 rows to prevent WebKit list diff main thread spikes
- **Faceted Filter Panel**: Create with `createStore({ filters: ... })` for structural diff optimization
- **Windowed Infinite Scroll**: Maintain ±50 results in DOM with intersection observers, flush off-screen nodes to prevent 300MB+ heap snapshots in Tauri WebView
- **Keyboard Navigation**: Add ⌘K for search focus, ⌘/⌥ ↑/↓ for suggestion cycling
- **Visual Polish**: Use Tailwind `sticky + backdrop-blur` for native-feeling search bar

```typescript
// Example Search Interface Structure
export const SearchInterface: Component = () => {
  const [searchQuery, setSearchQuery] = createSignal('')
  const [searchResults, setSearchResults] = createSignal<SearchResult[]>([])
  const [isSearching, setIsSearching] = createSignal(false)
  
  // Debounced search with RAF + 120ms delay
  const debouncedSearch = debounce((query: string) => {
    requestAnimationFrame(() => performSearch(query))
  }, 120)
  
  return (
    <div class="h-full flex flex-col">
      {/* Search bar with streaming suggestions */}
      {/* Virtualized results list */}
      {/* Faceted filters panel */}
    </div>
  )
}
```

**Day 3-4: Rich Result Previews**
- **File Type Icons**: Use SVG spritesheet (50+ icons) to reduce 60KB vs icon fonts
- **Thumbnail System**: Display pre-generated WebP thumbnails (320px width) cached in IndexedDB
- **Syntax Highlighting**: Use pre-tokenized HTML snippets from Rust backend with Prism.js for class application only
- **Rich Text Preview**: Display HTML fragments in sandboxed `<iframe sandbox>` to prevent CSS bleed
- **Audio/Video Preview**: Implement with `controlsList="nodownload"` and lazy waveform rendering

**Day 5-7: Interactive Features**
- **Drag-and-Drop**: Handle with macOS sandbox compatibility, show "Indexing X files..." toasts with real-time progress
- **Batch Selection**: Use `Set` in Solid signal with `classList={{ "ring-2 ring-brand": selected.has(id) }}` for efficient boolean-only diffing
- **Export Functions**: Implement CSV/JSON sync, PDF export via off-screen WebView
- **Responsive Design**: Collapse filter sidebar to bottom sheet at `window.innerWidth < 950px`
- **Undo System**: Command pattern with inverse operation closures, ⌘Z + 10s "Undo" snackbar

#### Week 11: Advanced UI Features

**Day 1-2: File Preview System**
- **Progressive Tile Rendering**: Use `requestAnimationFrame` to decode only visible tiles at target zoom level
- **Virtualized Thumbnails**: For >1000-page documents with intersection observers
- **Annotation Layer**: Page-relative coordinates with separate `<canvas>` overlay
- **Side-by-Side Comparison**: Leverage Rust diff engine for changed pages/hunks
- **Navigation**: ⌥← / ⌥→ shortcuts, breadcrumb stack with one-animation transitions

**Day 3-4: Timeline and Visualization**
- **Timeline Component**: Use Solid.js + `vis-timeline` with server-side event binning (≤5000 nodes)
- **Date Picker**: Headless `@thisbeyond/solid-datepicker` with focus trapping
- **Charts**: Static Canvas charts (uPlot) to avoid GPU contention with Metal inference
- **Interactive Features**: Zoom levels (day ↔ week ↔ month), IndexedDB caching for navigation

```typescript
// Timeline Component Example
export const TimelineView: Component = () => {
  const [timelineData, setTimelineData] = createSignal<TimelineEvent[]>([])
  const [zoomLevel, setZoomLevel] = createSignal<'day' | 'week' | 'month'>('week')
  
  // Cache timeline payloads in IndexedDB
  const cachedTimeline = createMemo(() => {
    return getCachedTimelineData(zoomLevel())
  })
  
  return (
    <div class="timeline-container">
      {/* vis-timeline integration */}
      {/* Zoom controls */}
      {/* Event details */}
    </div>
  )
}
```

**Day 5-7: Advanced Workflow Features**
- **Collections UI**: Drag-select → C key → modal workflow with manual ordering
- **Bulk Operations**: Rust worker integration with IPC streaming and optimistic updates
- **Search Alerts**: Background notifications with Focus mode awareness
- **Rule Compilation**: Visual rule builder for workflow automation

#### Week 12: User Experience Polish

**Day 1-2: Keyboard Shortcuts & Power Features**
- **Customizable Bindings**: Persist `{action: Shortcut}` JSON with version upgrade merging
- **Command Palette**: Client-side fuzzy matching with `fuse.js` for 0ms latency
- **Global Shortcuts**: Conflict handling with graceful fallback
- **Vim Mode**: Scoped navigation preserving standard ⌘ operations

**Day 3-4: Search History & Personalization**
- **History Reconstruction**: Persist `query_ast`, `ranking_config_hash`, `filters_json`
- **Personalized Dashboard**: Pre-materialized views for instant rendering
- **Privacy Controls**: *Pause History* toggle, auto-cleanup policies
- **Bookmarking**: Tag integration with autocomplete

**Day 5-7: Onboarding & Accessibility**
- **Interactive Tours**: ARIA-live polite regions with Escape dismissal
- **Tooltip System**: Deferred mounting with `setTimeout(0)` post-hydration
- **i18n Infrastructure**: Runtime-loaded JSON bundles (`i18next-solid`)
- **Dark/Light Mode**: System preference respect with manual override
- **Screen Reader Support**: `aria-live="assertive"` for search result announcements

## API Integration

### Tauri Backend Communication

```typescript
// src/api.ts - Backend Communication Layer
import { invoke } from '@tauri-apps/api/core'

export interface SearchResult {
  id: string
  content: string
  file_path: string
  score: number
  file_type: string
  created_at: string
  modified_at: string
}

export interface IndexingStatus {
  indexed: number
  total: number
  current_file?: string
  is_running: boolean
}

export const searchAPI = {
  search: (query: string): Promise<SearchResult[]> => 
    invoke('search_documents', { query }),
  
  indexFile: (path: string): Promise<void> => 
    invoke('index_file', { path }),
  
  getIndexingStatus: (): Promise<IndexingStatus> =>
    invoke('get_indexing_status'),
  
  getFileContent: (path: string): Promise<string> =>
    invoke('get_file_content', { path }),
  
  getTimeline: (start: string, end: string): Promise<TimelineEvent[]> =>
    invoke('get_timeline', { start, end }),
  
  exportResults: (format: 'csv' | 'json' | 'pdf', results: SearchResult[]): Promise<string> =>
    invoke('export_results', { format, results })
}
```

### State Management

```typescript
// src/stores/searchStore.ts - Centralized Search State
import { createSignal, createEffect, createMemo } from 'solid-js'
import { searchAPI } from '../api'

export const [searchQuery, setSearchQuery] = createSignal('')
export const [searchResults, setSearchResults] = createSignal<SearchResult[]>([])
export const [isSearching, setIsSearching] = createSignal(false)
export const [filters, setFilters] = createSignal<SearchFilters>({})
export const [selectedResults, setSelectedResults] = createSignal<Set<string>>(new Set())

// Debounced search effect
createEffect(() => {
  const query = searchQuery()
  const timeoutId = setTimeout(() => {
    if (query.trim()) {
      performSearch(query)
    } else {
      setSearchResults([])
    }
  }, 300)
  
  return () => clearTimeout(timeoutId)
})

export const performSearch = async (query: string) => {
  setIsSearching(true)
  try {
    const results = await searchAPI.search(query)
    setSearchResults(results)
  } catch (error) {
    console.error('Search failed:', error)
    setSearchResults([])
  } finally {
    setIsSearching(false)
  }
}
```

## Component Architecture

### Core Components Structure
```
src/
├── components/
│   ├── SearchInterface.tsx        # Main search UI
│   ├── SearchResults.tsx         # Result display and pagination
│   ├── FilePreview.tsx           # Document preview modal
│   ├── TimelineView.tsx          # Timeline visualization
│   ├── CollectionsPanel.tsx      # Collections and tagging
│   ├── SettingsModal.tsx         # App configuration
│   └── common/
│       ├── Button.tsx
│       ├── Modal.tsx
│       ├── Dropdown.tsx
│       └── LoadingSpinner.tsx
├── stores/
│   ├── searchStore.ts            # Search state management
│   ├── uiStore.ts               # UI state (modals, themes)
│   ├── settingsStore.ts         # App settings
│   └── indexingStore.ts         # Background indexing status
├── types/
│   ├── api.ts                   # Backend API types
│   ├── search.ts                # Search-related types
│   └── ui.ts                    # UI component types
└── utils/
    ├── formatting.ts            # Date, file size formatting
    ├── keyboard.ts              # Keyboard shortcut handling
    └── storage.ts               # LocalStorage utilities
```

### Design System Guidelines

**Color Palette** (using Tailwind CSS variables):
```css
:root {
  --color-primary: 59 130 246;      /* blue-500 */
  --color-secondary: 107 114 128;    /* gray-500 */
  --color-success: 34 197 94;       /* green-500 */
  --color-warning: 245 158 11;      /* amber-500 */
  --color-danger: 239 68 68;        /* red-500 */
  --color-background: 255 255 255;  /* white */
  --color-surface: 249 250 251;     /* gray-50 */
  --color-text: 17 24 39;           /* gray-900 */
}

[data-theme="dark"] {
  --color-background: 17 24 39;     /* gray-900 */
  --color-surface: 31 41 55;        /* gray-800 */
  --color-text: 243 244 246;        /* gray-100 */
}
```

**Typography Scale**:
- `text-xs` (12px): Metadata, timestamps
- `text-sm` (14px): Body text, descriptions  
- `text-base` (16px): Primary UI text
- `text-lg` (18px): Headings, important labels
- `text-xl` (20px): Page titles
- `text-2xl` (24px): Major headings

## Performance Guidelines

### Critical Performance Requirements
- **Search Response**: <200ms for 95% of queries
- **Memory Usage**: <2GB baseline RAM
- **UI Responsiveness**: 60fps animations, <16ms frame time
- **Bundle Size**: <5MB total frontend assets

### Optimization Techniques

**1. Virtualization for Large Lists**
```typescript
// Use intersection observers for infinite scroll
const [visibleRange, setVisibleRange] = createSignal([0, 50])

const virtualizedResults = createMemo(() => {
  const [start, end] = visibleRange()
  return searchResults().slice(start, end)
})
```

**2. Debouncing and RAF**
```typescript
// Combine debouncing with requestAnimationFrame
const optimizedHandler = (callback: Function) => {
  let timeoutId: number
  let rafId: number
  
  return (...args: any[]) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => {
      rafId = requestAnimationFrame(() => callback(...args))
    }, 120)
  }
}
```

**3. Efficient State Updates**
```typescript
// Use createStore for structural diffing
const [uiState, setUiState] = createStore({
  filters: {},
  selectedIds: new Set(),
  viewMode: 'grid'
})

// Batch updates
batch(() => {
  setUiState('filters', 'fileType', 'pdf')
  setUiState('viewMode', 'list')
})
```

## Accessibility Requirements

### WCAG 2.1 AA Compliance
- **Keyboard Navigation**: All functionality accessible via keyboard
- **Screen Reader Support**: Proper ARIA labels and live regions
- **Color Contrast**: 4.5:1 ratio for normal text, 3:1 for large text
- **Focus Management**: Visible focus indicators, logical tab order

### Implementation Examples
```typescript
// Screen reader announcements
const announceSearch = (count: number) => {
  const announcement = `${count} results loaded`
  // Use aria-live region
  document.getElementById('search-announcements')!.textContent = announcement
}

// Keyboard shortcuts
const handleKeyDown = (e: KeyboardEvent) => {
  if (e.metaKey && e.key === 'k') {
    e.preventDefault()
    focusSearchBar()
  }
  if (e.key === 'Escape') {
    closeModals()
  }
}
```

## Testing Strategy

### Frontend Testing Approach
```typescript
// Component testing with Solid Testing Library
import { render, screen, fireEvent } from '@solidjs/testing-library'
import { SearchInterface } from '../SearchInterface'

test('search interface handles user input', async () => {
  render(() => <SearchInterface />)
  
  const searchInput = screen.getByPlaceholderText('Search your documents...')
  fireEvent.input(searchInput, { target: { value: 'test query' } })
  
  expect(searchInput.value).toBe('test query')
})
```

### E2E Testing with Playwright
```typescript
// tests/e2e/search-flow.spec.ts
import { test, expect } from '@playwright/test'

test('complete search workflow', async ({ page }) => {
  await page.goto('/')
  
  // Type search query
  await page.fill('[data-testid="search-input"]', 'test document')
  
  // Wait for results
  await expect(page.locator('[data-testid="search-results"]')).toBeVisible()
  
  // Click first result
  await page.click('[data-testid="result-item"]:first-child')
  
  // Verify preview opens
  await expect(page.locator('[data-testid="file-preview"]')).toBeVisible()
})
```

## Development Workflow

### Getting Started
1. **Prerequisites**: Node.js 18+, Rust, Tauri CLI
2. **Setup**: Clone repo, `npm install` in frontend directory
3. **Development**: `npm run tauri` for hot-reload development
4. **Building**: `npm run tauri:build` for production builds

### Code Quality Tools
- **ESLint**: TypeScript and Solid.js specific rules
- **Prettier**: Code formatting with Tailwind class sorting
- **TypeScript**: Strict mode with comprehensive type checking

### Git Workflow
- **Feature Branches**: `feature/search-interface`, `feature/timeline-view`
- **Commit Convention**: Conventional commits with frontend scope
- **PR Process**: Code review required, automated testing

## Design Patterns

### Component Composition
```typescript
// Prefer composition over large monolithic components
export const SearchPage: Component = () => {
  return (
    <PageLayout>
      <SearchHeader />
      <div class="flex flex-1">
        <SearchSidebar />
        <SearchResults />
        <SearchPreview />
      </div>
    </PageLayout>
  )
}
```

### Event Handling
```typescript
// Use custom events for cross-component communication
const dispatchSearchEvent = (query: string) => {
  window.dispatchEvent(new CustomEvent('search', { detail: { query } }))
}

// Listen in components that need to react
onMount(() => {
  const handleSearch = (e: CustomEvent) => {
    performSearch(e.detail.query)
  }
  window.addEventListener('search', handleSearch)
  onCleanup(() => window.removeEventListener('search', handleSearch))
})
```

## Browser Compatibility

### Target Browsers
- **Primary**: Latest macOS Safari, Chrome
- **Secondary**: Firefox, Edge
- **Minimum**: Safari 14+, Chrome 90+, Firefox 88+

### Tauri WebView Considerations
- No localStorage/sessionStorage - use in-memory state
- Limited IndexedDB - cache sparingly
- WebGL limitations - prefer Canvas for visualizations

## Deployment Considerations

### Build Optimization
```javascript
// vite.config.ts optimizations
export default defineConfig({
  build: {
    target: 'es2020',
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['solid-js', '@solidjs/router'],
          ui: ['@tauri-apps/api']
        }
      }
    }
  }
})
```

### Asset Management
- **Images**: WebP format with fallbacks
- **Icons**: SVG sprites for efficiency
- **Fonts**: System fonts primarily, minimal web fonts

This guide provides the frontend developer with comprehensive guidance for building a high-performance, accessible, and maintainable user interface for the Desktop AI Search application while working independently from the backend development.