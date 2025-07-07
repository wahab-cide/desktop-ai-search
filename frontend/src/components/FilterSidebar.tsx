import { Component, For, Show, createSignal } from 'solid-js'
import { filters, setFilters, availableFileTypes, dateRanges } from '../stores/searchStore'
import { theme, themeClasses } from '../stores/themeStore'

export const FilterSidebar: Component = () => {
  const [isExpanded, setIsExpanded] = createSignal(true)
  const [activeSection, setActiveSection] = createSignal<string | null>(null)
  
  const handleFileTypeToggle = (fileType: string) => {
    const currentFilters = filters()
    const newFileTypes = currentFilters.fileTypes.includes(fileType)
      ? currentFilters.fileTypes.filter(ft => ft !== fileType)
      : [...currentFilters.fileTypes, fileType]
    
    setFilters({
      ...currentFilters,
      fileTypes: newFileTypes
    })
  }
  
  const handleDateRangeChange = (range: string) => {
    setFilters({
      ...filters(),
      dateRange: range
    })
  }

  const getActiveFilterCount = () => {
    const f = filters()
    let count = 0
    if (f.fileTypes.length > 0) count++
    if (f.dateRange !== 'all') count++
    if (f.searchMode !== 'hybrid') count++
    if (!f.includeContent || !f.includeOCR || !f.searchImages) count++
    return count
  }
  
  return (
    <div class={`${themeClasses.bgTertiary()} border-r ${themeClasses.border()} transition-all duration-300 shadow-lg backdrop-blur-sm ${
      isExpanded() ? 'w-72' : 'w-16'
    }`}>
      <div class="h-full flex flex-col">
        {/* Header */}
        <div class="p-4 border-b border-zinc-800/60">
          <div class="flex items-center justify-between">
            <Show when={isExpanded()}>
              <div class="flex items-center space-x-3">
                <div class="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center border border-purple-500/30">
                  <svg class="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                  </svg>
                </div>
                <div>
                  <h3 class={`font-semibold ${themeClasses.textSecondary()}`}>Filters</h3>
                  <Show when={getActiveFilterCount() > 0}>
                    <span class="text-xs text-purple-400">{getActiveFilterCount()} active</span>
                  </Show>
                </div>
              </div>
            </Show>
            <button
              onClick={() => setIsExpanded(!isExpanded())}
              class={`p-2 rounded-lg ${themeClasses.hover()} transition-colors`}
              title={isExpanded() ? 'Collapse sidebar' : 'Expand sidebar'}
            >
              <svg
                class={`w-4 h-4 ${themeClasses.textMuted()} transform transition-transform ${
                  isExpanded() ? 'rotate-180' : ''
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M11 19l-7-7 7-7m8 14l-7-7 7-7"
                />
              </svg>
            </button>
          </div>
        </div>
        
        <Show when={isExpanded()}>
          <div class="flex-1 overflow-y-auto">
            <div class="p-4 space-y-6">
              {/* File Types Section */}
              <div class={`rounded-xl p-4 border ${theme() === 'light' ? 'bg-gray-50/80 border-gray-200' : 'bg-black/40 border-zinc-800/40'}`}>
                <div class="flex items-center justify-between mb-4">
                  <h4 class={`text-sm font-semibold ${themeClasses.textSecondary()} flex items-center`}>
                    <svg class={`w-4 h-4 mr-2 ${themeClasses.textMuted()}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    File Types
                  </h4>
                  <Show when={filters().fileTypes.length > 0}>
                    <span class="text-xs bg-purple-500/20 text-purple-300 px-2 py-1 rounded-full border border-purple-500/30">
                      {filters().fileTypes.length}
                    </span>
                  </Show>
                </div>
                <div class="space-y-2">
                  <For each={availableFileTypes()}>
                    {(fileType) => (
                      <label class={`flex items-center space-x-3 p-3 rounded-lg ${themeClasses.hover()} transition-colors cursor-pointer group`}>
                        <input
                          type="checkbox"
                          class={`rounded text-purple-500 focus:ring-purple-500 focus:ring-2 ${theme() === 'light' ? 'border-gray-300 bg-white' : 'border-zinc-700 bg-zinc-900'}`}
                          checked={filters().fileTypes.includes(fileType.type)}
                          onChange={() => handleFileTypeToggle(fileType.type)}
                        />
                        <span class={`flex-1 text-sm font-medium ${themeClasses.textSecondary()} group-hover:${themeClasses.text()}`}>
                          {fileType.label}
                        </span>
                        <span class={`text-xs ${themeClasses.textMuted()} px-2 py-1 rounded-full ${theme() === 'light' ? 'bg-gray-100' : 'bg-zinc-800/50'}`}>
                          {fileType.count}
                        </span>
                      </label>
                    )}
                  </For>
                </div>
              </div>
              
              {/* Date Range Section */}
              <div class={`rounded-xl p-4 border ${theme() === 'light' ? 'bg-gray-50/80 border-gray-200' : 'bg-black/40 border-zinc-800/40'}`}>
                <div class="flex items-center justify-between mb-4">
                  <h4 class={`text-sm font-semibold ${themeClasses.textSecondary()} flex items-center`}>
                    <svg class={`w-4 h-4 mr-2 ${themeClasses.textMuted()}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    Date Modified
                  </h4>
                  <Show when={filters().dateRange !== 'all'}>
                    <span class={`text-xs px-2 py-1 rounded-full border ${theme() === 'light' ? 'bg-green-100 text-green-700 border-green-300' : 'bg-green-500/20 text-green-300 border-green-500/30'}`}>
                      Active
                    </span>
                  </Show>
                </div>
                <div class="space-y-2">
                  <For each={dateRanges}>
                    {(range) => (
                      <label class={`flex items-center space-x-3 p-3 rounded-lg ${themeClasses.hover()} transition-colors cursor-pointer group`}>
                        <input
                          type="radio"
                          name="dateRange"
                          class={`text-purple-500 focus:ring-purple-500 focus:ring-2 ${theme() === 'light' ? 'border-gray-300 bg-white' : 'border-zinc-600 bg-zinc-800'}`}
                          checked={filters().dateRange === range.value}
                          onChange={() => handleDateRangeChange(range.value)}
                        />
                        <span class={`text-sm font-medium ${themeClasses.textSecondary()} group-hover:${themeClasses.text()}`}>
                          {range.label}
                        </span>
                      </label>
                    )}
                  </For>
                </div>
              </div>
              
              {/* Search Mode Section */}
              <div class={`rounded-xl p-4 border ${theme() === 'light' ? 'bg-gray-50/80 border-gray-200' : 'bg-black/40 border-zinc-700/30'}`}>
                <h4 class={`text-sm font-semibold ${themeClasses.textSecondary()} mb-4 flex items-center`}>
                  <svg class={`w-4 h-4 mr-2 ${themeClasses.textMuted()}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  Search Intelligence
                </h4>
                <div class="space-y-2">
                  <For each={[
                    { value: 'hybrid', label: '⚡ Hybrid', desc: 'Best match combining exact and semantic' },
                    { value: 'exact', label: '🎯 Exact', desc: 'Precise keyword matching only' },
                    { value: 'semantic', label: '🧠 Semantic', desc: 'Meaning-based AI search' },
                    { value: 'fuzzy', label: '🔍 Fuzzy', desc: 'Flexible matching with typos' }
                  ]}>
                    {(mode) => (
                      <label class={`flex items-start space-x-3 p-3 rounded-lg ${themeClasses.hover()} transition-colors cursor-pointer group border ${theme() === 'light' ? 'border-gray-200 hover:border-gray-300' : 'border-zinc-700/30 hover:border-zinc-600/50'}`}>
                        <input
                          type="radio"
                          name="searchMode"
                          class={`text-purple-500 focus:ring-purple-500 focus:ring-2 mt-1 ${theme() === 'light' ? 'border-gray-300 bg-white' : 'border-zinc-600 bg-zinc-800'}`}
                          checked={filters().searchMode === mode.value}
                          onChange={() => setFilters({ ...filters(), searchMode: mode.value })}
                        />
                        <div class="flex-1 min-w-0">
                          <div class={`text-sm font-medium ${themeClasses.textSecondary()} group-hover:${themeClasses.text()}`}>
                            {mode.label}
                          </div>
                          <div class={`text-xs ${themeClasses.textMuted()} mt-1`}>
                            {mode.desc}
                          </div>
                        </div>
                      </label>
                    )}
                  </For>
                </div>
              </div>
              
              {/* Advanced Options Section */}
              <div class={`rounded-xl p-4 border ${theme() === 'light' ? 'bg-gray-50/80 border-gray-200' : 'bg-black/40 border-zinc-700/30'}`}>
                <h4 class={`text-sm font-semibold ${themeClasses.textSecondary()} mb-4 flex items-center`}>
                  <svg class={`w-4 h-4 mr-2 ${themeClasses.textMuted()}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
                  </svg>
                  Advanced Options
                </h4>
                <div class="space-y-3">
                  <label class={`flex items-center space-x-3 p-3 rounded-lg ${themeClasses.hover()} transition-colors cursor-pointer group`}>
                    <input
                      type="checkbox"
                      class={`rounded text-purple-500 focus:ring-purple-500 focus:ring-2 ${theme() === 'light' ? 'border-gray-300 bg-white' : 'border-zinc-600 bg-zinc-800'}`}
                      checked={filters().includeContent}
                      onChange={(e) => setFilters({ ...filters(), includeContent: e.currentTarget.checked })}
                    />
                    <div class="flex-1 min-w-0">
                      <div class={`text-sm font-medium ${themeClasses.textSecondary()} group-hover:${themeClasses.text()}`}>
                        📄 Search file content
                      </div>
                      <div class={`text-xs ${themeClasses.textMuted()}`}>
                        Search inside document text
                      </div>
                    </div>
                  </label>
                  <label class={`flex items-center space-x-3 p-3 rounded-lg ${themeClasses.hover()} transition-colors cursor-pointer group`}>
                    <input
                      type="checkbox"
                      class={`rounded text-purple-500 focus:ring-purple-500 focus:ring-2 ${theme() === 'light' ? 'border-gray-300 bg-white' : 'border-zinc-600 bg-zinc-800'}`}
                      checked={filters().includeOCR}
                      onChange={(e) => setFilters({ ...filters(), includeOCR: e.currentTarget.checked })}
                    />
                    <div class="flex-1 min-w-0">
                      <div class={`text-sm font-medium ${themeClasses.textSecondary()} group-hover:${themeClasses.text()}`}>
                        🔍 Include OCR text
                      </div>
                      <div class={`text-xs ${themeClasses.textMuted()}`}>
                        Search text extracted from images
                      </div>
                    </div>
                  </label>
                  <label class={`flex items-center space-x-3 p-3 rounded-lg ${themeClasses.hover()} transition-colors cursor-pointer group`}>
                    <input
                      type="checkbox"
                      class={`rounded text-purple-500 focus:ring-purple-500 focus:ring-2 ${theme() === 'light' ? 'border-gray-300 bg-white' : 'border-zinc-600 bg-zinc-800'}`}
                      checked={filters().searchImages}
                      onChange={(e) => setFilters({ ...filters(), searchImages: e.currentTarget.checked })}
                    />
                    <div class="flex-1 min-w-0">
                      <div class={`text-sm font-medium ${themeClasses.textSecondary()} group-hover:${themeClasses.text()}`}>
                        🖼️ Search images (CLIP)
                      </div>
                      <div class={`text-xs ${themeClasses.textMuted()}`}>
                        AI-powered image understanding
                      </div>
                    </div>
                  </label>
                </div>
              </div>
              
              {/* File Size Filter */}
              <div class={`rounded-xl p-4 border ${theme() === 'light' ? 'bg-gray-50/80 border-gray-200' : 'bg-black/40 border-zinc-700/30'}`}>
                <h4 class={`text-sm font-semibold ${themeClasses.textSecondary()} mb-4 flex items-center`}>
                  <svg class={`w-4 h-4 mr-2 ${themeClasses.textMuted()}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                  </svg>
                  File Size
                </h4>
                <div class="space-y-2">
                  <For each={[
                    { value: 'any', label: 'Any size' },
                    { value: 'small', label: 'Small (< 1MB)' },
                    { value: 'medium', label: 'Medium (1-10MB)' },
                    { value: 'large', label: 'Large (> 10MB)' }
                  ]}>
                    {(size) => (
                      <label class={`flex items-center space-x-3 p-3 rounded-lg ${themeClasses.hover()} transition-colors cursor-pointer group`}>
                        <input
                          type="radio"
                          name="fileSize"
                          class={`text-purple-500 focus:ring-purple-500 focus:ring-2 ${theme() === 'light' ? 'border-gray-300 bg-white' : 'border-zinc-600 bg-zinc-800'}`}
                          checked={filters().fileSize === size.value}
                          onChange={() => setFilters({ ...filters(), fileSize: size.value })}
                        />
                        <span class={`text-sm font-medium ${themeClasses.textSecondary()} group-hover:${themeClasses.text()}`}>
                          {size.label}
                        </span>
                      </label>
                    )}
                  </For>
                </div>
              </div>
            </div>
          </div>
          
          {/* Footer Actions */}
          <div class={`p-4 border-t ${theme() === 'light' ? 'border-gray-200 bg-gray-50' : 'border-zinc-700/50 bg-zinc-800/30'}`}>
            <div class="space-y-3">
              <button
                class="w-full text-sm text-red-400 hover:text-red-300 py-3 px-4 border border-red-500/30 rounded-lg hover:bg-red-500/10 transition-colors font-medium"
                onClick={() => setFilters({
                  fileTypes: [],
                  dateRange: 'all',
                  searchMode: 'hybrid',
                  includeContent: true,
                  includeOCR: true,
                  searchImages: true,
                  fileSize: 'any'
                })}
              >
                Clear All Filters
              </button>
              <div class={`text-xs ${themeClasses.textMuted()} text-center`}>
                <Show when={getActiveFilterCount() > 0}>
                  <span class="text-purple-400">{getActiveFilterCount()}</span> filter{getActiveFilterCount() > 1 ? 's' : ''} active
                </Show>
                <Show when={getActiveFilterCount() === 0}>
                  No filters applied
                </Show>
              </div>
            </div>
          </div>
        </Show>
      </div>
    </div>
  )
}