import { Component, Show, For, createSignal, createEffect } from 'solid-js'
import { searchQuery, setSearchQuery, searchSuggestions, isLoadingSuggestions } from '../stores/searchStore'
import { theme, themeClasses } from '../stores/themeStore'

export const SearchBar: Component = () => {
  const [isFocused, setIsFocused] = createSignal(false)
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = createSignal(-1)
  const [searchMode, setSearchMode] = createSignal<'auto' | 'precise' | 'semantic'>('auto')
  const [showAdvanced, setShowAdvanced] = createSignal(false)

  const handleKeyDown = (e: KeyboardEvent) => {
    const suggestions = searchSuggestions()
    
    // Handle keyboard shortcuts
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedSuggestionIndex(Math.min(selectedSuggestionIndex() + 1, suggestions.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedSuggestionIndex(Math.max(selectedSuggestionIndex() - 1, -1))
    } else if (e.key === 'Enter' && selectedSuggestionIndex() >= 0) {
      e.preventDefault()
      setSearchQuery(suggestions[selectedSuggestionIndex()].query)
      setSelectedSuggestionIndex(-1)
      setIsFocused(false)
    } else if (e.key === 'Escape') {
      setIsFocused(false)
      setSelectedSuggestionIndex(-1)
    } else if (e.key === 'Tab' && e.shiftKey) {
      e.preventDefault()
      setShowAdvanced(!showAdvanced())
    }
  }

  // Smart placeholder text based on search mode
  const getPlaceholder = () => {
    switch (searchMode()) {
      case 'precise':
        return 'Search with exact keywords... (try: "machine learning" type:pdf)'
      case 'semantic':
        return 'Search by meaning and concepts... (try: "documents about AI")'
      default:
        return 'Search documents, images, emails... (try: "invoice from last month" or "type:pdf machine learning")'
    }
  }

  // Detect search intent from query
  const getSearchIntent = () => {
    const query = searchQuery().toLowerCase()
    if (query.includes('type:') || query.includes('"')) return 'precise'
    if (query.includes('about') || query.includes('related to') || query.includes('similar')) return 'semantic'
    return 'auto'
  }

  return (
    <div class="relative space-y-4">
      {/* Main Search Bar */}
      <div class="relative">
        <div class={`relative transition-all duration-300 ${
          isFocused() ? 'ring-1 ring-purple-500/50 shadow-2xl shadow-purple-500/10' : 'shadow-lg'
        } rounded-2xl ${themeClasses.input()} backdrop-blur-sm`}>
          
          {/* Search Icon */}
          <div class="absolute left-4 top-1/2 transform -translate-y-1/2">
            <svg
              class={`w-5 h-5 transition-colors duration-200 ${
                isFocused() ? 'text-purple-400' : themeClasses.textMuted()
              }`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>
          
          {/* Search Input */}
          <input
            type="text"
            placeholder={getPlaceholder()}
            class={`w-full h-16 pl-12 pr-32 text-lg bg-transparent border-0 focus:outline-none focus:ring-0 ${theme() === 'light' ? 'placeholder-gray-500 text-gray-900' : 'placeholder-zinc-500 text-white'}`}
            value={searchQuery()}
            onInput={(e) => setSearchQuery(e.currentTarget.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setTimeout(() => setIsFocused(false), 200)}
            onKeyDown={handleKeyDown}
          />
          
          {/* Right Side Controls */}
          <div class="absolute right-3 top-1/2 transform -translate-y-1/2 flex items-center space-x-2">
            
            {/* Search Mode Indicator */}
            <Show when={searchQuery().length > 0}>
              <div class={`px-3 py-1.5 rounded-full text-xs font-medium ${
                getSearchIntent() === 'precise' 
                  ? 'bg-green-500/20 text-green-300 border border-green-500/30'
                  : getSearchIntent() === 'semantic'
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                  : 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
              }`}>
                {getSearchIntent() === 'precise' && '🎯 Precise'}
                {getSearchIntent() === 'semantic' && '🧠 Semantic'}
                {getSearchIntent() === 'auto' && '⚡ Smart'}
              </div>
            </Show>
            
            {/* Advanced Toggle */}
            <button
              class={`p-2 rounded-lg transition-all duration-200 ${
                showAdvanced() 
                  ? 'bg-purple-500/20 text-purple-300 hover:bg-purple-500/30' 
                  : `${themeClasses.textMuted()} hover:${themeClasses.textSecondary()} ${themeClasses.hover()}`
              }`}
              onClick={() => setShowAdvanced(!showAdvanced())}
              title="Advanced Search (Shift+Tab)"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
              </svg>
            </button>
            
            {/* Clear Button */}
            <Show when={searchQuery().length > 0}>
              <button
                class={`p-2 ${themeClasses.textMuted()} hover:${themeClasses.textSecondary()} ${themeClasses.hover()} rounded-lg transition-all duration-200`}
                onClick={() => setSearchQuery('')}
                title="Clear search"
              >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </Show>
          </div>
        </div>
      </div>

      {/* Advanced Search Panel */}
      <Show when={showAdvanced()}>
        <div class={`${themeClasses.card()} rounded-xl p-5 space-y-5 shadow-lg backdrop-blur-sm animate-fadeIn`}>
          <div class="flex items-center justify-between">
            <h3 class={`text-sm font-medium ${themeClasses.textSecondary()}`}>Advanced Search</h3>
            <span class={`text-xs ${themeClasses.textMuted()}`}>Shift+Tab to toggle</span>
          </div>
          
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Search Mode */}
            <div>
              <label class={`block text-xs font-medium ${themeClasses.textSecondary()} mb-3`}>Search Mode</label>
              <div class="space-y-2">
                <For each={[
                  { key: 'auto', label: '⚡ Smart', desc: 'Automatic' },
                  { key: 'precise', label: '🎯 Precise', desc: 'Exact matches' },
                  { key: 'semantic', label: '🧠 Semantic', desc: 'Meaning-based' }
                ]}>
                  {(mode) => (
                    <button
                      class={`w-full p-3 text-xs rounded-lg border transition-all duration-200 text-left ${
                        searchMode() === mode.key
                          ? 'border-purple-500/50 bg-purple-500/10 text-purple-300'
                          : `${themeClasses.borderSecondary()} ${theme() === 'light' ? 'bg-gray-50 text-gray-600 hover:bg-gray-100 hover:text-gray-700' : 'bg-zinc-800/50 text-zinc-400 hover:bg-zinc-700/50 hover:text-zinc-300'}`
                      }`}
                      onClick={() => setSearchMode(mode.key as any)}
                    >
                      <div class="font-medium">{mode.label}</div>
                      <div class="text-xs opacity-75 mt-1">{mode.desc}</div>
                    </button>
                  )}
                </For>
              </div>
            </div>
            
            {/* File Types */}
            <div>
              <label class={`block text-xs font-medium ${themeClasses.textSecondary()} mb-3`}>File Types</label>
              <div class="flex flex-wrap gap-2">
                <For each={['PDF', 'DOC', 'TXT', 'IMG', 'VID', 'AUD']}>
                  {(type) => (
                    <button class={`px-3 py-2 text-xs rounded-lg transition-colors border ${theme() === 'light' ? 'bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-gray-700 border-gray-300' : 'bg-zinc-800/50 text-zinc-300 hover:bg-zinc-700/50 hover:text-zinc-200 border-zinc-700/50'}`}>
                      {type}
                    </button>
                  )}
                </For>
              </div>
            </div>
            
            {/* Date Range */}
            <div>
              <label class={`block text-xs font-medium ${themeClasses.textSecondary()} mb-3`}>Time Range</label>
              <select class={`w-full text-xs border rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-purple-500 focus:border-purple-500 ${theme() === 'light' ? 'border-gray-300 bg-white text-gray-900' : 'border-zinc-700 bg-zinc-800/50 text-zinc-300'}`}>
                <option>Any time</option>
                <option>Last hour</option>
                <option>Last 24 hours</option>
                <option>Last week</option>
                <option>Last month</option>
                <option>Last year</option>
              </select>
            </div>
          </div>
          
          {/* Quick Filters */}
          <div>
            <label class={`block text-xs font-medium ${themeClasses.textSecondary()} mb-3`}>Quick Filters</label>
            <div class="flex flex-wrap gap-2">
              <For each={[
                'Recent documents',
                'Large files (>10MB)', 
                'Images with text',
                'Frequently accessed',
                'Shared files',
                'Has attachments'
              ]}>
                {(filter) => (
                  <button class={`px-3 py-2 text-xs rounded-full hover:bg-purple-500/20 hover:text-purple-300 transition-colors border hover:border-purple-500/30 ${theme() === 'light' ? 'bg-gray-100/50 text-gray-500 border-gray-200' : 'bg-zinc-800/30 text-zinc-400 border-zinc-700/30'}`}>
                    {filter}
                  </button>
                )}
              </For>
            </div>
          </div>
        </div>
      </Show>
      
      <Show when={isFocused() && (searchSuggestions().length > 0 || isLoadingSuggestions())}>
        <div class={`absolute top-full left-0 right-0 mt-2 rounded-xl z-[9999] max-h-96 overflow-y-auto backdrop-blur-md animate-fadeIn border-2 ${theme() === 'light' ? 'bg-white border-gray-300 shadow-xl shadow-gray-200/50' : 'bg-zinc-900 border-zinc-600 shadow-xl shadow-black/20'}`}>
          <Show when={isLoadingSuggestions()}>
            <div class={`px-4 py-4 text-sm ${themeClasses.textMuted()} flex items-center space-x-3`}>
              <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-purple-500"></div>
              <span>Finding suggestions...</span>
            </div>
          </Show>
          
          <For each={searchSuggestions()}>
            {(suggestion, index) => (
              <div
                class={`px-4 py-3 cursor-pointer transition-all duration-200 group ${
                  index() === selectedSuggestionIndex()
                    ? 'bg-gradient-to-r from-purple-500/10 to-blue-500/10 text-purple-300 shadow-sm border-l-2 border-purple-500'
                    : `${theme() === 'light' ? 'hover:bg-gray-50 text-gray-700' : 'hover:bg-zinc-800/30 text-zinc-300'}`
                } ${index() === 0 ? 'rounded-t-xl' : ''} ${
                  index() === searchSuggestions().length - 1 ? 'rounded-b-xl' : ''
                }`}
                onMouseEnter={() => setSelectedSuggestionIndex(index())}
                onClick={() => {
                  setSearchQuery(suggestion.query)
                  setIsFocused(false)
                }}
              >
                <div class="flex items-center justify-between">
                  <div class="flex items-center space-x-3">
                    <div class={`p-1.5 rounded-lg transition-all duration-200 ${
                      index() === selectedSuggestionIndex()
                        ? 'bg-purple-500/20 text-purple-300'
                        : theme() === 'light' ? 'bg-gray-100 text-gray-500 group-hover:bg-gray-200 group-hover:text-gray-600' : 'bg-zinc-800/50 text-zinc-500 group-hover:bg-zinc-700/50 group-hover:text-zinc-400'
                    }`}>
                      <svg
                        class="w-3.5 h-3.5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <Show
                          when={suggestion.type === 'history'}
                          fallback={
                            <path
                              stroke-linecap="round"
                              stroke-linejoin="round"
                              stroke-width="2"
                              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                            />
                          }
                        >
                          <path
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            stroke-width="2"
                            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                          />
                        </Show>
                      </svg>
                    </div>
                    <div class="flex-1 min-w-0">
                      <div class="text-sm font-medium truncate">{suggestion.query}</div>
                      <Show when={suggestion.category}>
                        <div class={`text-xs ${themeClasses.textMuted()} truncate`}>{suggestion.category}</div>
                      </Show>
                    </div>
                  </div>
                  <div class="flex items-center space-x-2">
                    <Show when={suggestion.count}>
                      <span class={`text-xs px-2 py-1 rounded-full font-medium ${
                        index() === selectedSuggestionIndex()
                          ? 'bg-purple-500/20 text-purple-300'
                          : theme() === 'light' ? 'bg-gray-100 text-gray-500' : 'bg-zinc-800/50 text-zinc-400'
                      }`}>
                        {suggestion.count}
                      </span>
                    </Show>
                    <Show when={suggestion.type === 'recent' || suggestion.type === 'popular'}>
                      <div class={`text-xs px-2 py-1 rounded-full ${
                        suggestion.type === 'recent' 
                          ? 'bg-green-500/20 text-green-300' 
                          : 'bg-orange-500/20 text-orange-300'
                      }`}>
                        {suggestion.type === 'recent' ? '🕒' : '🔥'}
                      </div>
                    </Show>
                    <svg
                      class={`w-4 h-4 transition-all duration-200 ${
                        index() === selectedSuggestionIndex()
                          ? 'text-purple-400 transform translate-x-1'
                          : theme() === 'light' ? 'text-gray-400 group-hover:text-gray-500' : 'text-zinc-600 group-hover:text-zinc-500'
                      }`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M9 5l7 7-7 7"
                      />
                    </svg>
                  </div>
                </div>
              </div>
            )}
          </For>
        </div>
      </Show>
    </div>
  )
}