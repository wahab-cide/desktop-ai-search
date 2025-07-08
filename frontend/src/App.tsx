import type { Component } from 'solid-js';
import { Show, createSignal } from 'solid-js';
import './index.css';
import { SearchBar } from './components/SearchBar';
import { SearchResults } from './components/SearchResults';
import { FilterSidebar } from './components/FilterSidebar';
import { IndexingPanel } from './components/IndexingPanel';
import { searchMetadata, searchQuery } from './stores/searchStore';
import { theme, toggleTheme, themeClasses } from './stores/themeStore';

const App: Component = () => {
  const [showSettings, setShowSettings] = createSignal(false)
  
  return (
    <div class={`h-screen flex flex-col ${themeClasses.bg()}`}>
      {/* Header */}
      <header class={`${themeClasses.bgSecondary()} border-b ${themeClasses.border()} backdrop-blur-sm relative z-50`}>
        <div class="px-6 py-4">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-3">
              <div class="w-8 h-8 bg-gradient-to-br from-purple-500 to-blue-500 rounded-lg flex items-center justify-center">
                <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <h1 class={`text-xl font-medium ${themeClasses.text()}`}>Desktop AI Search</h1>
            </div>
            
            <div class="flex items-center space-x-2">
              {/* Theme Toggle */}
              <button 
                class={`p-2 ${themeClasses.textMuted()} ${themeClasses.hover()} rounded-lg transition-all duration-200 hover:${themeClasses.textSecondary()}`}
                onClick={toggleTheme}
                title={`Switch to ${theme() === 'light' ? 'dark' : 'light'} mode`}
              >
                <Show 
                  when={theme() === 'light'} 
                  fallback={
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                    </svg>
                  }
                >
                  <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                  </svg>
                </Show>
              </button>
              
              {/* Settings */}
              <button 
                class={`p-2 ${themeClasses.textMuted()} ${themeClasses.hover()} rounded-lg transition-all duration-200 hover:${themeClasses.textSecondary()}`}
                onClick={() => setShowSettings(!showSettings())}
                title="Settings & Indexing"
              >
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
            </div>
          </div>
        </div>
        
        {/* Search Bar */}
        <div class="px-6 pb-6 relative z-50">
          <SearchBar />
          
          {/* Query Intent Display */}
          <Show when={searchMetadata()?.queryIntent && searchQuery()}>
            <div class={`mt-3 text-sm ${themeClasses.textMuted()}`}>
              <span>Searching for: </span>
              <span class={`font-medium ${themeClasses.textSecondary()}`}>{searchMetadata()!.queryIntent}</span>
            </div>
          </Show>
        </div>
      </header>
      
      {/* Main Content */}
      <div class="flex flex-1 overflow-hidden relative">
        <FilterSidebar />
        <Show when={!showSettings()} fallback={
          <div class="flex-1 p-6 overflow-y-auto">
            <IndexingPanel />
          </div>
        }>
          <SearchResults />
        </Show>
      </div>
      
      {/* Status Bar */}
      <div class={`${themeClasses.bgTertiary()} border-t ${themeClasses.border()} px-6 py-3 backdrop-blur-sm`}>
        <div class={`flex items-center justify-between text-xs ${themeClasses.textMuted()}`}>
          <div class="flex items-center space-x-4">
            <div class="flex items-center space-x-2">
              <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>Ready</span>
            </div>
            <Show when={searchMetadata()}>
              <span>•</span>
              <span class={themeClasses.textSecondary()}>
                {searchMetadata()!.totalResults} results in {searchMetadata()!.searchTime.toFixed(2)}s
              </span>
            </Show>
          </div>
          <div class="flex items-center space-x-4">
            <span>Index: 23,456 documents</span>
            <span>•</span>
            <span>Last updated: 2 hours ago</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;