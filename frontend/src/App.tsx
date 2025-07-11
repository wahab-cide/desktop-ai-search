import type { Component } from 'solid-js';
import { Show, createSignal, onMount } from 'solid-js';
import './index.css';
import { SearchBar } from './components/SearchBar';
import { SearchResults } from './components/SearchResults';
import { FilterSidebar } from './components/FilterSidebar';
import { IndexingPanel } from './components/IndexingPanel';
import { KeyboardShortcutsModal } from './components/KeyboardShortcutsModal';
import { searchMetadata, searchQuery, setSearchQuery } from './stores/searchStore';
import { theme, effectiveTheme, toggleTheme, themeClasses } from './stores/themeStore';
import { registerShortcut, unregisterShortcut, setShowShortcutsModal } from './stores/keyboardStore';
import { searchAPI } from './api';

const App: Component = () => {
  const [showSettings, setShowSettings] = createSignal(false)
  const [isRebuilding, setIsRebuilding] = createSignal(false)
  const [rebuildStatus, setRebuildStatus] = createSignal<string>('')
  
  const handleRebuildIndex = async () => {
    setIsRebuilding(true)
    setRebuildStatus('Rebuilding search index...')
    try {
      const result = await searchAPI.rebuildSearchIndex()
      setRebuildStatus(result)
      // Clear status after 3 seconds
      setTimeout(() => setRebuildStatus(''), 3000)
    } catch (error) {
      setRebuildStatus(`Error: ${error}`)
    } finally {
      setIsRebuilding(false)
    }
  }
  
  // Register global keyboard shortcuts
  onMount(() => {
    const shortcuts = [
      {
        key: 'k',
        ctrlKey: true,
        action: () => {
          // Focus search bar
          const searchInput = document.querySelector('input[type="text"]') as HTMLInputElement
          if (searchInput) {
            searchInput.focus()
            searchInput.select()
          }
        },
        description: 'Focus search bar',
        category: 'Navigation'
      },
      {
        key: 'k',
        metaKey: true,
        action: () => {
          // Focus search bar (Mac)
          const searchInput = document.querySelector('input[type="text"]') as HTMLInputElement
          if (searchInput) {
            searchInput.focus()
            searchInput.select()
          }
        },
        description: 'Focus search bar',
        category: 'Navigation'
      },
      {
        key: 'Escape',
        action: () => {
          // Clear search and unfocus
          setSearchQuery('')
          const searchInput = document.querySelector('input[type="text"]') as HTMLInputElement
          if (searchInput) {
            searchInput.blur()
          }
        },
        description: 'Clear search and unfocus',
        category: 'Navigation'
      },
      {
        key: 't',
        ctrlKey: true,
        action: () => toggleTheme(),
        description: 'Toggle theme',
        category: 'Interface'
      },
      {
        key: 't',
        metaKey: true,
        action: () => toggleTheme(),
        description: 'Toggle theme',
        category: 'Interface'
      },
      {
        key: 's',
        ctrlKey: true,
        action: () => setShowSettings(!showSettings()),
        description: 'Toggle settings panel',
        category: 'Interface'
      },
      {
        key: 's',
        metaKey: true,
        action: () => setShowSettings(!showSettings()),
        description: 'Toggle settings panel',
        category: 'Interface'
      },
      {
        key: '?',
        action: () => setShowShortcutsModal(true),
        description: 'Show keyboard shortcuts',
        category: 'Help'
      },
      {
        key: 'r',
        ctrlKey: true,
        shiftKey: true,
        action: () => {
          handleRebuildIndex()
        },
        description: 'Rebuild search index',
        category: 'Search'
      }
    ]
    
    shortcuts.forEach(shortcut => registerShortcut(shortcut))
    
    return () => {
      shortcuts.forEach(shortcut => unregisterShortcut(shortcut))
    }
  })
  
  return (
    <div class={`h-screen flex flex-col ${themeClasses.bg()}`}>
      {/* Header */}
      <header class={`${themeClasses.bgSecondary()} border-b ${themeClasses.border()} backdrop-blur-sm relative z-50`}>
        <div class="px-4 md:px-6 py-4">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-3">
              <div class="w-8 h-8 bg-gradient-to-br from-purple-500 to-blue-500 rounded-lg flex items-center justify-center">
                <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <h1 class={`text-lg md:text-xl font-medium ${themeClasses.text()} hidden sm:block`}>Desktop AI Search</h1>
              <h1 class={`text-lg font-medium ${themeClasses.text()} sm:hidden`}>AI Search</h1>
            </div>
            
            <div class="flex items-center space-x-2">
              {/* Theme Toggle */}
              <button 
                class={`relative p-2 ${themeClasses.textMuted()} ${themeClasses.hover()} rounded-lg transition-all duration-200 hover:${themeClasses.textSecondary()}`}
                onClick={toggleTheme}
                title={`Theme: ${theme() === 'system' ? 'System' : theme() === 'light' ? 'Light' : 'Dark'} (click to cycle)`}
              >
                <Show 
                  when={theme() === 'light'} 
                  fallback={
                    <Show
                      when={theme() === 'dark'}
                      fallback={
                        <div class="relative">
                          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                          </svg>
                          <div class="absolute -top-1 -right-1 w-3 h-3 bg-blue-500 rounded-full border-2 border-white dark:border-black"></div>
                        </div>
                      }
                    >
                      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                      </svg>
                    </Show>
                  }
                >
                  <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                  </svg>
                </Show>
              </button>
              
              {/* Rebuild Index Button */}
              <button 
                class={`p-2 ${themeClasses.textMuted()} ${themeClasses.hover()} rounded-lg transition-all duration-200 hover:${themeClasses.textSecondary()} ${isRebuilding() ? 'animate-spin' : ''}`}
                onClick={handleRebuildIndex}
                disabled={isRebuilding()}
                title="Rebuild search index (Ctrl+Shift+R)"
              >
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
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
        <div class="px-4 md:px-6 pb-6 relative z-50">
          <SearchBar />
          
          {/* Query Intent Display */}
          <Show when={searchMetadata()?.queryIntent && searchQuery()}>
            <div class={`mt-3 text-sm ${themeClasses.textMuted()}`}>
              <span>Searching for: </span>
              <span class={`font-medium ${themeClasses.textSecondary()}`}>{searchMetadata()!.queryIntent}</span>
            </div>
          </Show>
          
          {/* Rebuild Status Display */}
          <Show when={rebuildStatus()}>
            <div class={`mt-3 text-sm px-3 py-2 rounded-lg ${
              rebuildStatus().includes('Error') 
                ? 'bg-red-500/10 text-red-400 border border-red-500/30' 
                : rebuildStatus().includes('✅')
                  ? 'bg-green-500/10 text-green-400 border border-green-500/30'
                  : 'bg-blue-500/10 text-blue-400 border border-blue-500/30'
            }`}>
              {rebuildStatus()}
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
      <div class={`${themeClasses.bgTertiary()} border-t ${themeClasses.border()} px-4 md:px-6 py-3 backdrop-blur-sm`}>
        <div class={`flex items-center justify-between text-xs ${themeClasses.textMuted()}`}>
          <div class="flex items-center space-x-2 md:space-x-4">
            <div class="flex items-center space-x-2">
              <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>Ready</span>
            </div>
            <Show when={searchMetadata()}>
              <span class="hidden sm:inline">•</span>
              <span class={`${themeClasses.textSecondary()} text-xs`}>
                {searchMetadata()!.totalResults} results in {searchMetadata()!.searchTime.toFixed(2)}s
              </span>
            </Show>
          </div>
          <div class="flex items-center space-x-2 md:space-x-4">
            <span class="hidden md:inline">Index: 23,456 documents</span>
            <span class="hidden md:inline">•</span>
            <span class="hidden sm:inline">Last updated: 2 hours ago</span>
            <span class="sm:hidden">2h ago</span>
          </div>
        </div>
      </div>
      
      {/* Keyboard Shortcuts Modal */}
      <KeyboardShortcutsModal />
    </div>
  );
};

export default App;