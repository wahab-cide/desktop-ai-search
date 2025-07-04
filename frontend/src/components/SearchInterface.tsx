import { Component, For, Show } from 'solid-js'
import { searchQuery, setSearchQuery, searchResults, isSearching } from '../stores/searchStore'

export const SearchInterface: Component = () => {
  return (
    <div class="flex flex-col h-full">
      <div class="p-4 border-b">
        <input
          type="text"
          placeholder="Search your documents..."
          class="w-full h-12 p-3 border rounded-lg"
          value={searchQuery()}
          onInput={(e) => setSearchQuery(e.currentTarget.value)}
        />
      </div>
      
      <div class="flex-1 overflow-y-auto p-4">
        <Show when={isSearching()}>
          <div class="text-center py-8">
            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
            <p class="mt-2 text-gray-600">Searching...</p>
          </div>
        </Show>
        
        <For each={searchResults()}>
          {(result) => (
            <div class="mb-4 p-4 border rounded-lg hover:bg-gray-50">
              <h3 class="font-medium text-gray-900">{result.file_path}</h3>
              <p class="text-sm text-gray-600 mt-1">{result.content}</p>
              <p class="text-xs text-gray-400 mt-2">Score: {result.score.toFixed(2)}</p>
            </div>
          )}
        </For>
      </div>
    </div>
  )
}