import { Component, Show, createSignal, onMount } from 'solid-js'
import { invoke } from '@tauri-apps/api/core'
import { mockSearchAPI, isDevelopment } from '../mock-api'
import { searchAPI } from '../api'
import { theme, themeClasses } from '../stores/themeStore'
import type { IndexingStatus, IndexingStatistics } from '../types/api'

// Use mock API in development, real API in production  
// Set to false to test real backend even in development
const USE_MOCK = false
const api = (isDevelopment && USE_MOCK) ? mockSearchAPI : searchAPI

export const IndexingPanel: Component = () => {
  const [indexingStatus, setIndexingStatus] = createSignal<IndexingStatus | null>(null)
  const [indexingStats, setIndexingStats] = createSignal<IndexingStatistics | null>(null)
  const [isIndexing, setIsIndexing] = createSignal(false)
  const [selectedPath, setSelectedPath] = createSignal('')
  const [lastError, setLastError] = createSignal<string | null>(null)

  // Quick path selection
  const selectTestDocuments = () => {
    setSelectedPath('/Users/outhills/Desktop/desktop-ai-search/test_documents')
  }

  const selectDocuments = () => {
    setSelectedPath('/Users/outhills/Documents')
  }

  const selectDesktop = () => {
    setSelectedPath('/Users/outhills/Desktop')
  }

  // Rebuild FTS index
  const rebuildFtsIndex = async () => {
    try {
      const result = await api.rebuildSearchIndex()
      console.log('FTS rebuild result:', result)
      alert(result)
    } catch (error) {
      console.error('FTS rebuild error:', error)
      setLastError(`FTS rebuild failed: ${error}`)
    }
  }

  // Cleanup missing files
  const cleanupMissingFiles = async () => {
    const confirmed = confirm('Remove all database entries for files that no longer exist on the filesystem?\n\nThis will:\n- Check all indexed files\n- Remove entries for missing files\n- Rebuild the search index\n\nThis action cannot be undone.')
    
    if (!confirmed) return

    try {
      setIsIndexing(true)
      const result = await api.cleanupMissingFiles()
      
      const message = `Cleanup completed:\n\n` +
        `• Checked ${result.checked_files} files\n` +
        `• Removed ${result.removed_documents} missing documents\n` +
        `• Removed ${result.removed_chunks} orphaned chunks\n\n` +
        `Search index has been rebuilt.`
      
      alert(message)
      await checkIndexingStatus() // Refresh status
    } catch (error) {
      console.error('Cleanup error:', error)
      setLastError(`Cleanup failed: ${error}`)
      alert(`Cleanup failed: ${error}`)
    } finally {
      setIsIndexing(false)
    }
  }

  // Reset entire database
  const resetDatabase = async () => {
    console.log('🔥 Reset database button clicked!')
    
    const confirmed = confirm('⚠️ DANGER: Reset entire database?\n\nThis will:\n- Delete ALL indexed documents\n- Delete ALL text chunks\n- Delete ALL search indices\n- Clear ALL cached data\n\nYou will need to re-index all your files.\n\nThis action CANNOT be undone!')
    
    if (!confirmed) {
      console.log('🔥 Reset cancelled by user (first confirm)')
      return
    }

    const doubleConfirm = confirm('Are you absolutely sure?\n\nClick OK to proceed.')
    if (!doubleConfirm) {
      console.log('🔥 Reset cancelled by user (second confirm)')
      return
    }

    console.log('🔥 User confirmed reset, proceeding...')
    try {
      setIsIndexing(true)
      console.log('🔥 Calling api.resetDatabase()...')
      const result = await api.resetDatabase()
      console.log('🔥 Reset result:', result)
      
      const message = `Database reset completed:\n\n` +
        `• Removed ${result.removed_documents} documents\n` +
        `• Removed ${result.removed_chunks} chunks\n\n` +
        `You can now re-index your files.`
      
      alert(message)
      await checkIndexingStatus() // Refresh status
    } catch (error) {
      console.error('🔥 Reset error:', error)
      setLastError(`Reset failed: ${error}`)
      alert(`Reset failed: ${error}`)
    } finally {
      setIsIndexing(false)
    }
  }

  // Poll for indexing status
  const checkIndexingStatus = async () => {
    try {
      const status = await api.getIndexingStatus()
      setIndexingStatus(status)
      
      const stats = await api.getIndexingStatistics()
      setIndexingStats(stats)
    } catch (error) {
      console.error('Failed to get indexing status:', error)
    }
  }

  // Test parameter passing
  const testParams = async () => {
    if (!selectedPath()) return
    
    let results = []
    
    // Test 1: camelCase parameter  
    try {
      console.log('🧪 Test 1: camelCase parameter')
      const result1 = await invoke<string>('test_indexing_params', { directoryPath: selectedPath() })
      results.push(`✅ test_indexing_params: ${result1}`)
    } catch (error) {
      results.push(`❌ test_indexing_params: ${error instanceof Error ? error.message : String(error)}`)
    }
    
    // Test 2: camelCase parameter with camelCase function
    try {
      console.log('🧪 Test 2: camelCase parameter')
      const result2 = await invoke<string>('test_camel_case', { directoryPath: selectedPath() })
      results.push(`✅ camelCase: ${result2}`)
    } catch (error) {
      results.push(`❌ camelCase: ${error instanceof Error ? error.message : String(error)}`)
    }
    
    // Test 3: Simple indexing without database state
    try {
      console.log('🧪 Test 3: simple indexing')
      const result3 = await invoke<string>('index_directory_simple', { directoryPath: selectedPath() })
      results.push(`✅ simple index: ${result3}`)
    } catch (error) {
      results.push(`❌ simple index: ${error instanceof Error ? error.message : String(error)}`)
    }
    
    setLastError(results.join('\n\n'))
  }

  // Start directory indexing
  const handleIndexDirectory = async () => {
    if (!selectedPath()) return
    
    setLastError(null)
    setIsIndexing(true)
    
    try {
      console.log('🔄 Starting indexing for:', selectedPath())
      await api.indexDirectory(selectedPath(), false)
      console.log('✅ Indexing completed successfully')
      await checkIndexingStatus()
    } catch (error) {
      console.error('❌ Failed to start indexing:', error)
      setLastError(error instanceof Error ? error.message : String(error))
    } finally {
      setIsIndexing(false)
    }
  }

  // Start background indexing
  const handleStartBackgroundIndexing = async () => {
    setIsIndexing(true)
    try {
      await api.startBackgroundIndexing()
      await checkIndexingStatus()
    } catch (error) {
      console.error('Failed to start background indexing:', error)
    } finally {
      setIsIndexing(false)
    }
  }

  // Reset indexing state
  const handleResetIndexing = async () => {
    try {
      await api.resetIndexingState()
      await checkIndexingStatus()
    } catch (error) {
      console.error('Failed to reset indexing:', error)
    }
  }

  onMount(() => {
    checkIndexingStatus()
    // Poll every 2 seconds for updates
    const interval = setInterval(checkIndexingStatus, 2000)
    return () => clearInterval(interval)
  })

  return (
    <div class={`${themeClasses.card()} rounded-xl p-6 space-y-6`}>
      <div class="flex items-center justify-between">
        <h2 class={`text-xl font-semibold ${themeClasses.text()}`}>Document Indexing</h2>
        <div class="flex items-center space-x-2">
          <div class={`w-3 h-3 rounded-full ${indexingStatus() ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`}></div>
          <span class={`text-sm ${themeClasses.textMuted()}`}>
            {indexingStatus() ? 'Indexing Active' : 'Ready'}
          </span>
        </div>
      </div>

      {/* Indexing Statistics */}
      <Show when={indexingStats()}>
        <div class={`rounded-lg p-4 ${theme() === 'light' ? 'bg-gray-50 border border-gray-200' : 'bg-zinc-800/50 border border-zinc-700/50'}`}>
          <h3 class={`text-sm font-medium ${themeClasses.textSecondary()} mb-3`}>Statistics</h3>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div class={`text-2xl font-bold ${themeClasses.text()}`}>
                {indexingStats()?.indexed_files?.toLocaleString() || 0}
              </div>
              <div class={`text-xs ${themeClasses.textMuted()}`}>Files Indexed</div>
            </div>
            <div>
              <div class={`text-2xl font-bold ${themeClasses.text()}`}>
                {indexingStats()?.total_files?.toLocaleString() || 0}
              </div>
              <div class={`text-xs ${themeClasses.textMuted()}`}>Total Files</div>
            </div>
            <div>
              <div class={`text-2xl font-bold ${themeClasses.text()}`}>
                {indexingStats()?.failed_files?.toLocaleString() || 0}
              </div>
              <div class={`text-xs ${themeClasses.textMuted()}`}>Failed</div>
            </div>
            <div>
              <div class={`text-2xl font-bold ${themeClasses.text()}`}>
                {indexingStats()?.files_per_second?.toFixed(1) || '0'}/s
              </div>
              <div class={`text-xs ${themeClasses.textMuted()}`}>Files/sec</div>
            </div>
            <Show when={indexingStats()?.documents_with_ocr || indexingStats()?.pending_ocr}>
              <div>
                <div class={`text-2xl font-bold ${themeClasses.text()}`}>
                  {indexingStats()?.documents_with_ocr || 0}
                  <Show when={indexingStats()?.pending_ocr}>
                    <span class={`text-sm ${themeClasses.textMuted()}`}>/{indexingStats()?.pending_ocr}</span>
                  </Show>
                </div>
                <div class={`text-xs ${themeClasses.textMuted()}`}>OCR Docs</div>
              </div>
            </Show>
            <Show when={indexingStats()?.documents_with_transcription || indexingStats()?.pending_transcription}>
              <div>
                <div class={`text-2xl font-bold ${themeClasses.text()}`}>
                  {indexingStats()?.documents_with_transcription || 0}
                  <Show when={indexingStats()?.pending_transcription}>
                    <span class={`text-sm ${themeClasses.textMuted()}`}>/{indexingStats()?.pending_transcription}</span>
                  </Show>
                </div>
                <div class={`text-xs ${themeClasses.textMuted()}`}>Audio Docs</div>
              </div>
            </Show>
          </div>
        </div>
      </Show>

      {/* Current Indexing Status */}
      <Show when={indexingStatus()}>
        <div class={`rounded-lg p-4 ${theme() === 'light' ? 'bg-blue-50 border border-blue-200' : 'bg-blue-900/20 border border-blue-700/50'}`}>
          <div class="flex items-center justify-between mb-2">
            <span class={`text-sm font-medium ${theme() === 'light' ? 'text-blue-700' : 'text-blue-300'}`}>
              Currently Indexing
            </span>
            <span class={`text-xs ${theme() === 'light' ? 'text-blue-600' : 'text-blue-400'}`}>
              {indexingStatus()?.indexed} / {indexingStatus()?.total}
            </span>
          </div>
          <div class={`w-full rounded-full h-2 ${theme() === 'light' ? 'bg-blue-200' : 'bg-blue-800'}`}>
            <div 
              class="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{
                width: `${((indexingStatus()?.indexed || 0) / (indexingStatus()?.total || 1)) * 100}%`
              }}
            ></div>
          </div>
          <Show when={indexingStatus()?.current_file}>
            <div class={`text-xs ${themeClasses.textMuted()} mt-2 truncate`}>
              {indexingStatus()?.current_file}
            </div>
          </Show>
          <Show when={indexingStatus()?.multimedia_processing_status}>
            <div class={`text-xs ${theme() === 'light' ? 'text-orange-600' : 'text-orange-400'} mt-2 flex items-center`}>
              <svg class="w-4 h-4 mr-1 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              {indexingStatus()?.multimedia_processing_status}
            </div>
          </Show>
        </div>
      </Show>

      {/* Error Display */}
      <Show when={lastError()}>
        <div class={`rounded-lg p-4 border-2 border-red-500 ${theme() === 'light' ? 'bg-red-50 text-red-800' : 'bg-red-900/20 text-red-300'}`}>
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-2">
              <span class="text-red-500">❌</span>
              <span class="font-medium">Indexing Error</span>
            </div>
            <button
              class={`text-red-500 hover:text-red-600 ${themeClasses.hover()}`}
              onClick={() => setLastError(null)}
            >
              ✕
            </button>
          </div>
          <div class="mt-2 text-sm font-mono">
            {lastError()}
          </div>
        </div>
      </Show>

      {/* Indexing Controls */}
      <div class="space-y-4">
        <div>
          <label class={`block text-sm font-medium ${themeClasses.textSecondary()} mb-2`}>
            Directory Path
          </label>
          <input
            type="text"
            placeholder="/path/to/your/documents"
            class={`w-full px-3 py-2 rounded-lg border focus:outline-none focus:ring-2 focus:ring-purple-500 ${themeClasses.input()}`}
            value={selectedPath()}
            onInput={(e) => setSelectedPath(e.currentTarget.value)}
          />
          
          {/* Quick Path Selection */}
          <div class="mt-2 flex flex-wrap gap-2">
            <button
              class={`px-3 py-1 text-xs rounded-md transition-colors ${theme() === 'light' ? 'bg-blue-100 text-blue-700 hover:bg-blue-200' : 'bg-blue-900/30 text-blue-300 hover:bg-blue-800/50'}`}
              onClick={selectTestDocuments}
            >
              📁 Test Documents
            </button>
            <button
              class={`px-3 py-1 text-xs rounded-md transition-colors ${theme() === 'light' ? 'bg-green-100 text-green-700 hover:bg-green-200' : 'bg-green-900/30 text-green-300 hover:bg-green-800/50'}`}
              onClick={selectDocuments}
            >
              📄 Documents Folder
            </button>
            <button
              class={`px-3 py-1 text-xs rounded-md transition-colors ${theme() === 'light' ? 'bg-purple-100 text-purple-700 hover:bg-purple-200' : 'bg-purple-900/30 text-purple-300 hover:bg-purple-800/50'}`}
              onClick={selectDesktop}
            >
              🖥️ Desktop
            </button>
          </div>
        </div>

        <div class="flex flex-wrap gap-3">
          <button
            class={`px-4 py-2 rounded-lg font-medium transition-colors ${
              isIndexing() 
                ? 'bg-gray-400 text-gray-600 cursor-not-allowed' 
                : 'bg-purple-500 hover:bg-purple-600 text-white'
            }`}
            onClick={handleIndexDirectory}
            disabled={isIndexing() || !selectedPath()}
          >
            {isIndexing() ? 'Indexing...' : 'Index Directory'}
          </button>

          <button
            class={`px-4 py-2 rounded-lg font-medium transition-colors ${
              isIndexing() 
                ? 'bg-gray-400 text-gray-600 cursor-not-allowed' 
                : `border-2 border-purple-500 text-purple-500 hover:bg-purple-500 hover:text-white ${themeClasses.hover()}`
            }`}
            onClick={handleStartBackgroundIndexing}
            disabled={isIndexing()}
          >
            Start Background Indexing
          </button>

          <button
            class={`px-4 py-2 rounded-lg font-medium transition-colors border-2 border-red-500 text-red-500 hover:bg-red-500 hover:text-white ${themeClasses.hover()}`}
            onClick={handleResetIndexing}
          >
            Reset Index
          </button>

          <button
            class={`px-3 py-2 text-sm rounded-lg font-medium transition-colors border-2 border-blue-500 text-blue-500 hover:bg-blue-500 hover:text-white ${themeClasses.hover()}`}
            onClick={testParams}
            disabled={isIndexing() || !selectedPath()}
          >
            🧪 Test Connection
          </button>

          <button
            class={`px-3 py-2 text-sm rounded-lg font-medium transition-colors border-2 border-yellow-500 text-yellow-500 hover:bg-yellow-500 hover:text-white ${themeClasses.hover()}`}
            onClick={rebuildFtsIndex}
          >
            🔧 Rebuild Search Index
          </button>

          <button
            class={`px-3 py-2 text-sm rounded-lg font-medium transition-colors border-2 border-orange-500 text-orange-500 hover:bg-orange-500 hover:text-white ${themeClasses.hover()}`}
            onClick={cleanupMissingFiles}
            disabled={isIndexing()}
          >
            🧹 Cleanup Missing Files
          </button>

          <button
            class={`px-3 py-2 text-sm rounded-lg font-medium transition-colors border-2 border-red-600 text-red-600 hover:bg-red-600 hover:text-white ${themeClasses.hover()}`}
            onClick={(e) => {
              console.log('🔥 Reset Database button click event fired!', e)
              e.preventDefault()
              e.stopPropagation()
              resetDatabase()
            }}
            disabled={isIndexing()}
          >
            🔥 Reset Database
          </button>
        </div>
      </div>

      {/* Instructions */}
      <div class={`rounded-lg p-4 ${theme() === 'light' ? 'bg-gray-50 border border-gray-200' : 'bg-zinc-800/30 border border-zinc-700/30'}`}>
        <h4 class={`text-sm font-medium ${themeClasses.textSecondary()} mb-2`}>Quick Start</h4>
        <ul class={`text-sm ${themeClasses.textMuted()} space-y-1`}>
          <li>• Enter a directory path to index your documents</li>
          <li>• Use "Index Directory" for a one-time scan</li>
          <li>• Use "Background Indexing" for continuous monitoring</li>
          <li>• Reset Index to clear all data and start fresh</li>
        </ul>
      </div>
    </div>
  )
}