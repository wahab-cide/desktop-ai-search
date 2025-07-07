import { Component, For, Show, createSignal } from 'solid-js'
import { searchResults, isSearching, searchMetadata } from '../stores/searchStore'
import type { SearchResult } from '../types/api'
import { theme, themeClasses } from '../stores/themeStore'

const FileTypeIcon: Component<{ fileType: string; filePath?: string }> = (props) => {
  const iconMap: Record<string, { icon: string; color: string; bgColor: string }> = {
    pdf: { icon: '📄', color: 'text-red-400', bgColor: 'bg-red-500/10' },
    docx: { icon: '📝', color: 'text-blue-400', bgColor: 'bg-blue-500/10' },
    doc: { icon: '📝', color: 'text-blue-400', bgColor: 'bg-blue-500/10' },
    txt: { icon: '📃', color: 'text-zinc-400', bgColor: 'bg-zinc-500/10' },
    md: { icon: '📋', color: 'text-purple-400', bgColor: 'bg-purple-500/10' },
    jpg: { icon: '🖼️', color: 'text-green-400', bgColor: 'bg-green-500/10' },
    jpeg: { icon: '🖼️', color: 'text-green-400', bgColor: 'bg-green-500/10' },
    png: { icon: '🖼️', color: 'text-green-400', bgColor: 'bg-green-500/10' },
    gif: { icon: '🎞️', color: 'text-pink-400', bgColor: 'bg-pink-500/10' },
    mp3: { icon: '🎵', color: 'text-orange-400', bgColor: 'bg-orange-500/10' },
    mp4: { icon: '🎬', color: 'text-indigo-400', bgColor: 'bg-indigo-500/10' },
    email: { icon: '📧', color: 'text-blue-400', bgColor: 'bg-blue-500/10' },
    default: { icon: '📄', color: 'text-zinc-400', bgColor: 'bg-zinc-500/10' }
  }
  
  const typeInfo = iconMap[props.fileType] || iconMap.default
  
  return (
    <div class={`w-12 h-12 rounded-xl ${typeInfo.bgColor} border ${themeClasses.border()} flex items-center justify-center shadow-sm`}>
      <span class="text-xl">{typeInfo.icon}</span>
    </div>
  )
}

const ResultCard: Component<{ result: SearchResult }> = (props) => {
  const [isExpanded, setIsExpanded] = createSignal(false)
  const [showPreview, setShowPreview] = createSignal(false)
  
  const formatDate = (date: string) => {
    return new Date(date).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    })
  }
  
  const getFileType = (path: string) => {
    const ext = path.split('.').pop()?.toLowerCase() || 'default'
    return ext
  }
  
  const getFileName = (path: string) => {
    return path.split('/').pop() || path
  }
  
  const isImageFile = (fileType: string) => {
    return ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg'].includes(fileType.toLowerCase())
  }
  
  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
  }
  
  const getRelativeTime = (date: string) => {
    const now = new Date()
    const past = new Date(date)
    const diffInSeconds = Math.floor((now.getTime() - past.getTime()) / 1000)
    
    if (diffInSeconds < 60) return 'just now'
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`
    if (diffInSeconds < 2592000) return `${Math.floor(diffInSeconds / 86400)}d ago`
    return formatDate(date)
  }
  
  const fileType = getFileType(props.result.file_path)
  
  return (
    <div class={`group ${themeClasses.card()} ${themeClasses.cardHover()} rounded-xl p-5 hover:shadow-2xl transition-all duration-300 relative overflow-hidden backdrop-blur-sm`}>
      {/* Background gradient on hover */}
      <div class="absolute inset-0 bg-gradient-to-r from-purple-500/0 via-purple-500/5 to-blue-500/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      
      <div class="relative flex items-start space-x-4">
        <div class="flex-shrink-0">
          <FileTypeIcon fileType={fileType} filePath={props.result.file_path} />
        </div>
        
        <div class="flex-1 min-w-0">
          <div class="flex items-start justify-between mb-3">
            <div class="flex-1 min-w-0">
              <h3 class={`text-lg font-semibold ${themeClasses.text()} truncate pr-2 group-hover:text-purple-300 transition-colors`}>
                {getFileName(props.result.file_path)}
              </h3>
              <div class={`flex items-center space-x-2 text-sm ${themeClasses.textMuted()} mt-1`}>
                <span class="truncate max-w-xs">{props.result.file_path}</span>
                <Show when={props.result.modified_date}>
                  <span>•</span>
                  <span class={`whitespace-nowrap ${themeClasses.textSubtle()}`}>{getRelativeTime(props.result.modified_date!)}</span>
                </Show>
                <Show when={props.result.file_size}>
                  <span>•</span>
                  <span class={`whitespace-nowrap ${themeClasses.textSubtle()}`}>{formatFileSize(props.result.file_size!)}</span>
                </Show>
              </div>
            </div>
            
            <div class="flex items-center space-x-2 ml-4">
              <Show when={props.result.match_type}>
                <span class={`text-xs px-3 py-1.5 rounded-full font-medium ${
                  props.result.match_type === 'exact' 
                    ? 'bg-green-500/20 text-green-300 border border-green-500/30'
                    : props.result.match_type === 'semantic'
                    ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                    : 'bg-zinc-500/20 text-zinc-400 border border-zinc-500/30'
                }`}>
                  {props.result.match_type === 'exact' && '🎯 '}
                  {props.result.match_type === 'semantic' && '🧠 '}
                  {props.result.match_type?.charAt(0).toUpperCase() + props.result.match_type?.slice(1)}
                </span>
              </Show>
            </div>
          </div>
          
          {/* Content Preview */}
          <div class="mb-4">
            <div class={`rounded-lg p-4 border ${theme() === 'light' ? 'bg-gray-50/80 border-gray-200' : 'bg-black/50 border-zinc-800/50'}`}>
              <div 
                class={`${themeClasses.textSecondary()} text-sm leading-relaxed ${isExpanded() ? '' : 'line-clamp-3'}`}
                innerHTML={props.result.highlighted_content || props.result.content}
              />
              
              <Show when={props.result.content.length > 200}>
                <button
                  class="text-sm text-purple-500 hover:text-purple-400 mt-3 font-medium transition-colors"
                  onClick={() => setIsExpanded(!isExpanded())}
                >
                  {isExpanded() ? 'Show less' : 'Show more'}
                </button>
              </Show>
            </div>
          </div>
          
          {/* Metadata and Actions */}
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-4">
              {/* Relevance Score */}
              <div class="flex items-center space-x-2">
                <span class={`text-xs ${themeClasses.textMuted()}`}>Relevance:</span>
                <div class="flex items-center space-x-2">
                  <div class={`w-20 rounded-full h-2 ${theme() === 'light' ? 'bg-gray-200' : 'bg-zinc-800'}`}>
                    <div
                      class={`h-2 rounded-full transition-all duration-500 ${
                        props.result.score > 0.8 ? 'bg-green-400' :
                        props.result.score > 0.6 ? 'bg-blue-400' :
                        props.result.score > 0.4 ? 'bg-yellow-400' : 'bg-zinc-500'
                      }`}
                      style={{ width: `${Math.min(props.result.score * 100, 100)}%` }}
                    />
                  </div>
                  <span class={`text-xs font-medium ${themeClasses.textSecondary()}`}>
                    {(props.result.score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              
              {/* Additional metadata for images */}
              <Show when={isImageFile(fileType)}>
                <div class={`flex items-center space-x-1 text-xs ${themeClasses.textMuted()}`}>
                  <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <span>Image</span>
                </div>
              </Show>
            </div>
            
            {/* Action Buttons */}
            <div class="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-all duration-200">
              <button 
                class={`p-2 ${themeClasses.textMuted()} hover:text-blue-500 hover:bg-blue-500/10 rounded-lg transition-all duration-200`}
                title="Open file"
              >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </button>
              <button 
                class={`p-2 ${themeClasses.textMuted()} hover:text-green-500 hover:bg-green-500/10 rounded-lg transition-all duration-200`}
                title="Show in folder"
              >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
                </svg>
              </button>
              <Show when={isImageFile(fileType)}>
                <button 
                  class={`p-2 ${themeClasses.textMuted()} hover:text-purple-500 hover:bg-purple-500/10 rounded-lg transition-all duration-200`}
                  title="Quick preview"
                  onClick={() => setShowPreview(!showPreview())}
                >
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                </button>
              </Show>
            </div>
          </div>
        </div>
      </div>
      
      {/* Image Preview Modal */}
      <Show when={showPreview() && isImageFile(fileType)}>
        <div class={`fixed inset-0 backdrop-blur-sm flex items-center justify-center z-50 ${theme() === 'light' ? 'bg-white/90' : 'bg-black/90'}`}
             onClick={() => setShowPreview(false)}>
          <div class={`${themeClasses.card()} rounded-xl p-6 max-w-4xl max-h-[85vh] overflow-auto shadow-2xl`}>
            <div class="flex items-center justify-between mb-6">
              <h3 class={`text-lg font-semibold ${themeClasses.text()}`}>{getFileName(props.result.file_path)}</h3>
              <button 
                onClick={() => setShowPreview(false)} 
                class="text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 rounded-lg p-2 transition-colors"
              >
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <img 
              src={`file://${props.result.file_path}`} 
              alt={getFileName(props.result.file_path)}
              class="max-w-full max-h-96 object-contain mx-auto rounded-lg"
            />
          </div>
        </div>
      </Show>
    </div>
  )
}

export const SearchResults: Component = () => {
  return (
    <div class={`flex-1 overflow-y-auto ${themeClasses.bg()}`}>
      <Show
        when={!isSearching()}
        fallback={
          <div class="flex items-center justify-center h-64">
            <div class="text-center">
              <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto"></div>
              <p class={`mt-4 ${themeClasses.textMuted()}`}>Searching your documents...</p>
              <p class={`text-sm ${themeClasses.textSubtle()} mt-1`}>Using AI to find the best matches</p>
            </div>
          </div>
        }
      >
        <Show
          when={searchResults().length > 0}
          fallback={
            <div class="flex items-center justify-center h-64">
              <div class="text-center">
                <div class={`w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4 ${theme() === 'light' ? 'bg-gray-100' : 'bg-zinc-900'}`}>
                  <svg class={`w-8 h-8 ${themeClasses.textMuted()}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <p class={`mt-4 ${themeClasses.textMuted()} text-lg font-medium`}>No results found</p>
                <p class={`text-sm ${themeClasses.textSubtle()} mt-2`}>Try adjusting your search query or filters</p>
                <div class={`mt-4 space-y-2 text-xs ${themeClasses.textSubtle()}`}>
                  <p>• Use quotes for exact phrases: "machine learning"</p>
                  <p>• Try semantic search: "documents about AI"</p>
                  <p>• Filter by file type or date range</p>
                </div>
              </div>
            </div>
          }
        >
          <div class="p-6">
            <div class="mb-6 flex items-center justify-between">
              <div class="flex items-center space-x-4">
                <p class={`text-sm ${themeClasses.textMuted()}`}>
                  Found <span class={`font-semibold ${themeClasses.textSecondary()}`}>{searchResults().length}</span> results
                  <Show when={searchMetadata()?.searchTime}>
                    <span class={themeClasses.textSubtle()}> in {searchMetadata()!.searchTime.toFixed(2)}s</span>
                  </Show>
                </p>
                <Show when={searchMetadata()?.queryIntent}>
                  <span class="text-xs px-2 py-1 bg-purple-500/10 text-purple-300 rounded-full border border-purple-500/20">
                    AI Enhanced
                  </span>
                </Show>
              </div>
              
              <div class="flex items-center space-x-3">
                <select class={`text-sm border rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-purple-500 focus:border-purple-500 ${theme() === 'light' ? 'border-gray-300 bg-white text-gray-900' : 'border-zinc-700 bg-zinc-900 text-zinc-300'}`}>
                  <option>Relevance</option>
                  <option>Date (Newest)</option>
                  <option>Date (Oldest)</option>
                  <option>File Size</option>
                </select>
              </div>
            </div>
            
            <div class="space-y-4">
              <For each={searchResults()}>
                {(result) => <ResultCard result={result} />}
              </For>
            </div>
          </div>
        </Show>
      </Show>
    </div>
  )
}