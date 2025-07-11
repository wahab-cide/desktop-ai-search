import { Component, For, Show } from 'solid-js'
import { 
  showShortcutsModal, 
  setShowShortcutsModal, 
  getShortcutsByCategory, 
  formatShortcutKey 
} from '../stores/keyboardStore'
import { effectiveTheme, themeClasses } from '../stores/themeStore'

export const KeyboardShortcutsModal: Component = () => {
  const shortcuts = getShortcutsByCategory()
  
  return (
    <Show when={showShortcutsModal()}>
      <div 
        class={`fixed inset-0 backdrop-blur-sm flex items-center justify-center z-[9999] ${
          effectiveTheme() === 'light' ? 'bg-black/20' : 'bg-black/50'
        }`}
        onClick={() => setShowShortcutsModal(false)}
      >
        <div 
          class={`${themeClasses.card()} rounded-2xl p-8 max-w-4xl max-h-[80vh] overflow-auto shadow-2xl mx-4 animate-fadeIn`}
          onClick={e => e.stopPropagation()}
        >
          {/* Header */}
          <div class="flex items-center justify-between mb-6">
            <div>
              <h2 class={`text-2xl font-bold ${themeClasses.text()}`}>Keyboard Shortcuts</h2>
              <p class={`text-sm ${themeClasses.textMuted()} mt-1`}>
                Boost your productivity with these shortcuts
              </p>
            </div>
            <button 
              onClick={() => setShowShortcutsModal(false)}
              class={`p-2 ${themeClasses.textMuted()} hover:${themeClasses.textSecondary()} ${themeClasses.hover()} rounded-lg transition-colors`}
              title="Close (Esc)"
            >
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          {/* Shortcuts Grid */}
          <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
            <For each={Object.entries(shortcuts)}>
              {([category, categoryShortcuts]) => (
                <div>
                  <h3 class={`text-lg font-semibold ${themeClasses.textSecondary()} mb-4 flex items-center`}>
                    <div class={`w-2 h-2 rounded-full bg-purple-500 mr-3`}></div>
                    {category}
                  </h3>
                  <div class="space-y-3">
                    <For each={categoryShortcuts}>
                      {(shortcut) => (
                        <div class={`flex items-center justify-between p-3 rounded-lg ${themeClasses.hover()} transition-colors`}>
                          <span class={`text-sm ${themeClasses.text()}`}>
                            {shortcut.description}
                          </span>
                          <div class={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium ${
                            effectiveTheme() === 'light' 
                              ? 'bg-gray-100 text-gray-700 border border-gray-300' 
                              : 'bg-zinc-800 text-zinc-300 border border-zinc-700'
                          }`}>
                            {formatShortcutKey(shortcut)}
                          </div>
                        </div>
                      )}
                    </For>
                  </div>
                </div>
              )}
            </For>
          </div>
          
          {/* Footer */}
          <div class={`mt-8 pt-6 border-t ${themeClasses.border()}`}>
            <div class="flex items-center justify-between">
              <div class={`text-xs ${themeClasses.textMuted()}`}>
                💡 Tip: Press <kbd class={`px-2 py-1 rounded ${effectiveTheme() === 'light' ? 'bg-gray-100 text-gray-700' : 'bg-zinc-800 text-zinc-300'}`}>
                  {navigator.platform.includes('Mac') ? '⌘' : 'Ctrl'} + ?
                </kbd> to toggle this dialog
              </div>
              <button 
                onClick={() => setShowShortcutsModal(false)}
                class={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  effectiveTheme() === 'light' 
                    ? 'bg-gray-100 text-gray-700 hover:bg-gray-200' 
                    : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
                }`}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </Show>
  )
}