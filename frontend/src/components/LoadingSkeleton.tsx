import { Component } from 'solid-js'
import { effectiveTheme, themeClasses } from '../stores/themeStore'

interface LoadingSkeletonProps {
  type?: 'search-result' | 'search-bar' | 'card' | 'text' | 'avatar'
  count?: number
  className?: string
}

export const LoadingSkeleton: Component<LoadingSkeletonProps> = (props) => {
  const { type = 'text', count = 1, className = '' } = props

  const baseClasses = `animate-shimmer ${effectiveTheme() === 'light' ? 'bg-gray-200' : 'bg-zinc-800'}`

  const renderSkeleton = () => {
    switch (type) {
      case 'search-result':
        return (
          <div class={`${themeClasses.card()} rounded-xl p-5 ${className}`}>
            <div class="flex items-start space-x-4">
              {/* File icon skeleton */}
              <div class={`w-12 h-12 rounded-xl ${baseClasses}`} />
              
              <div class="flex-1 space-y-3">
                {/* Title skeleton */}
                <div class={`h-6 ${baseClasses} rounded-lg`} style={{ width: '60%' }} />
                
                {/* Path skeleton */}
                <div class={`h-4 ${baseClasses} rounded`} style={{ width: '80%' }} />
                
                {/* Content skeleton */}
                <div class="space-y-2">
                  <div class={`h-4 ${baseClasses} rounded`} style={{ width: '100%' }} />
                  <div class={`h-4 ${baseClasses} rounded`} style={{ width: '75%' }} />
                  <div class={`h-4 ${baseClasses} rounded`} style={{ width: '50%' }} />
                </div>
                
                {/* Metadata skeleton */}
                <div class="flex items-center justify-between pt-2">
                  <div class="flex items-center space-x-4">
                    <div class={`h-3 w-20 ${baseClasses} rounded`} />
                    <div class={`h-2 w-16 ${baseClasses} rounded-full`} />
                  </div>
                  <div class="flex space-x-2">
                    <div class={`h-8 w-8 ${baseClasses} rounded-lg`} />
                    <div class={`h-8 w-8 ${baseClasses} rounded-lg`} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )
      
      case 'search-bar':
        return (
          <div class={`${themeClasses.input()} rounded-2xl p-4 ${className}`}>
            <div class="flex items-center space-x-3">
              <div class={`w-5 h-5 ${baseClasses} rounded-full`} />
              <div class={`flex-1 h-6 ${baseClasses} rounded-lg`} />
              <div class={`w-20 h-6 ${baseClasses} rounded-lg`} />
            </div>
          </div>
        )
      
      case 'card':
        return (
          <div class={`${themeClasses.card()} rounded-xl p-6 ${className}`}>
            <div class="space-y-4">
              <div class={`h-6 ${baseClasses} rounded-lg`} style={{ width: '70%' }} />
              <div class="space-y-2">
                <div class={`h-4 ${baseClasses} rounded`} style={{ width: '100%' }} />
                <div class={`h-4 ${baseClasses} rounded`} style={{ width: '80%' }} />
                <div class={`h-4 ${baseClasses} rounded`} style={{ width: '60%' }} />
              </div>
            </div>
          </div>
        )
      
      case 'avatar':
        return (
          <div class={`w-8 h-8 ${baseClasses} rounded-full ${className}`} />
        )
      
      case 'text':
      default:
        return (
          <div class={`h-4 ${baseClasses} rounded ${className}`} style={{ width: '100%' }} />
        )
    }
  }

  return (
    <div class="space-y-4">
      {Array.from({ length: count }, (_, i) => (
        <div key={i} class="animate-slideIn" style={{ 'animation-delay': `${i * 0.1}s` }}>
          {renderSkeleton()}
        </div>
      ))}
    </div>
  )
}

export const SearchResultsSkeleton: Component = () => {
  return (
    <div class="p-6 space-y-6">
      {/* Header skeleton */}
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-4">
          <div class={`h-4 w-32 ${effectiveTheme() === 'light' ? 'bg-gray-200' : 'bg-zinc-800'} animate-shimmer rounded`} />
          <div class={`h-6 w-20 ${effectiveTheme() === 'light' ? 'bg-gray-200' : 'bg-zinc-800'} animate-shimmer rounded-full`} />
        </div>
        <div class={`h-8 w-32 ${effectiveTheme() === 'light' ? 'bg-gray-200' : 'bg-zinc-800'} animate-shimmer rounded-lg`} />
      </div>
      
      {/* Results skeleton */}
      <LoadingSkeleton type="search-result" count={5} />
    </div>
  )
}