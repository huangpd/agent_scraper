import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChatMessage, WsEvent } from '../types'

let msgId = 0
const PROGRESS_ID_PREFIX = '__progress__'

export function useWebSocket() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  const connect = useCallback((taskId: string) => {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${location.host}/ws/${taskId}`)
    wsRef.current = ws

    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)

    ws.onmessage = (ev) => {
      const event: WsEvent = JSON.parse(ev.data)
      const id = String(++msgId)
      const timestamp = Date.now()

      switch (event.type) {
        case 'step':
          setMessages(prev => [...prev, {
            id, type: 'step', timestamp,
            content: event.data.summary
              ? `${event.data.tool} — ${event.data.summary}`
              : `${event.data.tool} [${event.data.status}]`,
            data: event.data,
          }])
          break
        case 'log': {
          const level = (event.data.level as string) || 'info'
          const module = (event.data.module as string) || ''
          const message = (event.data.message as string) || ''
          if (!message.trim()) break
          setMessages(prev => [...prev, {
            id, type: 'system', timestamp,
            content: message,
            data: { level, module },
          }])
          break
        }
        case 'progress':
          // 就地更新最后一条 progress，不追加新气泡
          setMessages(prev => {
            const progressId = PROGRESS_ID_PREFIX + taskId
            const idx = prev.findIndex(m => m.id === progressId)
            const msg: ChatMessage = {
              id: progressId, type: 'progress', timestamp,
              content: `提取进度: ${event.data.current}/${event.data.total}`,
              data: event.data,
            }
            if (idx >= 0) {
              const next = [...prev]
              next[idx] = msg
              return next
            }
            return [...prev, msg]
          })
          break
        case 'result':
          setMessages(prev => [...prev, {
            id, type: 'result', timestamp,
            content: `提取完成, 共 ${event.data.total} 条数据`,
            data: event.data,
          }])
          break
        case 'error':
          setMessages(prev => [...prev, {
            id, type: 'error', timestamp,
            content: event.data.message as string,
          }])
          // 任务出错，自动断开
          ws.close()
          break
        case 'done':
          setMessages(prev => [...prev, {
            id, type: 'system', timestamp,
            content: '✓ 任务完成',
          }])
          // 任务完成，自动断开
          ws.close()
          break
      }
    }
  }, [])

  const disconnect = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  const addUserMessage = useCallback((content: string) => {
    setMessages(prev => [...prev, {
      id: String(++msgId),
      type: 'user',
      content,
      timestamp: Date.now(),
    }])
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  useEffect(() => {
    return () => { wsRef.current?.close() }
  }, [])

  return { messages, connected, connect, disconnect, addUserMessage, clearMessages }
}
