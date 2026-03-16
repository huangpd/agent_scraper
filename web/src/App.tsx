import { useEffect, useState } from 'react'
import { ChatArea } from './components/ChatArea'
import { InputBar } from './components/InputBar'
import { useWebSocket } from './hooks/useWebSocket'
import type { TaskResponse } from './types'

export default function App() {
  const { messages, connected, connect, disconnect, addUserMessage, clearMessages } = useWebSocket()
  const [loading, setLoading] = useState(false)
  const [headless, setHeadless] = useState(false)

  // WS 断开时（任务完成/出错自动断开）重置 loading
  useEffect(() => {
    if (!connected && loading) {
      setLoading(false)
    }
  }, [connected])

  const handleSend = async (instruction: string, images?: string[]) => {
    const imgHint = images?.length ? ` [+${images.length}张参考图]` : ''
    addUserMessage(instruction + imgHint)
    setLoading(true)
    try {
      const res = await fetch('/api/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction, headless, images: images || [] }),
      })
      const data: TaskResponse = await res.json()
      connect(data.task_id)
    } catch (e) {
      addUserMessage(`[错误] 无法连接服务器: ${e}`)
      setLoading(false)
    }
  }

  const handleStop = async () => {
    try {
      await fetch('/api/stop-all', { method: 'POST' })
    } catch {
      // 即使请求失败也要重置前端状态
    }
    disconnect()
    clearMessages()
    setLoading(false)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Agent Scraper</h1>
        <span className={`status ${connected ? 'online' : ''}`}>
          {connected ? '已连接' : '未连接'}
        </span>
      </header>
      <ChatArea messages={messages} />
      <InputBar
        onSend={handleSend}
        onStop={handleStop}
        disabled={loading}
        running={loading || connected}
        headless={headless}
        onHeadlessChange={setHeadless}
      />
    </div>
  )
}
