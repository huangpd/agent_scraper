import { useState } from 'react'
import { ChatArea } from './components/ChatArea'
import { InputBar } from './components/InputBar'
import { useWebSocket } from './hooks/useWebSocket'
import type { TaskResponse } from './types'

export default function App() {
  const { messages, connected, connect, addUserMessage } = useWebSocket()
  const [loading, setLoading] = useState(false)
  const [headless, setHeadless] = useState(false)

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
    } finally {
      setLoading(false)
    }
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
        disabled={loading}
        headless={headless}
        onHeadlessChange={setHeadless}
      />
    </div>
  )
}
