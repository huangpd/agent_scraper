import { useEffect, useState } from 'react'
import { ChatArea } from './components/ChatArea'
import { InputBar } from './components/InputBar'
import { OptimizePreview } from './components/OptimizePreview'
import { useWebSocket } from './hooks/useWebSocket'
import type { OptimizeResponse, TaskResponse } from './types'

export default function App() {
  const { messages, connected, connect, disconnect, addUserMessage, clearMessages } = useWebSocket()
  const [loading, setLoading] = useState(false)
  const [headless, setHeadless] = useState(false)
  const [optimizing, setOptimizing] = useState(false)
  const [optimizeData, setOptimizeData] = useState<OptimizeResponse | null>(null)
  // 暂存发送时的参数，供优化确认后使用
  const [pendingImages, setPendingImages] = useState<string[]>([])
  const [pendingOriginal, setPendingOriginal] = useState('')

  // WS 断开时（任务完成/出错自动断开）重置 loading
  useEffect(() => {
    if (!connected && loading) {
      setLoading(false)
    }
  }, [connected])

  /** 实际提交任务 */
  const submitTask = async (instruction: string, images: string[]) => {
    setLoading(true)
    try {
      const res = await fetch('/api/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction, headless, images }),
      })
      const data: TaskResponse = await res.json()
      connect(data.task_id)
    } catch (e) {
      addUserMessage(`[错误] 无法连接服务器: ${e}`)
      setLoading(false)
    }
  }

  const handleSend = async (instruction: string, images?: string[]) => {
    const imgs = images || []
    const imgHint = imgs.length ? ` [+${imgs.length}张参考图]` : ''
    addUserMessage(instruction + imgHint)

    // 暂存参数
    setPendingOriginal(instruction)
    setPendingImages(imgs)

    // 先调优化接口
    setOptimizing(true)
    try {
      const res = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction }),
      })
      const data: OptimizeResponse = await res.json()

      if (data.skippable) {
        // 无需优化，直接执行
        setOptimizing(false)
        await submitTask(instruction, imgs)
      } else {
        // 弹出预览弹窗
        setOptimizeData(data)
        setOptimizing(false)
      }
    } catch {
      // 优化接口失败，降级为直接执行
      setOptimizing(false)
      await submitTask(instruction, imgs)
    }
  }

  const handleOptimizeConfirm = async (optimizedInstruction: string) => {
    addUserMessage(`[策略优化] 使用优化后的指令`)
    setOptimizeData(null)
    await submitTask(optimizedInstruction, pendingImages)
  }

  const handleOptimizeSkip = async () => {
    setOptimizeData(null)
    await submitTask(pendingOriginal, pendingImages)
  }

  const handleOptimizeCancel = () => {
    setOptimizeData(null)
    setOptimizing(false)
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
    setOptimizing(false)
    setOptimizeData(null)
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
        disabled={loading || optimizing}
        running={loading || connected || optimizing}
        headless={headless}
        onHeadlessChange={setHeadless}
      />
      {optimizeData && (
        <OptimizePreview
          data={optimizeData}
          onConfirm={handleOptimizeConfirm}
          onSkip={handleOptimizeSkip}
          onCancel={handleOptimizeCancel}
        />
      )}
    </div>
  )
}
