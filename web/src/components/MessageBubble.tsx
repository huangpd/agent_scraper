import type { ChatMessage } from '../types'
import { ResultTable } from './ResultTable'

/** 从日志内容中提取模块标签，如 [Navigator]、[Extractor] 等 */
function parseLogTag(content: string): { tag: string; text: string } {
  const m = content.match(/^\[(\w+)](.*)$/)
  if (m) return { tag: m[1], text: m[2].trim() }
  // 缩进行：子模块输出（以空格开头）
  const indent = content.match(/^\s{2,}(.+)$/)
  if (indent) return { tag: '', text: indent[1].trim() }
  return { tag: '', text: content }
}

function LogContent({ content }: { content: string }) {
  const { tag, text } = parseLogTag(content)
  if (tag) {
    return <><span className="log-tag">[{tag}]</span> {text}</>
  }
  return <>{text}</>
}

export function MessageBubble({ msg }: { msg: ChatMessage }) {
  if (msg.type === 'user') {
    return (
      <div className="msg-row msg-right">
        <div className="bubble bubble-user">{msg.content}</div>
      </div>
    )
  }

  if (msg.type === 'result') {
    return (
      <div className="msg-row msg-left">
        <div className="bubble bubble-result">
          <div className="result-header">{msg.content}</div>
          <ResultTable data={msg.data} />
        </div>
      </div>
    )
  }

  if (msg.type === 'error') {
    return (
      <div className="msg-row msg-left">
        <div className="bubble bubble-error">{msg.content}</div>
      </div>
    )
  }

  if (msg.type === 'step') {
    return (
      <div className="msg-row msg-left">
        <div className="bubble bubble-step">{msg.content}</div>
      </div>
    )
  }

  if (msg.type === 'progress') {
    const cur = (msg.data?.current as number) || 0
    const total = (msg.data?.total as number) || 1
    const pct = Math.round((cur / total) * 100)
    return (
      <div className="msg-row msg-left">
        <div className="bubble bubble-system">
          <div className="progress-text">{msg.content}</div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
        </div>
      </div>
    )
  }

  // system / log — 带模块标签高亮
  return (
    <div className="msg-row msg-left">
      <div className="bubble bubble-system">
        <LogContent content={msg.content} />
      </div>
    </div>
  )
}
