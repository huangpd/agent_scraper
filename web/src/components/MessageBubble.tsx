import type { ChatMessage } from '../types'
import { ResultTable } from './ResultTable'

/** 兜底清理 ANSI 转义序列（防御历史事件/重连缓存） */
function stripAnsi(s: string): string {
  return s.replace(/\x1b\[[0-9;]*m/g, '').replace(/\[([0-9;]*)m/g, '')
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

  if (msg.type === 'anomaly') {
    const categories = (msg.data?.categories ?? {}) as Record<string, {
      label: string; count: number; items: Array<Record<string, string>>
    }>
    return (
      <div className="msg-row msg-left">
        <div className="bubble bubble-anomaly">
          <div className="anomaly-header">{msg.content}</div>
          {Object.entries(categories).map(([cat, group]) => (
            <div key={cat} className="anomaly-group">
              <div className="anomaly-group-title">
                <span className="anomaly-tag">{group.label}</span>
                <span className="anomaly-count">{group.count} 条</span>
              </div>
              <div className="anomaly-items">
                {group.items.map((item, i) => (
                  <div key={i} className="anomaly-item">
                    <span className="anomaly-entry" title={item.entry}>
                      {item.entry.length > 80 ? item.entry.slice(0, 80) + '\u2026' : item.entry}
                    </span>
                    <span className="anomaly-reason">{item.reason}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (msg.type === 'error') {
    return (
      <div className="msg-row msg-left">
        <div className="bubble bubble-error">{stripAnsi(msg.content)}</div>
      </div>
    )
  }

  if (msg.type === 'step') {
    const status = (msg.data?.status as string) || ''
    const icon = status === 'success' ? '✓' : status === 'failed' ? '✗' : '▸'
    return (
      <div className="msg-row msg-left">
        <div className={`bubble bubble-step step-${status}`}>
          <span className="step-icon">{icon}</span> {stripAnsi(msg.content)}
        </div>
      </div>
    )
  }

  if (msg.type === 'progress') {
    const cur = (msg.data?.current as number) || 0
    const total = (msg.data?.total as number) || 1
    const pct = Math.round((cur / total) * 100)
    return (
      <div className="msg-row msg-left">
        <div className="bubble bubble-log level-info">
          <div className="progress-text">{msg.content}</div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
        </div>
      </div>
    )
  }

  // system / log — 按 level 和 module 分级渲染
  const level = (msg.data?.level as string) || 'info'
  const mod = (msg.data?.module as string) || ''
  const text = stripAnsi(msg.content)

  return (
    <div className="msg-row msg-left">
      <div className={`bubble bubble-log level-${level}`}>
        {mod && <span className="log-module">{mod}</span>}
        <span className="log-text">{text}</span>
      </div>
    </div>
  )
}
