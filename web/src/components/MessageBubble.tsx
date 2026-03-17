import type { ChatMessage } from '../types'
import { ResultTable } from './ResultTable'

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
    const summary = (msg.data?.summary as string) || msg.content
    const categories = (msg.data?.categories as Record<string, any>) || {}
    const anomalyCount = (msg.data?.anomaly_count as number) || 0
    const catOrder = ['foreign_domain', 'suspicious_page', 'abnormal_format', 'structural_outlier']

    return (
      <div className="msg-row msg-left">
        <div className="bubble bubble-anomaly">
          <div className="anomaly-summary">
            {anomalyCount > 0 ? '⚠' : '✓'} {stripAnsi(summary)}
          </div>
          {catOrder.map(cat => {
            const group = categories[cat]
            if (!group) return null
            return (
              <div key={cat} className="anomaly-category">
                <div className={`anomaly-cat-header cat-${cat}`}>
                  {group.label}（{group.count}）
                </div>
                <ul className="anomaly-list">
                  {(group.items as any[]).map((it: any, idx: number) => (
                    <li key={idx}>
                      <span className="anomaly-entry">{it.entry}</span>
                      <span className="anomaly-reason">{it.reason}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )
          })}
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
    const icon = status === 'success' ? '✅' : status === 'failed' ? '❌' : '⏺'
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
