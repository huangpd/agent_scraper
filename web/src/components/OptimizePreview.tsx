import { useState } from 'react'
import type { OptimizeResponse } from '../types'

interface Props {
  data: OptimizeResponse
  onConfirm: (instruction: string) => void
  onSkip: () => void
  onCancel: () => void
}

export function OptimizePreview({ data, onConfirm, onSkip, onCancel }: Props) {
  const [editedText, setEditedText] = useState(data.optimized)

  return (
    <div className="optimize-overlay">
      <div className="optimize-modal">
        <h3 className="optimize-title">策略优化</h3>

        {/* 推理过程 */}
        <div className="optimize-section">
          <label className="optimize-label">推理过程</label>
          <div className="optimize-reasoning">{data.reasoning}</div>
        </div>

        {/* 改动列表 */}
        {data.changes.length > 0 && (
          <div className="optimize-section">
            <label className="optimize-label">改动</label>
            <ul className="optimize-changes">
              {data.changes.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </div>
        )}

        {/* 优化后指令（可编辑） */}
        <div className="optimize-section">
          <label className="optimize-label">优化后指令（可编辑）</label>
          <textarea
            className="optimize-textarea"
            value={editedText}
            onChange={e => setEditedText(e.target.value)}
            rows={10}
          />
        </div>

        {/* 按钮 */}
        <div className="optimize-actions">
          <button className="opt-btn opt-confirm" onClick={() => onConfirm(editedText)}>
            确认执行
          </button>
          <button className="opt-btn opt-skip" onClick={onSkip}>
            跳过优化
          </button>
          <button className="opt-btn opt-cancel" onClick={onCancel}>
            取消
          </button>
        </div>
      </div>
    </div>
  )
}
