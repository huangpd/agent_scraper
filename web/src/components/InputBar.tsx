import { useState, useEffect, useRef, useCallback, useLayoutEffect } from 'react'

interface Props {
  onSend: (text: string, images?: string[]) => void
  disabled: boolean
  headless: boolean
  onHeadlessChange: (v: boolean) => void
}

type Mode = 'extract' | 'capture'

const TEMPLATES: Record<Mode, string> = {
  extract: `步骤1: 打开网址 https://
步骤2: 找到并点击 ""
步骤3:
步骤4: 提取所有文件名和下载链接，用json格式

遍历方式: (可选，删除不需要的行)
- 点击 "Load more" 加载全部
- 遍历所有子文件夹
- 翻页提取所有页

样本数据:
{"字段名1":"样本值1","字段名2":"样本值2"}
{"字段名1":"样本值3","字段名2":"样本值4"}`,

  capture: `步骤1: 打开网址 https://
步骤2:
步骤3:
步骤4: 获取当前页面的下载链接URL，保存为JSON

提取字段:
- download_url: 下载链接`,
}

const MODE_LABELS: Record<Mode, { name: string; desc: string }> = {
  extract: { name: '样本模式', desc: '批量提取列表/表格数据，支持样本训练' },
  capture: { name: '自由模式', desc: '浏览器操作后捕获少量值（URL、文本等）' },
}

/** 读取文件为 base64 data URL */
function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

export function InputBar({ onSend, disabled, headless, onHeadlessChange }: Props) {
  const [text, setText] = useState('')
  const [mode, setMode] = useState<Mode>('extract')
  const [hasEdited, setHasEdited] = useState(false)
  const [images, setImages] = useState<string[]>([])  // base64 data URLs
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // 切换模式时自动填充模板
  useEffect(() => {
    if (!hasEdited || text === '' || Object.values(TEMPLATES).includes(text)) {
      setText(TEMPLATES[mode])
      setHasEdited(false)
    }
  }, [mode])

  // 全局粘贴监听：支持 Ctrl+V 粘贴图片
  useEffect(() => {
    const handlePaste = async (e: ClipboardEvent) => {
      const items = e.clipboardData?.items
      if (!items) return
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          e.preventDefault()
          const file = item.getAsFile()
          if (file) {
            const dataUrl = await readFileAsDataURL(file)
            setImages(prev => [...prev, dataUrl])
          }
        }
      }
    }
    document.addEventListener('paste', handlePaste)
    return () => document.removeEventListener('paste', handlePaste)
  }, [])

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
    setHasEdited(true)
  }

  const handleSend = () => {
    const trimmed = text.trim()
    if (!trimmed) return
    onSend(trimmed, images.length > 0 ? images : undefined)
    setText('')
    setImages([])
    setHasEdited(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleReset = () => {
    setText(TEMPLATES[mode])
    setImages([])
    setHasEdited(false)
  }

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files) return
    for (const file of files) {
      if (file.type.startsWith('image/')) {
        const dataUrl = await readFileAsDataURL(file)
        setImages(prev => [...prev, dataUrl])
      }
    }
    // 清空 input 以便重复选择同一文件
    e.target.value = ''
  }, [])

  const removeImage = (index: number) => {
    setImages(prev => prev.filter((_, i) => i !== index))
  }

  // ── 拖拽拉伸 textarea ──
  const [textareaHeight, setTextareaHeight] = useState<number | null>(null)
  const dragRef = useRef<{ startY: number; startH: number } | null>(null)

  // 模式切换时重置高度
  useEffect(() => { setTextareaHeight(null) }, [mode])

  const onDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    const ta = textareaRef.current
    if (!ta) return
    dragRef.current = { startY: e.clientY, startH: ta.offsetHeight }

    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current) return
      const delta = dragRef.current.startY - ev.clientY  // 向上拉 → 增大
      setTextareaHeight(Math.max(80, dragRef.current.startH + delta))
    }
    const onUp = () => {
      dragRef.current = null
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }, [])

  return (
    <div className="input-bar">
      {/* 拖拽条 */}
      <div className="resize-handle" onMouseDown={onDragStart}>
        <span className="resize-handle-bar" />
      </div>

      {/* 模式切换 */}
      <div className="mode-switch">
        {(Object.keys(MODE_LABELS) as Mode[]).map(m => (
          <button
            key={m}
            className={`mode-btn ${mode === m ? 'active' : ''}`}
            onClick={() => setMode(m)}
            disabled={disabled}
            title={MODE_LABELS[m].desc}
          >
            {MODE_LABELS[m].name}
          </button>
        ))}
        <span className="mode-desc">{MODE_LABELS[mode].desc}</span>
      </div>

      {/* 图片预览区 */}
      {images.length > 0 && (
        <div className="image-preview-bar">
          {images.map((img, i) => (
            <div key={i} className="image-preview-item">
              <img src={img} alt={`参考图 ${i + 1}`} />
              <button className="image-remove-btn" onClick={() => removeImage(i)} title="移除">
                &times;
              </button>
            </div>
          ))}
        </div>
      )}

      {/* 输入区 */}
      <div className="input-row">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleTextChange}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={mode === 'extract' ? 12 : 8}
          spellCheck={false}
          style={textareaHeight ? { height: textareaHeight, resize: 'none' } : undefined}
        />
        <div className="input-actions">
          <button className="send-btn" onClick={handleSend} disabled={disabled || !text.trim()}>
            发送
          </button>
          <button
            className="upload-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            title="上传参考截图（红框标注要点击的元素）"
          >
            截图
          </button>
          <button className="reset-btn" onClick={handleReset} disabled={disabled}>
            重置
          </button>
        </div>
      </div>

      {/* 隐藏的文件选择器 */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />

      {/* 底部选项 */}
      <div className="input-options">
        <label className="headless-toggle">
          <input
            type="checkbox"
            checked={headless}
            onChange={e => onHeadlessChange(e.target.checked)}
            disabled={disabled}
          />
          <span>无头模式</span>
        </label>
        <span className="hint">Ctrl+V 粘贴截图 · Shift+Enter 换行 · Enter 发送</span>
      </div>
    </div>
  )
}
