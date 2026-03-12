import { useEffect, useRef } from 'react'
import type { ChatMessage } from '../types'
import { MessageBubble } from './MessageBubble'

interface Props {
  messages: ChatMessage[]
}

export function ChatArea({ messages }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="chat-area">
      {messages.length === 0 && (
        <div className="empty-hint">
          选择模式，填写模板，点击发送开始爬取<br />
          <span style={{ fontSize: 13 }}>样本模式：批量提取列表数据 &nbsp;|&nbsp; 自由模式：浏览器操作后捕获值</span>
        </div>
      )}
      {messages.map(msg => (
        <MessageBubble key={msg.id} msg={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
