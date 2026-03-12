export type MessageType = 'user' | 'system' | 'result' | 'error' | 'step' | 'progress'

export interface ChatMessage {
  id: string
  type: MessageType
  content: string
  data?: Record<string, unknown>
  timestamp: number
}

export interface WsEvent {
  type: string
  data: Record<string, unknown>
}

export interface TaskResponse {
  task_id: string
}
