export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface ToolActivity {
  tool: string;
  status: 'start' | 'end';
  timestamp: Date;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  toolActivity: ToolActivity | null;
  error: string | null;
}

export interface ChatRequest {
  user_id: string;
  thread_id: string;
  message: string;
}

export type SSEEvent =
  | { type: 'token'; content: string }
  | { type: 'tool_start'; tool: string }
  | { type: 'tool_end'; tool: string }
  | { type: 'done' }
  | { type: 'error'; message: string };
