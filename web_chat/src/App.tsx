import { useState, useRef, useEffect, useCallback } from 'react';
import { ChatBubble } from './components/ChatBubble';
import { ChatInput } from './components/ChatInput';
import { ToolIndicator } from './components/ToolIndicator';
import type { Message, ToolActivity, SSEEvent } from './types';

const API_BASE = 'http://localhost:8000';

function generateId() {
  return Math.random().toString(36).substring(2, 15);
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [toolActivity, setToolActivity] = useState<ToolActivity | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [threadId] = useState(() => generateId());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, toolActivity]);

  const handleSend = useCallback(
    async (message: string) => {
      const userMessage: Message = {
        id: generateId(),
        role: 'user',
        content: message,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);
      setError(null);
      setToolActivity(null);

      abortControllerRef.current = new AbortController();

      try {
        const response = await fetch(`${API_BASE}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: 'ba91d8a5-46b2-46f7-aaf4-189a54e17fe9', // TODO: get from auth
            thread_id: threadId,
            message,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error('No response body');

        const decoder = new TextDecoder();
        let buffer = '';
        let aiContent = '';
        let aiMessageId = generateId();

        const assistantMessage: Message = {
          id: aiMessageId,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;

            try {
              const event: SSEEvent = JSON.parse(line.slice(6));

              switch (event.type) {
                case 'token':
                  aiContent += event.content;
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === aiMessageId
                        ? { ...m, content: aiContent }
                        : m
                    )
                  );
                  break;

                case 'tool_start':
                  setToolActivity({
                    tool: event.tool,
                    status: 'start',
                    timestamp: new Date(),
                  });
                  break;

                case 'tool_end':
                  setToolActivity({
                    tool: event.tool,
                    status: 'end',
                    timestamp: new Date(),
                  });
                  break;

                case 'done':
                  setIsLoading(false);
                  break;

                case 'error':
                  setError(event.message);
                  setIsLoading(false);
                  break;
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }

        if (buffer.startsWith('data: ')) {
          try {
            const event: SSEEvent = JSON.parse(buffer.slice(6));
            if (event.type === 'error') {
              setError(event.message);
            }
          } catch {
            // Skip
          }
        }

        setIsLoading(false);
      } catch (err) {
        if (err instanceof Error && err.name !== 'AbortError') {
          setError(err.message);
          setIsLoading(false);
        }
      }
    },
    [threadId]
  );

  const handleCancel = () => {
    abortControllerRef.current?.abort();
    setIsLoading(false);
  };

  return (
    <div className="flex flex-col h-full bg-[#f2f2f7]">
      {/* Header */}
      <header
        className="px-4 py-3 bg-white border-b border-[#e5e5ea] flex items-center justify-between"
        style={{ paddingTop: 'calc(12px+env(safe-area-inset-top))' }}
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-[#0a84ff] flex items-center justify-center">
            <span className="text-white text-lg font-semibold">M</span>
          </div>
          <div>
            <h1 className="text-[17px] font-semibold text-[#000]">Mint Money</h1>
            <p className="text-xs text-[#8e8e93]">AI Financial Assistant</p>
          </div>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto pt-4">
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center h-full text-center px-8">
            <div className="w-16 h-16 rounded-full bg-[#0a84ff]/10 flex items-center justify-center mb-4">
              <span className="text-3xl">💰</span>
            </div>
            <h2 className="text-[20px] font-semibold text-[#000] mb-2">
              Welcome to Mint Money
            </h2>
            <p className="text-[#8e8e93] text-[15px]">
              Ask me anything about your finances, expenses, budgets, or financial goals.
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <ChatBubble key={msg.id} message={msg} />
        ))}

        {toolActivity && (
          <ToolIndicator tool={toolActivity.tool} status={toolActivity.status} />
        )}

        {isLoading && messages.length > 0 && (
          <div className="flex justify-start mb-3 px-4">
            <div className="bg-[#e5e5ea] rounded-2xl rounded-bl-md px-4 py-3">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-[#8e8e93] rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-[#8e8e93] rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-[#8e8e93] rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="flex justify-center mb-3 px-4">
            <div className="bg-[#ff3b30]/10 text-[#ff3b30] px-4 py-2 rounded-xl text-sm">
              Error: {error}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </main>

      {/* Input */}
      {isLoading ? (
        <div className="px-4 pb-[calc(16px+env(safe-area-inset-bottom))] pt-3 bg-[#f2f2f7]">
          <button
            onClick={handleCancel}
            className="w-full py-3 bg-[#ff3b30] text-white rounded-2xl text-[17px] font-medium active:bg-[#d62c1a] transition-colors"
          >
            Cancel
          </button>
        </div>
      ) : (
        <ChatInput onSend={handleSend} />
      )}
    </div>
  );
}

export default App;
