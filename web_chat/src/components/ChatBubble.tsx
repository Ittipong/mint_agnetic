import type { Message } from '../types';

interface ChatBubbleProps {
  message: Message;
}

export function ChatBubble({ message }: ChatBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3 px-4`}
    >
      <div
        className={`max-w-[75%] px-4 py-2.5 rounded-2xl ${
          isUser
            ? 'bg-[#0a84ff] text-white rounded-br-md'
            : 'bg-[#e5e5ea] text-black rounded-bl-md'
        }`}
        style={{
          borderRadius: isUser ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
        }}
      >
        <p className="text-[17px] leading-[1.35] break-words">{message.content}</p>
        <span
          className={`text-xs mt-1 block ${
            isUser ? 'text-white/70' : 'text-[#8e8e93]'
          }`}
        >
          {message.timestamp.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>
    </div>
  );
}
