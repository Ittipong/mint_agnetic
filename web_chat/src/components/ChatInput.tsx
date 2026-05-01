import { useState, useRef, useEffect } from 'react';
import type { KeyboardEvent, FormEvent } from 'react';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [input]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSend(input.trim());
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="px-4 pb-[calc(16px+env(safe-area-inset-bottom))] pt-3 bg-[#f2f2f7]"
    >
      <div className="flex items-end gap-2 bg-white rounded-2xl px-4 py-2 shadow-sm border border-[#e5e5ea]">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message"
          disabled={disabled}
          className="flex-1 text-[17px] leading-[1.35] resize-none bg-transparent outline-none placeholder-[#8e8e93] min-h-[24px] max-h-[120px] disabled:opacity-50"
          rows={1}
        />
        <button
          type="submit"
          disabled={!input.trim() || disabled}
          className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors ${
            input.trim() && !disabled
              ? 'bg-[#0a84ff] text-white active:bg-[#0070e0]'
              : 'bg-[#e5e5ea] text-[#8e8e93]'
          }`}
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M22 2L11 13" />
            <path d="M22 2L15 22L11 13L2 9L22 2Z" />
          </svg>
        </button>
      </div>
    </form>
  );
}
