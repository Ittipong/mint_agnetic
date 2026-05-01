interface ToolIndicatorProps {
  tool: string;
  status: 'start' | 'end';
}

export function ToolIndicator({ tool, status }: ToolIndicatorProps) {
  const isStart = status === 'start';

  return (
    <div className="flex justify-start mb-3 px-4">
      <div
        className={`px-3 py-1.5 rounded-full text-xs font-medium ${
          isStart
            ? 'bg-[#34c759] text-white'
            : 'bg-[#8e8e93] text-white'
        }`}
      >
        {isStart ? `🔧 ${tool}` : `✓ ${tool}`}
      </div>
    </div>
  );
}
