"use client";

type ChatbotLauncherProps = {
  isOpen: boolean;
  onClick: () => void;
};

export function ChatbotLauncher({ isOpen, onClick }: ChatbotLauncherProps) {
  return (
    <button
      onClick={onClick}
      aria-label={isOpen ? "Close chatbot" : "Open movie recommendation chatbot"}
      aria-expanded={isOpen}
      aria-controls="chatbot-window"
      className="fixed bottom-5 right-5 z-[9998] flex h-[60px] w-[60px] items-center justify-center rounded-full border border-border bg-surface-alt text-2xl text-primary shadow-glow"
    >
      <span aria-hidden="true">{isOpen ? "✕" : "💬"}</span>
      {!isOpen && (
        <span className="pointer-events-none absolute bottom-[30px] right-[70px] translate-x-[10px] whitespace-nowrap rounded-xl border border-border bg-surface-alt px-3 py-1.5 text-sm text-primary opacity-0 transition-all [button:hover_&]:translate-x-0 [button:hover_&]:opacity-100">
          Check Out My Movie Recommendation Chatbot!
        </span>
      )}
    </button>
  );
}
