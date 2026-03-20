"use client";

import { useRef } from "react";
import { VisuallyHidden } from "@/components/ui/visually-hidden";

type ChatbotInputProps = {
  onSend: (message: string) => void;
  disabled: boolean;
};

export function ChatbotInput({ onSend, disabled }: ChatbotInputProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && inputRef.current?.value.trim()) {
      onSend(inputRef.current.value);
      inputRef.current.value = "";
    }
  };

  return (
    <div className="border-t border-border">
      <VisuallyHidden>
        <label htmlFor="chatbot-msg-input">Type a message to the chatbot</label>
      </VisuallyHidden>
      <input
        id="chatbot-msg-input"
        ref={inputRef}
        type="text"
        placeholder="Type a message... (First response takes time)"
        onKeyDown={handleKeyDown}
        disabled={disabled}
        className="w-full border-none bg-surface-alt px-3 py-[10px] text-[0.95rem] text-primary outline-none placeholder:text-muted disabled:opacity-50"
      />
    </div>
  );
}
