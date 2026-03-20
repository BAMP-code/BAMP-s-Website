"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ChatbotLauncher } from "./chatbot-launcher";
import { ChatbotMessage } from "./chatbot-message";
import { ChatbotInput } from "./chatbot-input";

type Message = {
  sender: "user" | "bot";
  text: string;
};

export function ChatbotWindow() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [sending, setSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = useCallback(async (text: string) => {
    setMessages((prev) => [...prev, { sender: "user", text }]);
    setSending(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: data.response ?? "Sorry, something went wrong!" },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Sorry, the chatbot is currently unavailable." },
      ]);
    } finally {
      setSending(false);
    }
  }, []);

  return (
    <>
      <ChatbotLauncher isOpen={isOpen} onClick={() => setIsOpen((o) => !o)} />

      {isOpen && (
        <div
          id="chatbot-window"
          role="dialog"
          aria-label="Movie recommendation chatbot"
          className="fixed bottom-[90px] right-5 z-[9999] flex h-[400px] max-h-[70%] w-80 max-w-[90%] animate-fade-in flex-col overflow-hidden rounded-chat border border-border bg-surface-alt shadow-chat"
        >
          <div className="border-b border-border bg-surface px-3 py-3 text-center font-bold text-primary">
            Chat with Chatbot
          </div>

          <div
            className="flex flex-1 flex-col gap-[10px] overflow-y-auto scroll-smooth p-[10px]"
            role="list"
            aria-live="polite"
            aria-relevant="additions"
          >
            {messages.map((msg, i) => (
              <ChatbotMessage key={i} sender={msg.sender} text={msg.text} />
            ))}
            <div ref={messagesEndRef} />
          </div>

          <ChatbotInput onSend={handleSend} disabled={sending} />
        </div>
      )}
    </>
  );
}
