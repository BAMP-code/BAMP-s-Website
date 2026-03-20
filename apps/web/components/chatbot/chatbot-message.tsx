type ChatbotMessageProps = {
  sender: "user" | "bot";
  text: string;
};

export function ChatbotMessage({ sender, text }: ChatbotMessageProps) {
  return (
    <div
      className={`max-w-[80%] animate-fade-in rounded-chat px-3 py-2 leading-relaxed shadow-[0_8px_18px_rgba(0,0,0,0.25)] ${
        sender === "user"
          ? "self-end rounded-br-sm bg-gradient-to-br from-[#71d7ff] to-[#8f7dff] text-[#08111f]"
          : "self-start rounded-bl-sm border border-border bg-surface text-primary"
      }`}
      role="listitem"
    >
      <span className="font-semibold">
        {sender === "user" ? "You" : "CHATBOT"}:
      </span>{" "}
      <span>{text}</span>
    </div>
  );
}
