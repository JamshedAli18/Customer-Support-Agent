import { useEffect, useRef } from "react";
import ChatMessage from "./ChatMessage";
import EmptyState from "./EmptyState";
import "./ChatArea.css";

export default function ChatArea({ messages, onReplayAudio }) {
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <main className="chat-area" id="chat-area">
      <div className="chat-area__inner">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="chat-area__messages">
            {messages.map((msg, i) => (
              <ChatMessage
                key={i}
                message={msg}
                onReplay={onReplayAudio}
              />
            ))}
          </div>
        )}
        <div ref={chatEndRef} className="chat-area__anchor" />
      </div>
    </main>
  );
}
