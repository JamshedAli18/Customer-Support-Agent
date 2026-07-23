import "./ChatMessage.css";

const INTENT_TAGS = {
  stock_check: "Inventory Check",
};

export default function ChatMessage({ message, onReplay }) {
  const { role, content, streaming, audioBase64, intent } = message;

  if (role === "system") {
    return (
      <div className="message message--system" id="system-message">
        <div className="message__system-bar" />
        <p className="message__system-text">{content}</p>
      </div>
    );
  }

  const isUser = role === "user";
  const isAssistant = role === "assistant";
  const intentTag = isAssistant ? INTENT_TAGS[intent] : null;

  return (
    <div className={`message message--${role}`}>
      {isAssistant && (
        <div className="message__avatar">
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M5 9C5 6.79 6.79 5 9 5C11.21 5 13 6.79 13 9" stroke="var(--color-accent)" strokeWidth="1.5" strokeLinecap="round" />
            <circle cx="9" cy="11.5" r="1.5" fill="var(--color-accent)" />
          </svg>
        </div>
      )}

      <div className={`message__bubble message__bubble--${role}`}>
        {intentTag && <span className="message__tag">{intentTag}</span>}

        <p className="message__text">
          {content}
          {streaming && <span className="message__cursor" />}
        </p>

        {/* {audioBase64 && (
          <button
            className="message__replay"
            onClick={() => onReplay(audioBase64)}
            id="replay-button"
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
              <polygon points="2,1 11,6 2,11" fill="currentColor" />
            </svg>
            Replay audio
          </button>
        )} */}
      </div>
    </div>
  );
}