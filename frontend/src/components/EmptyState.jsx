import "./EmptyState.css";

export default function EmptyState() {
  return (
    <div className="empty-state" id="empty-state">
      <div className="empty-state__icon">
        <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="24" cy="24" r="23" stroke="var(--color-border)" strokeWidth="1.5" strokeDasharray="4 3" />
          <path d="M16 26C16 26 18.5 30 24 30C29.5 30 32 26 32 26" stroke="var(--color-ink-muted)" strokeWidth="1.5" strokeLinecap="round" />
          <circle cx="18" cy="21" r="1.5" fill="var(--color-ink-muted)" />
          <circle cx="30" cy="21" r="1.5" fill="var(--color-ink-muted)" />
          <path d="M24 14V11" stroke="var(--color-accent)" strokeWidth="1.5" strokeLinecap="round" />
          <path d="M29 15.5L31 13.5" stroke="var(--color-accent)" strokeWidth="1.5" strokeLinecap="round" opacity="0.6" />
          <path d="M19 15.5L17 13.5" stroke="var(--color-accent)" strokeWidth="1.5" strokeLinecap="round" opacity="0.6" />
        </svg>
      </div>

      <h2 className="empty-state__title">How can I help you today?</h2>
      <p className="empty-state__description">
        Ask about your ShopNest Pulse earbuds or any product support.
        Type a message or tap the microphone to speak.
      </p>

      <div className="empty-state__hints">
        <span className="empty-state__hint">Text</span>
        <span className="empty-state__hint-divider" />
        <span className="empty-state__hint">Voice</span>
      </div>
    </div>
  );
}
