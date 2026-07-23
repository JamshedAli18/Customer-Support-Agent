import "./Header.css";

const SENTIMENT_CONFIG = {
  neutral: { label: "Calm", color: "var(--sentiment-neutral)" },
  frustrated: { label: "Frustrated", color: "var(--sentiment-frustrated)" },
  angry: { label: "Escalating", color: "var(--sentiment-angry)" },
};

export default function Header({ sentiment }) {
  const info = SENTIMENT_CONFIG[sentiment] || SENTIMENT_CONFIG.neutral;

  return (
    <header className="header" id="app-header">
      <div className="header__inner">
        <div className="header__brand">
          <div className="header__logo">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect width="28" height="28" rx="8" fill="var(--color-accent)" />
              <path d="M8 14C8 10.686 10.686 8 14 8C17.314 8 20 10.686 20 14" stroke="white" strokeWidth="2" strokeLinecap="round" />
              <circle cx="14" cy="17" r="2.5" fill="white" />
              <path d="M14 14.5V9.5" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </div>
          <div className="header__title-group">
            <h1 className="header__title">VoiceCart</h1>
            <span className="header__subtitle">Support Agent</span>
          </div>
        </div>

        <div
          className="header__sentiment"
          style={{ "--sentiment-color": info.color }}
          id="sentiment-badge"
        >
          <span className="header__sentiment-indicator" />
          <span className="header__sentiment-label">{info.label}</span>
        </div>
      </div>
    </header>
  );
}
