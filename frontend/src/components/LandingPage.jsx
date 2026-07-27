import { useRef, useState } from "react";
import "./LandingPage.css";

const EXIT_DURATION_MS = 420;

function IconCart() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="9" cy="21" r="1.5" fill="currentColor" />
      <circle cx="18" cy="21" r="1.5" fill="currentColor" />
      <path d="M2.5 3H4.5L6.8 14.2C6.98 15.06 7.75 15.68 8.63 15.68H18.4C19.25 15.68 20.01 15.09 20.21 14.26L22 6.5H5.4"
        stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconPackage() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M21 8.5V16a1.5 1.5 0 0 1-.76 1.3l-7.5 4.3a1.5 1.5 0 0 1-1.48 0l-7.5-4.3A1.5 1.5 0 0 1 3 16V8.5"
        stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3.27 7.5 12 12.5l8.73-5" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 22V12.5" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" />
      <path d="M16.5 5 7.5 10" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" />
      <path d="m7 2.5 8.73 5-4.36 2.5-8.73-5Z" stroke="currentColor" strokeWidth="1.75" strokeLinejoin="round" />
    </svg>
  );
}

function IconCancel() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.75" />
      <path d="M9.5 9.5 14.5 14.5M14.5 9.5 9.5 14.5" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" />
    </svg>
  );
}

function IconShield() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2.5 4 5.5V11c0 5.25 3.4 9.2 8 10.5 4.6-1.3 8-5.25 8-10.5V5.5Z"
        stroke="currentColor" strokeWidth="1.75" strokeLinejoin="round" />
      <path d="M8.75 12 11 14.25 15.5 9.5" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconHeadphones() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M4 13v-1a8 8 0 0 1 16 0v1" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" />
      <rect x="2.5" y="13" width="4.5" height="7" rx="1.75" stroke="currentColor" strokeWidth="1.75" />
      <rect x="17" y="13" width="4.5" height="7" rx="1.75" stroke="currentColor" strokeWidth="1.75" />
    </svg>
  );
}

function IconMic() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="9" y="2.5" width="6" height="12" rx="3" stroke="currentColor" strokeWidth="1.75" />
      <path d="M5.5 11.5c0 3.6 2.9 6.5 6.5 6.5s6.5-2.9 6.5-6.5" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" />
      <path d="M12 18v3.5" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" />
    </svg>
  );
}

const CAPABILITIES = [
  { Icon: IconCart, title: "Book an order", description: "Order your ShopNest Pulse earbuds in any color" },
  { Icon: IconPackage, title: "Track your order", description: "Check the status of an existing order" },
  { Icon: IconCancel, title: "Cancel an order", description: "Cancel an order that's still processing" },
  { Icon: IconShield, title: "File a warranty claim", description: "Get help with a defective product" },
  { Icon: IconHeadphones, title: "Product & troubleshooting help", description: "Ask about specs, setup, or fixes" },
  { Icon: IconMic, title: "Voice or text", description: "Talk or type, however you prefer" },
];

export default function LandingPage({ onStartChatting }) {
  const [isLeaving, setIsLeaving] = useState(false);
  const exitTimerRef = useRef(null);

  function handleStartChatting() {
    if (isLeaving) return;
    setIsLeaving(true);
    exitTimerRef.current = setTimeout(onStartChatting, EXIT_DURATION_MS);
  }

  return (
    <div className={`landing ${isLeaving ? "landing--leaving" : ""}`} id="landing-page">
      <div className="landing__glow" aria-hidden="true" />
      <div className="landing__aurora" aria-hidden="true">
        <span className="landing__aurora-blob landing__aurora-blob--one" />
        <span className="landing__aurora-blob landing__aurora-blob--two" />
        <span className="landing__aurora-blob landing__aurora-blob--three" />
      </div>
      <div className="landing__grid-overlay" aria-hidden="true" />

      <div className="landing__inner">
        <section className="landing__hero">
          <h1 className="landing__brand">ShopNest Pulse Support</h1>
          <p className="landing__tagline">
            Your AI-powered voice & chat assistant for ShopNest Pulse wireless earbuds
          </p>
        </section>

        <section className="landing__cards">
          {CAPABILITIES.map(({ Icon, title, description }) => (
            <div className="landing__card" key={title}>
              <div className="landing__card-icon">
                <Icon />
              </div>
              <h3 className="landing__card-title">{title}</h3>
              <p className="landing__card-description">{description}</p>
            </div>
          ))}
        </section>

        <button
          className="landing__cta"
          onClick={handleStartChatting}
          disabled={isLeaving}
          id="start-chatting-button"
        >
          Start Chatting
        </button>
      </div>
    </div>
  );
}
