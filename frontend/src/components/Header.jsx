import "./Header.css";

function IconCart() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="9" cy="21" r="1.5" fill="currentColor" />
      <circle cx="18" cy="21" r="1.5" fill="currentColor" />
      <path d="M2.5 3H4.5L6.8 14.2C6.98 15.06 7.75 15.68 8.63 15.68H18.4C19.25 15.68 20.01 15.09 20.21 14.26L22 6.5H5.4"
        stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconPackage() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M21 8.5V16a1.5 1.5 0 0 1-.76 1.3l-7.5 4.3a1.5 1.5 0 0 1-1.48 0l-7.5-4.3A1.5 1.5 0 0 1 3 16V8.5"
        stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3.27 7.5 12 12.5l8.73-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 22V12.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="m7 2.5 8.73 5-4.36 2.5-8.73-5Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

function IconCancel() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
      <path d="M9.5 9.5 14.5 14.5M14.5 9.5 9.5 14.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function IconShield() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2.5 4 5.5V11c0 5.25 3.4 9.2 8 10.5 4.6-1.3 8-5.25 8-10.5V5.5Z"
        stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
      <path d="M8.75 12 11 14.25 15.5 9.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconHeadphones() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M4 13v-1a8 8 0 0 1 16 0v1" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <rect x="2.5" y="13" width="4.5" height="7" rx="1.75" stroke="currentColor" strokeWidth="2" />
      <rect x="17" y="13" width="4.5" height="7" rx="1.75" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

function IconMic() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="9" y="2.5" width="6" height="12" rx="3" stroke="currentColor" strokeWidth="2" />
      <path d="M5.5 11.5c0 3.6 2.9 6.5 6.5 6.5s6.5-2.9 6.5-6.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M12 18v3.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

const CAPABILITIES = [
  { Icon: IconCart, label: "Order" },
  { Icon: IconPackage, label: "Track" },
  { Icon: IconCancel, label: "Cancel" },
  { Icon: IconShield, label: "Warranty" },
  { Icon: IconHeadphones, label: "Support" },
  { Icon: IconMic, label: "Voice" },
];

export default function Header() {
  return (
    <header className="header" id="app-header">
      <div className="header__inner">
        <div className="header__brand">
          <div className="header__logo">
            <svg width="20" height="20" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
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

        <div className="header__badges" id="capability-badges">
          {CAPABILITIES.map(({ Icon, label }) => (
            <span className="header__badge" key={label}>
              <Icon />
              {label}
            </span>
          ))}
        </div>
      </div>
    </header>
  );
}
