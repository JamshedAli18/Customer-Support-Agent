import "./InputBar.css";

const VOICE_STAGE_LABEL = {
  idle: "Tap to speak",
  recording: "Listening...",
  transcribing: "Transcribing...",
  thinking: "Thinking...",
  speaking: "Speaking...",
};

export default function InputBar({
  inputText,
  onInputChange,
  onSendText,
  onMicTap,
  onStopSpeaking,
  voiceStage,
  isStreaming,
}) {
  const inputDisabled = isStreaming || voiceStage !== "idle";
  const isRecording = voiceStage === "recording";
  const isVoiceActive = voiceStage !== "idle";

  function handleKeyDown(e) {
    if (e.key === "Enter") onSendText();
  }

  return (
    <footer className="input-bar" id="input-bar">
      <div className="input-bar__inner">
        <div className={`input-bar__shell ${isVoiceActive ? "input-bar__shell--active" : ""}`}>

          {isVoiceActive ? (
            <div className="input-bar__voice-status">
              <div className="input-bar__voice-waves">
                <span className="input-bar__wave" />
                <span className="input-bar__wave" />
                <span className="input-bar__wave" />
              </div>
              <span className="input-bar__voice-label">
                {VOICE_STAGE_LABEL[voiceStage]}
              </span>
            </div>
          ) : (
            <input
              type="text"
              className="input-bar__input"
              value={inputText}
              onChange={(e) => onInputChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message..."
              disabled={inputDisabled}
              id="text-input"
            />
          )}

          <div className="input-bar__actions">
            {voiceStage === "speaking" ? (
              <button
                className="input-bar__stop"
                onClick={onStopSpeaking}
                aria-label="Stop speaking"
                id="stop-button"
              >
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                  <rect x="4" y="4" width="10" height="10" rx="1.5" fill="currentColor" />
                </svg>
              </button>
            ) : (
              <>
                <button
                  className={`input-bar__mic ${isRecording ? "input-bar__mic--recording" : ""}`}
                  onClick={onMicTap}
                  disabled={isStreaming || (voiceStage !== "idle" && voiceStage !== "recording")}
                  aria-label={isRecording ? "Stop recording" : "Start recording"}
                  id="mic-button"
                >
                  {isRecording ? (
                    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                      <rect x="4" y="4" width="10" height="10" rx="2" fill="currentColor" />
                    </svg>
                  ) : (
                    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                      <rect x="7" y="2" width="4" height="9" rx="2" fill="currentColor" />
                      <path d="M4 9C4 11.76 6.24 14 9 14C11.76 14 14 11.76 14 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                      <line x1="9" y1="14" x2="9" y2="16.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                    </svg>
                  )}
                </button>

                <button
                  className="input-bar__send"
                  onClick={onSendText}
                  disabled={inputDisabled || !inputText.trim()}
                  aria-label="Send message"
                  id="send-button"
                >
                  <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                    <path d="M3 9H15M15 9L10 4M15 9L10 14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              </>
            )}
          </div>
        </div>

        <p className="input-bar__hint">
          Press Enter to send, or use the microphone for voice input
        </p>
      </div>
    </footer>
  );
}
