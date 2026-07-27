import { useState, useRef } from "react";
import Header from "./components/Header";
import ChatArea from "./components/ChatArea";
import InputBar from "./components/InputBar";
import LandingPage from "./components/LandingPage";
import "./App.css";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [showChat, setShowChat] = useState(false);
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [sentiment, setSentiment] = useState("neutral");
  const [inputText, setInputText] = useState("");
  const [voiceStage, setVoiceStage] = useState("idle");
  const [isStreaming, setIsStreaming] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const activeAudioRef = useRef(null);

  function appendMessage(msg) {
    setMessages((prev) => [...prev, msg]);
  }

  function updateLastMessage(updater) {
    setMessages((prev) => {
      const updated = [...prev];
      updated[updated.length - 1] = updater(updated[updated.length - 1]);
      return updated;
    });
  }

  // ---------- TEXT: streaming path ----------
  async function handleSendText() {
    const trimmed = inputText.trim();
    if (!trimmed || isStreaming || voiceStage !== "idle") return;
    setInputText("");
    setIsStreaming(true);

    appendMessage({ role: "user", content: trimmed });
    appendMessage({ role: "assistant", content: "", streaming: true });

    const formData = new FormData();
    formData.append("text", trimmed);
    if (sessionId) formData.append("session_id", sessionId);

    try {
      const res = await fetch(`${API_BASE}/api/text/stream`, { method: "POST", body: formData });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split("\n\n");
        buffer = events.pop();

        for (const evt of events) {
          const lines = evt.split("\n");
          const eventType = lines[0]?.replace("event: ", "");
          const dataLine = lines[1]?.replace("data: ", "");
          if (!dataLine) continue;
          const parsed = JSON.parse(dataLine);

          if (eventType === "meta") {
            setSessionId(parsed.session_id);
            setSentiment(parsed.sentiment || "neutral");
            updateLastMessage((m) => ({ ...m, intent: parsed.intent || null }));
          } else if (eventType === "token") {
            updateLastMessage((m) => ({ ...m, content: m.content + parsed.text }));
          } else if (eventType === "done") {
            updateLastMessage((m) => ({
              ...m,
              role: parsed.is_handoff_turn ? "system" : m.role,
              streaming: false,
              audioBase64: parsed.audio_base64 || null,
            }));
            if (parsed.audio_base64) {
              if (activeAudioRef.current) {
                activeAudioRef.current.pause();
              }
              const audio = new Audio(`data:audio/wav;base64,${parsed.audio_base64}`);
              activeAudioRef.current = audio;
              setVoiceStage("speaking");
              audio.play().catch(() => {
                setVoiceStage("idle");
                activeAudioRef.current = null;
              });
              audio.onended = () => {
                setVoiceStage("idle");
                activeAudioRef.current = null;
              };
              audio.onerror = () => {
                setVoiceStage("idle");
                activeAudioRef.current = null;
              };
            }
          }
        }
      }
    } catch (err) {
      updateLastMessage((m) => ({ ...m, content: `Something went wrong: ${err.message}`, streaming: false, role: "system" }));
    } finally {
      setIsStreaming(false);
    }
  }

  // ---------- VOICE: staged status path ----------
  async function handleMicTap() {
    if (voiceStage === "idle") {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
        const options = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? { mimeType: "audio/webm;codecs=opus" }
          : {};
        const mediaRecorder = new MediaRecorder(stream, options);
        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (e) => audioChunksRef.current.push(e.data);
        mediaRecorder.onstop = () => {
          const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
          stream.getTracks().forEach((track) => track.stop());
          sendVoice(blob);
        };

        mediaRecorder.start();
        mediaRecorderRef.current = mediaRecorder;
        setVoiceStage("recording");
      } catch {
        appendMessage({ role: "system", content: "Microphone access was denied or unavailable." });
      }
    } else if (voiceStage === "recording") {
      mediaRecorderRef.current?.stop();
    }
  }

  async function sendVoice(blob) {
    setVoiceStage("transcribing");

    const formData = new FormData();
    formData.append("file", blob, "recording.webm");
    if (sessionId) formData.append("session_id", sessionId);

    const thinkingTimer = setTimeout(() => setVoiceStage("thinking"), 900);

    try {
      const res = await fetch(`${API_BASE}/api/voice`, { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Request failed");
      }
      const data = await res.json();
      clearTimeout(thinkingTimer);
      setVoiceStage("speaking");

      setSessionId(data.session_id);
      setSentiment(data.sentiment || "neutral");

      appendMessage({ role: "user", content: data.transcript });
      appendMessage({
        role: data.is_handoff_turn ? "system" : "assistant",
        content: data.response_text,
        audioBase64: data.response_audio_base64,
        intent: data.intent || null,
      });

      if (data.response_audio_base64) {
        if (activeAudioRef.current) {
          activeAudioRef.current.pause();
        }
        const audio = new Audio(`data:audio/wav;base64,${data.response_audio_base64}`);
        activeAudioRef.current = audio;
        audio.play().catch(() => {
          setVoiceStage("idle");
          activeAudioRef.current = null;
        });
        audio.onended = () => {
          setVoiceStage("idle");
          activeAudioRef.current = null;
        };
        audio.onerror = () => {
          setVoiceStage("idle");
          activeAudioRef.current = null;
        };
      } else {
        setVoiceStage("idle");
      }
    } catch (err) {
      clearTimeout(thinkingTimer);
      appendMessage({ role: "system", content: `Something went wrong: ${err.message}` });
      setVoiceStage("idle");
    }
  }

  function replayAudio(base64) {
    if (activeAudioRef.current) {
      activeAudioRef.current.pause();
    }
    const audio = new Audio(`data:audio/wav;base64,${base64}`);
    activeAudioRef.current = audio;
    setVoiceStage("speaking");
    audio.play().catch(() => {
      setVoiceStage("idle");
      activeAudioRef.current = null;
    });
    audio.onended = () => {
      setVoiceStage("idle");
      activeAudioRef.current = null;
    };
    audio.onerror = () => {
      setVoiceStage("idle");
      activeAudioRef.current = null;
    };
  }

  function handleStopSpeaking() {
    if (activeAudioRef.current) {
      activeAudioRef.current.pause();
      activeAudioRef.current = null;
    }
    setVoiceStage("idle");
  }

  if (!showChat) {
    return <LandingPage onStartChatting={() => setShowChat(true)} />;
  }

  return (
    <div className="app app--chat-enter" id="app-root">
      <div className="app__aurora" aria-hidden="true">
        <span className="app__aurora-blob app__aurora-blob--one" />
        <span className="app__aurora-blob app__aurora-blob--two" />
        <span className="app__aurora-blob app__aurora-blob--three" />
      </div>
      <Header sentiment={sentiment} />
      <ChatArea messages={messages} onReplayAudio={replayAudio} />
      <InputBar
        inputText={inputText}
        onInputChange={setInputText}
        onSendText={handleSendText}
        onMicTap={handleMicTap}
        onStopSpeaking={handleStopSpeaking}
        voiceStage={voiceStage}
        isStreaming={isStreaming}
      />
    </div>
  );
}