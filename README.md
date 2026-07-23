# 🎧 VoiceCart — Sentiment-Aware Voice Support Agent

VoiceCart is a full-stack, agentic customer support system built for **ShopNest Pulse** — a fictional wireless earbuds brand. It handles product inquiries, orders, warranty claims, and emotionally-aware escalation, entirely through natural conversation — by text or by voice.

Built with **LangGraph**, grounded in a **hybrid-search RAG pipeline**, and wired into real backend infrastructure (MongoDB, Slack, email), this project is an end-to-end demonstration of production-style agentic design: multi-turn state machines, sentiment-driven escalation, deterministic tool-calling, and anti-hallucination guardrails.

---

## 📐 Architecture Overview

```
User (voice or text)
      │
      ▼
┌─────────────────┐      ┌──────────────────────┐
│  STT (Whisper)   │      │   React Frontend      │
└─────────────────┘      └──────────────────────┘
      │                             │
      ▼                             ▼
┌───────────────────────────────────────────────┐
│              FastAPI Backend                    │
│   /api/voice   /api/text/stream   /api/session  │
└───────────────────────────────────────────────┘
      │
      ▼
┌───────────────────────────────────────────────┐
│           LangGraph State Machine               │
│  classify → route → [node] → response           │
└───────────────────────────────────────────────┘
      │
      ├── Pinecone (hybrid RAG) ── Cohere embeddings
      ├── MongoDB Atlas (orders, tickets, claims, inventory)
      ├── Resend (email notifications)
      ├── Slack (3 channel webhooks)
      └── Cartesia / Deepgram (TTS)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| 🧠 LLM | Groq — Llama 3.3 70B & Llama 3.1 8B Instant |
| 🎙️ Speech-to-Text | Groq Whisper (`whisper-large-v3`) |
| 🔊 Text-to-Speech | Cartesia (`sonic-turbo`) with Deepgram Aura-2 fallback |
| 🔍 Embeddings | Cohere `embed-english-v3.0` (1024-dim) |
| 📚 Vector Database | Pinecone (hybrid dense + BM25 search, dotproduct metric) |
| 🕸️ Orchestration | LangGraph |
| 📊 Observability | LangSmith |
| ⚙️ Backend | FastAPI |
| 🗄️ Database | MongoDB Atlas |
| 📧 Email | Resend |
| 💬 Notifications | Slack Webhooks |
| 🎨 Frontend | React (Vite) |

---

## ✅ What's Done

### 📚 Knowledge Base & RAG
- ✅ Hybrid search (dense + BM25) via Pinecone, tuned `alpha=0.75`
- ✅ 5 namespaces, **79 chunks** across product-info, usage-guidance, troubleshooting, policies, and limitations
- ✅ Retrieval score threshold tuned to **0.25** through iterative real-query testing
- ✅ Anti-hallucination grounding — answers only from retrieved context, honest "I don't know" fallback otherwise

### 🕸️ LangGraph Nodes
| Node | Responsibility |
|---|---|
| `classify_node` | Intent, sentiment, namespace, and field extraction for every turn |
| `inquiry_node` | RAG-grounded question answering, supports multi-topic questions |
| `empathetic_response_node` | Sentiment-aware troubleshooting responses |
| `escalate_handoff_node` / `escalation_followup_node` | 3-strike same-issue escalation → email consent → ticket creation |
| `post_escalation_node` | One-time acknowledgment, then resumes normal conversation |
| `other_node` | Greetings, small talk, name memory, out-of-scope handling |
| `check_stock_node` | Live MongoDB-backed inventory checks |
| `order_booking_node` | Adaptive multi-turn order collection, address memory, stock validation |
| `order_tracking_node` | Deterministic order lookup by ID |
| `order_cancel_node` | Cancellation with inventory restoration |
| `warranty_claim_node` / `warranty_email_node` | Adaptive claim collection, 12-month validation, mandatory email |

### 🛒 Order System
- ✅ Single-message booking (all details at once) or adaptive follow-up
- ✅ Real-time stock validation before booking
- ✅ Inventory decrement on booking, restoration on cancellation
- ✅ Multiple independent orders per session (correct state reset)
- ✅ Shipping address memory — resolves "same address as before" with explicit confirmation
- ✅ Order tracking with malformed-ID detection
- ✅ Cash on Delivery (COD) payment messaging
- ✅ Slack notification on every booking/cancellation

### 🛡️ Warranty System
- ✅ **Explicit-request-only** trigger — no inferred eligibility from symptom severity
- ✅ Adaptive field collection that reads conversation history (not just the current message)
- ✅ Deterministic 12-month ownership validation — parses duration, rejects out-of-window claims
- ✅ Mandatory customer email collection before claim finalization
- ✅ Terminal-state reset — supports multiple independent claims per session

### 📈 Escalation System
- ✅ **Same-issue-streak detection** — escalates only when one specific problem goes unresolved for 3 consecutive turns (not just 3 unrelated complaints in a row)
- ✅ Email consent flow before creating a support ticket
- ✅ LLM-generated ticket summaries for the support team
- ✅ One-time post-escalation acknowledgment, then normal conversation resumes

### 🔌 Integrations
- ✅ MongoDB Atlas — `inventory`, `orders`, `tickets`, `warranty_claims` collections
- ✅ Resend — transactional emails for tickets and warranty claims
- ✅ Slack — 3 dedicated channels (`#orders`, `#support-tickets`, `#warranty-claims`)

### 🎙️ Voice Pipeline
- ✅ Groq Whisper transcription
- ✅ Cartesia TTS (primary), client-level timeout to prevent hangs
- ✅ Deepgram TTS (fallback via direct REST call), independently timeout-bounded
- ✅ Graceful degradation on quota/rate-limit errors

### 💻 Frontend
- ✅ React chat interface with streaming responses (SSE)
- ✅ Voice input and playback
- ✅ Sentiment-aware UI styling for handoff/escalation turns

---

## 🚧 What's Remaining

- ⏳ **Order confirmation email to customer** — currently skipped; Resend's sandbox domain can only send to the account owner's own verified email, not arbitrary customer addresses. Needs either a verified custom domain or a Gmail API integration to unblock.
- ⏳ **Deployment** — backend (Render), frontend (Vercel), and production environment configuration not yet done.
- ⏳ **Session persistence** — sessions currently live in an in-memory dict in `main.py`; a `sessions` MongoDB collection is scaffolded but not yet wired in, so session state is lost on server restart.
- ⏳ **Minor KB gaps** — a small number of very specific phrasings (e.g. unusual troubleshooting complaints not covered in the current 79 chunks) can still fall back to "I don't have that information" — expected and honest behavior, but content coverage can keep growing over time.

---

## 📁 Project Structure

```
Customer-Support/
├── backend/
│   ├── main.py                  # FastAPI app, voice + streaming text endpoints
│   ├── recreate_index.py        # Pinecone index setup
│   ├── requirements.txt
│   ├── .env.example
│   ├── app/
│   │   ├── db.py                 # MongoDB collections
│   │   ├── email_service.py      # Resend integration
│   │   ├── slack_service.py      # Slack webhook integration
│   │   ├── graph/
│   │   │   ├── nodes.py          # All LangGraph node logic
│   │   │   ├── graph.py          # StateGraph wiring + routing
│   │   │   └── voice_pipeline.py # STT/TTS orchestration
│   │   ├── ingest/
│   │   │   ├── chunks.py         # PDF → chunk parser
│   │   │   ├── upload.py         # Embeds + upserts to Pinecone
│   │   │   ├── seed_inventory.py
│   │   │   └── *.pdf             # Knowledge base source documents
│   │   └── retrieval/
│   │       └── retriever.py      # Hybrid search implementation
└── frontend/
    └── src/components/            # Chat UI (React)
```

---

## 🚀 Getting Started

```bash
# Backend
cd backend
pip install -r requirements.txt --break-system-packages
cp .env.example .env   # fill in your real API keys
python recreate_index.py
python app/ingest/upload.py
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

---

## 📝 License

This is a personal / portfolio project built for learning and demonstration purposes.