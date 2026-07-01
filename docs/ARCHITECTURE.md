# Hearth Companion Architecture

Hearth is an emotionally intelligent AI companion built around a strong Agentic Self-Correction Loop and a robust crisis classification mechanism.

## System Architecture

The application is structured into the following main components:
- **Frontend**: A modern glassmorphic web interface.
- **Backend API (Flask)**: Serves the web UI and handles chat generation.
- **LLM Engine (Groq / LoRA)**: Handles text generation.
- **Crisis Classifier**: A parallel DistilBERT model detecting distress signals instantly.

```mermaid
graph TD
    User([User]) -->|Sends Message| UI[Frontend Web Interface]
    UI -->|POST /api/chat| API[Flask Backend API]
    
    API -->|Evaluate Risk| Classifier[DistilBERT Crisis Classifier]
    Classifier -.->|Safe / Concern / Crisis| API
    
    API -->|Draft Response| Groq[Groq API / Local LoRA]
    Groq -.->|Draft 1| API
    
    API -->|Critic Review| GroqCritic[Agentic Critic Loop]
    GroqCritic -.->|Reject & Refine| Groq
    GroqCritic -.->|Approve| API
    
    API -->|Final Message| UI
```

## The Agentic Loop

The core innovation of Hearth is its internal "Draft -> Critic -> Refine" loop. This prevents the typical "helpful assistant" persona of base models from leaking through.

```mermaid
sequenceDiagram
    participant User
    participant Backend
    participant Generator
    participant Critic

    User->>Backend: "I'm so tired of trying."
    Backend->>Generator: Generate Draft (Prompt + Context)
    Generator-->>Backend: Draft: "I'm sorry you feel overwhelmed. Have you tried..."
    
    Backend->>Critic: Does this break the rules? (e.g., giving advice)
    Critic-->>Backend: YES (Rule Broken: Contains advice)
    
    Backend->>Generator: Rewrite this. Do not give advice. Be extremely short.
    Generator-->>Backend: Draft 2: "That sounds exhausting. I'm here."
    
    Backend->>Critic: Does this break the rules?
    Critic-->>Backend: NO (Passes all checks)
    
    Backend-->>User: "That sounds exhausting. I'm here."
```

## Crisis Classifier

Because LLMs can hallucinate or handle high-risk situations poorly, Hearth uses an independent DistilBERT classifier. 
- It is incredibly fast (< 50ms).
- It overrides the standard prompt logic if a crisis is detected.
- It operates completely outside the LLM reasoning layers.
