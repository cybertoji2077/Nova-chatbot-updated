---
title: NOVA AI Chat
emoji: ✦
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
---

# NOVA — AI Chat Interface

A full-featured AI chatbot with:
- 💬 Streaming chat responses
- 🎨 AI image generation (FLUX via Hugging Face)
- 🔍 Web search toggle
- 🎤 Voice input
- 💾 Persistent chat history
- 📥 Download chat logs
- 🖼️ Custom bot name & avatar

## Setup

Set these in your Space's **Settings → Variables and Secrets**:

| Secret | Value |
|--------|-------|
| `OPENROUTER_KEY` | Your OpenRouter API key |
| `HF_TOKEN` | Your Hugging Face token |
