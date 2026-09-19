# Tabby BNPL Customer Support AI Agent

An AI-powered customer support agent for a Buy Now, Pay Later (BNPL) platform, built to answer customer questions, retrieve relevant knowledge, and escalate complex queries — simulating a real fintech support workflow.

## Features
- Conversational agent powered by Groq (LLM inference)
- Knowledge base search using TF-IDF similarity
- Automatic tool selection and escalation logic for unresolved queries
- Conversation memory across turns
- Error handling for failed or ambiguous queries
- Conversation logging to CSV for analytics
- Streamlit web interface for live interaction

## Tech Stack
- Python
- Groq API (LLM)
- scikit-learn (TF-IDF)
- Streamlit
- Pandas / NumPy

## How It Works
1. User submits a question through the Streamlit UI.
2. The agent searches a knowledge base using TF-IDF similarity to find relevant context.
3. Based on confidence and query type, the agent either answers directly or escalates to a human-support flow.
4. Each interaction (question, response type, timestamp) is logged to a CSV file for later analysis.

## Live Demo
[Try it live](https://tabby-bnpl-support-agent-2e7ccvrxldd5ttvs25drgx.streamlit.app/)

## Project Context
This is Project 1 in a 3-part portfolio series aimed at AI automation / AI engineer roles in the UAE fintech and BNPL sector (Tabby, Tamara). The series progresses from:
1. **This project** — a working AI support agent
2. An observability dashboard analyzing this agent's real usage data
3. A LangGraph-orchestrated, workflow-automated version integrated with n8n

## Author
Hashim — [GitHub](https://github.com/Hashimkhan303)
