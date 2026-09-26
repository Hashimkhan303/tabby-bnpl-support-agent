import streamlit as st
import os
import time
from groq import Groq
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
from datetime import datetime
import os

# ---- Setup (runs once when app starts) ----
API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=API_KEY)

tabby_facts = [
    "Tabby is a BNPL (Buy Now Pay Later) service operating in the UAE.",
    "Tabby allows customers to split payments into 4 interest-free installments.",
    "Tabby does not charge interest or fees for payments made on time.",
    "Tabby accepts payment methods including credit cards, debit cards, and bank transfers in the UAE.",
    "To use Tabby, customers need to be 18+ years old and have a valid UAE mobile number.",
    "Tabby performs a soft credit check that doesn't affect your credit score.",
    "Tabby's approval is based on factors like order value, payment history, and risk assessment.",
    "If a purchase is declined, possible reasons include: insufficient limit, suspicious activity, or eligibility issues.",
    "Customers can check their available balance in the Tabby app under 'My Limit'.",
    "Tabby payments are due in 4 installments: 25% upfront, 25% after 2 weeks, 25% after 4 weeks, 25% after 6 weeks.",
    "Tabby sends payment reminders 48 hours before each installment is due.",
    "If you miss a payment, Tabby may charge a late fee and restrict your account until payment is made.",
    "Customers can contact Tabby support through the app, email at help@tabby.ai, or phone.",
    "Tabby's complaint process: submit via app or email, response within 48 hours.",
    "Tabby uses 3D Secure authentication for additional security on transactions.",
    "Tabby is regulated by the UAE Central Bank and complies with local financial regulations.",
    "Refunds on Tabby purchases are processed to your Tabby account balance within 5-7 business days.",
    "Tabby's 'Share it' feature allows customers to split payments with friends and family.",
    "Tabby offers a loyalty program with rewards for on-time payments.",
    "Tabby partners with major UAE retailers including Amazon, Noon, and Carrefour."
]

vectorizer = TfidfVectorizer(max_features=384)
tfidf_matrix = vectorizer.fit_transform(tabby_facts)
embeddings = tfidf_matrix.toarray()

def get_relevant_context(question, top_k=3):
    question_vector = vectorizer.transform([question]).toarray()[0]
    similarities = cosine_similarity([question_vector], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return " ".join([tabby_facts[i] for i in top_indices])

tools = [
    {
        "type": "function",
        "function": {
            "name": "escalate_to_support",
            "description": "Escalate to human support for account-specific or personal-data questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "enum": ["transaction_status", "payment_schedule",
                               "available_balance", "purchase_declined", "account_verification", "other_account_issue"]},
                    "message": {"type": "string"}
                },
                "required": ["reason", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "answer_from_knowledge_base",
            "description": "Answer general policy questions using the knowledge base",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"]
            }
        }
    }
]

system_prompt_tool = """
You are this business's AI customer support assistant.

RULES:
1. If a question asks about the user's SPECIFIC account, transaction, balance, or personal status — use escalate_to_support.
2. If a question is about GENERAL policies or how the business works — use answer_from_knowledge_base.
3. NEVER guess or infer information not explicitly in the knowledge base.
4. If a question is unrelated to the business's services, politely say you can only help with questions about the business's services and policies.
5. Be helpful, professional, and concise.
"""

def log_interaction(question, response_type, answer, response_time=None, tokens_used=None):
    log_file = "conversation_log.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "response_type", "answer", "response_time_sec", "tokens_used"])
        writer.writerow([datetime.now().isoformat(), question, response_type, answer, response_time, tokens_used])

def get_response(question, history):
    start_time = time.time()
    context = get_relevant_context(question)
    messages = [{"role": "system", "content": system_prompt_tool}] + history + \
               [{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}]

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=500
        )
        response_time = round(time.time() - start_time, 2)
        tokens_used = response.usage.total_tokens if response.usage else None

        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "escalate_to_support":
                reply = f"{args['message']}\n\n📞 Contact Tabby support at help@tabby.ai or through the app."
                log_interaction(question, "escalate", reply, response_time, tokens_used)
                return reply
            else:
                log_interaction(question, "answer", args["answer"], response_time, tokens_used)
                return args["answer"]
        else:
            reply = response.choices[0].message.content
            log_interaction(question, "answer", reply, response_time, tokens_used)
            return reply

    except Exception as e:
        response_time = round(time.time() - start_time, 2)
        reply = "Sorry, I'm having trouble answering right now. Please try again in a moment, or contact support at help@tabby.ai."
        log_interaction(question, "error", str(e), response_time, None)
        return reply
# ---- Streamlit UI ----
st.set_page_config(page_title="AI Support Assistant", page_icon="💬")
st.title("💬 AI Support Assistant — Demo")
st.caption("Sample demo trained on a fintech FAQ. Ask about payments, refunds, or policies. Account-specific questions get routed to human support. I customize this with your business's actual content.")

if os.path.isfile("conversation_log.csv"):
    with open("conversation_log.csv", "rb") as f:
        st.download_button(
            label="Download conversation log (CSV)",
            data=f,
            file_name="conversation_log.csv",
            mime="text/csv"
        )


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = get_response(prompt, st.session_state.messages[:-1])
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
