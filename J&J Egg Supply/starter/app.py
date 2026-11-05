"""
J&J Egg Supply AI Chatbot - Backend (Flask)
--------------------------------------------
One backend, two entry points:
  - POST /chat      -> used by the web chat widget (static/index.html)
  - GET/POST /webhook -> used by Facebook Messenger

Both entry points call the exact same ask_luna() function, so the
knowledge base, retrieval, and grounding rules only exist in one place.
"""

import os
import re
import json
import glob

import requests
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")
client = OpenAI(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1",
)

# NVIDIA NIM model id (build.nvidia.com) -- verify it's still available
# before your demo, NVIDIA occasionally moves models between catalog tiers.
MODEL = "meta/llama-3.2-11b-vision-instruct"

FALLBACK = (
    "I don't have verified information about that yet. "
    "Please contact J&J Egg Supply directly, or check back soon!"
)


# ---------------------------------------------------------------------------
# Knowledge base loading + retrieval (FR-01 to FR-03, section 6 of the SRS)
# ---------------------------------------------------------------------------

def load_knowledge_base():
    """Loads every kb/*.json file into a dict keyed by filename."""
    kb = {}
    for path in glob.glob(os.path.join("kb", "*.json")):
        category = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            kb[category] = json.load(f)
    return kb


KB = load_knowledge_base()


STOPWORDS = {
    "a", "an", "the", "is", "are", "do", "does", "did", "you", "your", "we",
    "i", "to", "of", "for", "in", "on", "at", "and", "or", "it", "that",
    "this", "what", "how", "much", "many", "can", "have", "has", "will",
    "there", "any", "with", "my", "me", "us", "our", "please", "hi", "hello",
}


def retrieve_context(question):
    """Simple keyword-overlap retrieval across all KB entries.

    This is intentionally NOT a vector database. At the size of your
    knowledge base (a handful of facts), keyword matching retrieves the
    right chunk reliably and is far easier to debug than embeddings.
    Swap this out for a vector store later only if the KB grows a lot.
    """
    question_words = set(re.findall(r"[a-zA-Z0-9]+", question.lower())) - STOPWORDS
    scored = []
    for category, entries in KB.items():
        for entry in entries:
            text = json.dumps(entry).lower()
            score = sum(1 for w in question_words if w in text)
            if score > 0:
                scored.append((score, category, entry))

    scored.sort(key=lambda row: row[0], reverse=True)
    top_matches = [entry for _, _, entry in scored[:5]]

    if not top_matches:
        return None
    return json.dumps(top_matches, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Grounding (maps to AI-01 .. AI-08 and BR-01 .. BR-05 in the SRS)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the customer support assistant for J&J Egg Supply, \
a small student-run egg business.

Rules you must always follow:
1. Only use facts found inside the <context> block below. Never invent \
prices, stock levels, policies, or delivery details that are not in <context>.
2. If the answer is not fully supported by <context>, reply with exactly \
this sentence and nothing else: "{fallback}"
3. Be warm, brief, and clear. This is a chat conversation, not an essay.

<context>
{context}
</context>
"""


def looks_ungrounded(answer, context):
    """Post-generation check (AI-07): reject answers that mention a peso
    amount which does not appear anywhere in the retrieved context.
    This is a deliberately simple check -- extend it if you have time
    (e.g. also check product names, delivery claims, payment methods)."""
    amounts_in_answer = set(re.findall(r"\u20b1\s?\d+(?:\.\d+)?", answer))
    if not amounts_in_answer:
        return False
    context_str = (context or "").replace(" ", "")
    return any(amt.replace(" ", "") not in context_str for amt in amounts_in_answer)


def ask_luna(question):
    context = retrieve_context(question)
    if context is None:
        return FALLBACK

    system_prompt = SYSTEM_PROMPT.format(fallback=FALLBACK, context=context)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,  # AI-08: low temperature to reduce fabrication risk
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    answer = response.choices[0].message.content

    if looks_ungrounded(answer, context):
        return FALLBACK
    return answer


# ---------------------------------------------------------------------------
# Web chat entry point (for your own demo -- J&J has no website, see SRS s.13)
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"reply": "Please type a question!"})

    try:
        answer = ask_luna(question)
    except Exception as e:  # NFR-07: fail safely, never crash the chat
        print("Luna API error:", e)
        answer = "Sorry, I'm having trouble reaching my brain right now -- please try again in a moment."

    return jsonify({"reply": answer})


# ---------------------------------------------------------------------------
# Facebook Messenger entry point (FR-12 -- required delivery channel)
# ---------------------------------------------------------------------------

VERIFY_TOKEN = os.environ.get("MESSENGER_VERIFY_TOKEN", "changeme")
PAGE_ACCESS_TOKEN = os.environ.get("PAGE_ACCESS_TOKEN", "")


@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Meta calls this once, when you first register the webhook URL."""
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return request.args.get("hub.challenge"), 200
    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def receive_message():
    """Meta calls this every time a customer messages the Page."""
    data = request.get_json()

    for entry in data.get("entry", []):
        for event in entry.get("messaging", []):
            sender_id = event["sender"]["id"]
            message = event.get("message", {})

            if "text" in message:
                try:
                    answer = ask_luna(message["text"])
                except Exception as e:
                    print("Luna API error:", e)
                    answer = "Sorry, I'm having trouble right now -- please try again shortly."
                send_message(sender_id, answer)

    return "ok", 200


def send_message(recipient_id, text):
    url = f"https://graph.facebook.com/v21.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    payload = {"recipient": {"id": recipient_id}, "message": {"text": text}}
    resp = requests.post(url, json=payload, timeout=10)
    if resp.status_code != 200:
        print("Messenger send error:", resp.status_code, resp.text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
