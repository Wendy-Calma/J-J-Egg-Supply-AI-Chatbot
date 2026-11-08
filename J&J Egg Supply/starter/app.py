"""
J&J Egg Supply AI Chatbot - Backend (Flask)
--------------------------------------------
One backend, two entry points:
  - POST /chat        -> used by the web chat widget (static/index.html)
  - GET/POST /webhook -> used by Facebook Messenger

Both entry points call the exact same handle_message() function
(load prices -> build prompt -> call Ollama -> handle order), so the
knowledge base, prompt rules, and order logic only exist in one place.
Only this backend talks to Ollama; the browser never does.
"""

import os
import re
import io
import csv
import json
import glob
import time
import uuid
import sqlite3
import threading
from datetime import datetime

import requests
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")

# Ollama runs locally and speaks the OpenAI API at /v1, so we keep the
# OpenAI SDK and just point it at Ollama. The api_key is required by the
# SDK but ignored by Ollama.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT_SECONDS = 120  # loading the model from disk can take a while
TEMPERATURE = 0.3

client = OpenAI(
    api_key="ollama",
    base_url=f"{OLLAMA_URL}/v1",
    timeout=OLLAMA_TIMEOUT_SECONDS,
    max_retries=0,
)



def warm_up_ollama():
    """Loads the model into memory at startup so the first customer
    doesn't wait for it (runs in the background, errors are ignored)."""
    try:
        client.chat.completions.create(
            model=OLLAMA_MODEL, max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        print(f"Ollama ready ({OLLAMA_MODEL})")
    except Exception as e:
        print("Ollama warm-up failed (is Ollama running?):", repr(e))


threading.Thread(target=warm_up_ollama, daemon=True).start()

DB_PATH = os.environ.get("DB_PATH", "jj_egg.db")
HISTORY_LIMIT = 10        # only the last ~10 messages are sent to the model
BULK_ORDER_TRAYS = 12     # orders this big are referred to staff

# ---------------------------------------------------------------------------
# Knowledge base loading (FR-01 to FR-03, section 6 of the SRS)
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


# ---------------------------------------------------------------------------
# Live pricing from a published Google Sheet
# ---------------------------------------------------------------------------
# Prices change often, so the owner edits them in a Google Sheet instead of
# kb/pricing.json. Set PRICING_SHEET_CSV_URL to the sheet's "Publish to web"
# CSV link. kb/pricing.json is still used as a fallback if the sheet is not
# configured or can't be reached.

PRICING_SHEET_CSV_URL = os.environ.get("PRICING_SHEET_CSV_URL", "").strip()
PRICING_CACHE_SECONDS = 300  # re-check the sheet at most every 5 minutes

_pricing_cache = {"data": None, "checked_at": 0.0}


def _format_price(raw):
    """Turns '238', '238.00', '₱238' or '1,238' into '₱238.00'.
    Returns None if the value isn't a number."""
    cleaned = re.sub(r"[^\d.]", "", raw or "")
    try:
        return f"₱{float(cleaned):.2f}"
    except ValueError:
        return None


def fetch_live_pricing():
    """Downloads the pricing sheet as CSV. Returns a list of entries shaped
    like kb/pricing.json, or None if anything goes wrong."""
    try:
        resp = requests.get(PRICING_SHEET_CSV_URL, timeout=5)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        rows = csv.DictReader(io.StringIO(resp.text))
        entries = []
        for row in rows:
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            product = row.get("product")
            price = _format_price(row.get("price"))
            if not product or not price:
                continue
            entry = {"product": product, "price": price}
            for field in ("unit", "size_range", "last_updated"):
                if row.get(field):
                    entry[field] = row[field]
            entries.append(entry)
        return entries or None
    except Exception as e:
        print("Pricing sheet fetch error:", e)
        return None


def get_pricing():
    """Current prices: from the Google Sheet (cached), else kb/pricing.json."""
    fallback = KB.get("pricing", [])
    if not PRICING_SHEET_CSV_URL:
        return fallback

    now = time.time()
    # Only hit the sheet once per PRICING_CACHE_SECONDS, even if it failed,
    # so a broken sheet doesn't slow down every chat message.
    if now - _pricing_cache["checked_at"] >= PRICING_CACHE_SECONDS:
        _pricing_cache["checked_at"] = now
        live = fetch_live_pricing()
        if live:
            _pricing_cache["data"] = live
    # If the sheet has never loaded (or is broken), use kb/pricing.json.
    return _pricing_cache["data"] or fallback



def get_stock():
    return KB.get("stock", [])


def staff_contact():
    """Staff number: STAFF_CONTACT in .env, else the number in kb/ordering.json."""
    env_value = os.environ.get("STAFF_CONTACT", "").strip()
    if env_value:
        return env_value
    for entry in KB.get("ordering", []):
        match = re.search(r"09\d{9}", entry.get("contact", ""))
        if match:
            return match.group(0)
    return "our Facebook Page"


def payment_methods():
    for entry in KB.get("ordering", []):
        if entry.get("payment_methods"):
            return entry["payment_methods"]
    return []


def store_name():
    for entry in KB.get("business", []):
        if entry.get("name"):
            return entry["name"]
    return "J&J Egg Supply"


def ollama_down_message():
    return (
        "Pasensya na po, may problema po ang aming system ngayon. "
        f"Pakitawagan o i-text po ang aming staff sa {staff_contact()}. Salamat po!\n"
        "Sorry, our chat assistant is having trouble right now. "
        f"Please call or text our staff at {staff_contact()}. Thank you!"
    )


def unsure_message():
    return (
        "Pasensya na po, hindi po ako sigurado diyan. "
        f"Pakitanong po sa aming staff sa {staff_contact()}.\n"
        "Sorry, I'm not sure about that. "
        f"Please contact our staff at {staff_contact()}."
    )


# ---------------------------------------------------------------------------
# Database: chat history + orders (SQLite, one file, no server needed)
# ---------------------------------------------------------------------------

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                product TEXT NOT NULL,
                trays INTEGER NOT NULL,
                unit_price REAL NOT NULL,
                fulfillment TEXT NOT NULL,
                delivery_area TEXT,
                delivery_fee REAL NOT NULL,
                total REAL NOT NULL,
                customer_name TEXT NOT NULL,
                contact TEXT NOT NULL,
                address TEXT,
                payment_method TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'new',
                raw_json TEXT NOT NULL
            );
        """)


init_db()


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def save_message(session_id, role, content):
    with db() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now_iso()),
        )


def recent_messages(session_id, limit=HISTORY_LIMIT):
    with db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ---------------------------------------------------------------------------
# Prices, stock, and delivery fees as numbers (used for backend math)
# ---------------------------------------------------------------------------

def peso_to_float(text):
    match = re.search(r"\d[\d,]*(?:\.\d+)?", text or "")
    return float(match.group(0).replace(",", "")) if match else None


def find_product(name):
    """Matches 'Medium', 'm', 'Medium Eggs (M)', 'medium eggs' to a pricing entry."""
    wanted = (name or "").strip().lower()
    if not wanted:
        return None
    for entry in get_pricing():
        product = entry["product"].lower()
        letter = re.search(r"\((\w+)\)", product)
        first_word = product.split()[0]
        if wanted in (product, first_word, f"{first_word} eggs") or (letter and wanted == letter.group(1)):
            return entry
    return None


def stock_for(product_name):
    for entry in get_stock():
        if entry.get("product", "").lower() == product_name.lower():
            return entry.get("stock_trays")
    return None


def delivery_rule(area):
    """Returns (fee_per_tray, minimum_trays) for 'within_angeles' / 'outside_angeles'."""
    want_outside = area == "outside_angeles"
    for entry in KB.get("delivery", []):
        coverage = entry.get("coverage", "").lower()
        if coverage.startswith("outside") != want_outside:
            continue
        fee_text = entry.get("fee", "")
        fee = 0.0 if "free" in fee_text.lower() else (peso_to_float(fee_text) or 0.0)
        minimum = peso_to_float(entry.get("minimum_order", "")) or 1
        return fee, int(minimum)
    return None, None


def money(amount):
    return f"₱{amount:,.2f}"


# ---------------------------------------------------------------------------
# System prompt (built fresh for every message so prices/stock are current)
# ---------------------------------------------------------------------------

SKIP_KEYS = {"last_updated", "phrasing_hints", "note"}


def _describe(entry):
    parts = []
    for key, value in entry.items():
        if key in SKIP_KEYS:
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{key.replace('_', ' ')}: {value}")
    return "; ".join(parts)


def build_facts():
    lines = []

    lines.append("PRICES (per tray of 30 pcs):")
    for p in get_pricing():
        size = f" ({p['size_range']})" if p.get("size_range") else ""
        lines.append(f"- {p['product']}{size}: {p['price']} {p.get('unit', '')}".rstrip())

    lines.append("\nSTOCK (trays available now):")
    for s in get_stock():
        trays = s.get("stock_trays", 0)
        status = "OUT OF STOCK" if not trays else f"{trays} trays"
        lines.append(f"- {s['product']}: {status}")

    lines.append("\nDELIVERY:")
    for d in KB.get("delivery", []):
        lines.append(f"- {_describe(d)}")

    lines.append("\nPAYMENT METHODS: " + ", ".join(payment_methods()))
    lines.append(f"STAFF CONTACT NUMBER: {staff_contact()}")

    lines.append("\nSTORE INFO:")
    for b in KB.get("business", []):
        for key in ("name", "address", "hours_note", "walk_in_note", "facebook_page", "owners"):
            if b.get(key):
                value = ", ".join(b[key]) if isinstance(b[key], list) else b[key]
                lines.append(f"- {key.replace('_', ' ')}: {value}")

    lines.append("\nPOLICIES:")
    for p in KB.get("policies", []):
        lines.append(f"- {_describe(p)}")

    # kb/promotions.json is intentionally NOT included: discounts are handled
    # by staff only, so the bot never promises one.

    return "\n".join(lines)


def build_examples():
    """Few-shot examples built from live data so they never show stale prices."""
    examples = []
    pricing = get_pricing()
    if pricing:
        p = pricing[-1]
        examples.append(
            f"Customer: Magkano po ang isang tray ng {p['product'].split()[0]}?\n"
            f"Assistant: Ang {p['product']} po ay {p['price']} per tray of 30 pcs. "
            "Ilang tray po ang kailangan ninyo?"
        )
    outside = next((d for d in KB.get("delivery", []) if d.get("coverage", "").lower().startswith("outside")), None)
    if outside:
        minimum = f" {outside['minimum_order']}." if outside.get("minimum_order") else ""
        examples.append(
            "Customer: Do you deliver outside Angeles City?\n"
            f"Assistant: Yes! Delivery outside Angeles City: {outside['fee']}.{minimum} "
            "Would you like to place an order?"
        )
    methods = payment_methods()
    if methods:
        examples.append(
            "Customer: Pa-deliver po sana. Paano po magbayad?\n"
            f"Assistant: Pwede po kayong magbayad sa {' o '.join(methods)}. "
            "Ano pong size at ilang tray ang order ninyo?"
        )
    examples.append(
        "Customer: May discount po ba kung 20 trays?\n"
        "Assistant: Para po sa bulk orders at discounts, ang aming staff po ang tutulong sa inyo. "
        f"Pakitawagan o i-text po sila sa {staff_contact()}. Salamat po!"
    )
    examples.append(
        "Customer: What's the weather today?\n"
        f"Assistant: Sorry, I can only help with questions about {store_name()}'s eggs, "
        "prices, delivery, and orders. Is there anything about our eggs I can help you with?"
    )
    return "\n\n".join(examples)


SYSTEM_PROMPT = """You are the customer service assistant of {store}, a small egg supply store. \
You help customers with egg prices, stock, delivery, payment, and taking orders.

LANGUAGE
- Look at the customer's LATEST message and reply in that language: English -> English only. Tagalog -> Tagalog. Taglish (mixed) -> Taglish.
- In Tagalog or Taglish, be polite and use "po" and "opo".
- Never translate egg size names. Write them exactly as listed (e.g. "Medium Eggs (M)").
- Always write prices with the ₱ sign (e.g. ₱238.00).

TAGALOG WORDS CUSTOMERS USE
- itlog = egg
- tray / bandehado = one tray of 30 eggs
- kaha = case/box (ask how many trays they mean)
- magkano = how much
- meron pa ba = is there still any / is it in stock
- pa-deliver = please deliver

FACTS (the ONLY information you may use)
{facts}

RULES
1. Use ONLY the facts above. Never invent prices, discounts, stock, fees, delivery schedules, or any other detail. If the answer is not in the facts, say you are not sure and give the staff contact number {contact}.
2. If a size is OUT OF STOCK, say so and suggest a size that is in stock. Never accept more trays than are in stock.
3. Never promise, deny, or apply discounts. If the customer asks about a discount or promo, wants {bulk} trays or more, tries to negotiate the price, or has a complaint (broken or rotten eggs, late or wrong delivery, refunds), do not decide it yourself: say our staff will help and give the number {contact}.
4. Only talk about the store and its eggs. Politely redirect unrelated questions.
5. Keep replies short: 1 to 4 sentences.
6. For delivery, ask whether the address is within Angeles City, then tell the customer that area's delivery fee and minimum order (if any) before continuing. Do not accept a delivery order below the minimum.

TAKING AN ORDER
Collect these details, asking for only one or two at a time:
egg size, number of trays, delivery or pickup, customer name, contact number, delivery address (delivery only; ask whether it is within Angeles City), payment method.
Then show an order summary with the total (price per tray x trays, plus the delivery fee if delivery) and ask the customer to confirm.
Only AFTER you have shown the summary AND the customer clearly confirms it (e.g. "yes", "opo", "confirm"), end your reply with this line, on its own line, as valid JSON, filled in ONLY with details the customer actually gave you:
[ORDER] {{"size": "<exact size name>", "trays": <number>, "fulfillment": "<delivery or pickup>", "delivery_area": "<within_angeles or outside_angeles, null for pickup>", "name": "<customer's name>", "contact": "<customer's number>", "address": "<delivery address, empty for pickup>", "payment": "<payment method>"}}
Never write the [ORDER] line before the customer confirms, never write it twice for the same order, and never mention it to the customer.

EXAMPLES
{examples}
"""


def build_system_prompt():
    return SYSTEM_PROMPT.format(
        store=store_name(),
        facts=build_facts(),
        contact=staff_contact(),
        bulk=BULK_ORDER_TRAYS,
        examples=build_examples(),
    )


# ---------------------------------------------------------------------------
# Grounding check (AI-07): reject peso amounts that aren't from the facts
# ---------------------------------------------------------------------------

def _peso_amounts(text):
    return [float(m.replace(",", "")) for m in re.findall(r"₱\s?(\d[\d,]*(?:\.\d+)?)", text or "")]


def looks_ungrounded(answer, facts, user_message=""):
    """True if the answer mentions a ₱ amount that is neither in the facts /
    the customer's message nor a valid order amount (price x trays,
    fee x trays, or (price + fee) x trays)."""
    amounts = _peso_amounts(answer)
    if not amounts:
        return False
    known = _peso_amounts(facts) + _peso_amounts(user_message)
    prices = [peso_to_float(p["price"]) for p in get_pricing()]
    fees = [f for f in (delivery_rule("within_angeles")[0], delivery_rule("outside_angeles")[0]) if f]
    # (unit amount, max trays). Fee-only totals are small, so they get a
    # tight cap; otherwise a ₱5 fee would make any multiple of 5 "valid".
    max_trays = max([BULK_ORDER_TRAYS] + [s.get("stock_trays") or 0 for s in get_stock()])
    units = [(u, max_trays) for u in prices + [p + f for p in prices for f in fees] if u]
    units += [(f, BULK_ORDER_TRAYS) for f in fees]

    def ok(amount):
        if any(abs(amount - k) < 0.01 for k in known):
            return True
        for unit, limit in units:
            n = round(amount / unit)
            if 1 <= n <= limit and abs(n * unit - amount) < 0.01:
                return True
        return False

    return not all(ok(a) for a in amounts)


# ---------------------------------------------------------------------------
# Order handling: detect [ORDER] line, validate, recalculate, save
# ---------------------------------------------------------------------------

def extract_order(reply):
    """Returns (reply_without_order_line, order_dict_or_None, had_order_tag)."""
    idx = reply.find("[ORDER]")
    if idx == -1:
        return reply, None, False

    visible = reply[:idx]
    rest = reply[idx + len("[ORDER]"):]
    order, trailing = None, ""
    try:
        start = rest.index("{")
        order, end = json.JSONDecoder().raw_decode(rest[start:])
        if not isinstance(order, dict):
            order = None
        trailing = rest[start + end:]
    except ValueError:  # no "{" or invalid JSON (JSONDecodeError is a ValueError)
        order = None
    visible = (visible + trailing).replace("```json", "").replace("```", "").strip()
    return visible, order, True


CONFIRM_WORDS = re.compile(
    r"\b(yes|yep|yup|opo|oo|oho|confirm|confirmed|sige|go|ok|okay|tama|correct|g)\b", re.I
)


def is_confirmation(text):
    return bool(CONFIRM_WORDS.search(text or ""))


def details_came_from_customer(order, session_id):
    """The name and number in the order must appear in what the customer
    typed, so the model can't invent or copy them."""
    said = " ".join(m["content"] for m in recent_messages(session_id, limit=50) if m["role"] == "user")
    digits = re.sub(r"\D", "", str(order.get("contact", "")))
    name_words = str(order.get("name", "")).lower().split()
    return (bool(digits) and digits in re.sub(r"\D", "", said)
            and bool(name_words) and name_words[0] in said.lower())


def validate_and_price(order):
    """Recalculates the total from backend prices. Returns (record, error)."""
    product = find_product(str(order.get("size", "")))
    if not product:
        return None, "unknown egg size"
    unit_price = peso_to_float(product["price"])

    try:
        trays = int(order.get("trays"))
    except (TypeError, ValueError):
        return None, "invalid number of trays"
    if trays < 1:
        return None, "invalid number of trays"
    if trays >= BULK_ORDER_TRAYS:
        return None, f"bulk orders ({BULK_ORDER_TRAYS}+ trays) are handled by our staff"
    stock = stock_for(product["product"])
    if stock is not None and trays > stock:
        return None, f"only {stock} trays of {product['product']} are in stock"

    fulfillment = str(order.get("fulfillment", "")).strip().lower()
    if fulfillment not in ("delivery", "pickup"):
        return None, "missing delivery or pickup choice"

    name = str(order.get("name", "") or "").strip()
    contact = str(order.get("contact", "") or "").strip()
    address = str(order.get("address", "") or "").strip()
    if not name or not contact:
        return None, "missing name or contact number"

    methods = payment_methods()
    payment = str(order.get("payment", "") or "").strip()
    match = next((m for m in methods if m.lower() == payment.lower()), None)
    if methods and not match:
        return None, f"payment must be one of: {', '.join(methods)}"

    delivery_area, fee_total = None, 0.0
    if fulfillment == "delivery":
        delivery_area = str(order.get("delivery_area") or "").strip().lower()
        if delivery_area not in ("within_angeles", "outside_angeles"):
            return None, "unknown delivery area"
        if not address:
            return None, "missing delivery address"
        fee_per_tray, minimum = delivery_rule(delivery_area)
        if fee_per_tray is None:
            return None, "delivery is not available for that area"
        if trays < minimum:
            return None, f"delivery to that area needs at least {minimum} trays"
        fee_total = fee_per_tray * trays

    record = {
        "product": product["product"],
        "trays": trays,
        "unit_price": unit_price,
        "fulfillment": fulfillment,
        "delivery_area": delivery_area,
        "delivery_fee": fee_total,
        "total": unit_price * trays + fee_total,
        "customer_name": name,
        "contact": contact,
        "address": address,
        "payment_method": match or payment,
    }
    return record, None


def save_order(session_id, record, raw_order):
    with db() as conn:
        # Guard against the model repeating the same [ORDER] line.
        dup = conn.execute(
            "SELECT id FROM orders WHERE session_id = ? AND product = ? AND trays = ? "
            "AND contact = ? AND created_at >= ?",
            (session_id, record["product"], record["trays"], record["contact"],
             datetime.fromtimestamp(time.time() - 1800).isoformat(timespec="seconds")),
        ).fetchone()
        if dup:
            return dup["id"], False
        cur = conn.execute(
            """INSERT INTO orders (session_id, created_at, product, trays, unit_price,
               fulfillment, delivery_area, delivery_fee, total, customer_name, contact,
               address, payment_method, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, now_iso(), record["product"], record["trays"], record["unit_price"],
             record["fulfillment"], record["delivery_area"], record["delivery_fee"],
             record["total"], record["customer_name"], record["contact"], record["address"],
             record["payment_method"], json.dumps(raw_order, ensure_ascii=False)),
        )
        return cur.lastrowid, True


def order_confirmation(order_id, record):
    fee = f" + delivery fee {money(record['delivery_fee'])}" if record["delivery_fee"] else ""
    return (
        f"✅ Order #{order_id} recorded: {record['trays']} tray(s) {record['product']} "
        f"({money(record['unit_price'])} each{fee}). Total: {money(record['total'])}, "
        f"{record['fulfillment']}, {record['payment_method']}. "
        f"Our staff will contact you at {record['contact']} to confirm. Salamat po!"
    )


def order_summary(record, language):
    """Backend-computed summary, used when the model's own summary was wrong."""
    subtotal = record["unit_price"] * record["trays"]
    fee = f" + delivery fee {money(record['delivery_fee'])}" if record["delivery_fee"] else ""
    where = f"delivery to {record['address']}" if record["fulfillment"] == "delivery" else "pickup"
    details = (
        f"{record['trays']} tray(s) {record['product']} x {money(record['unit_price'])} = "
        f"{money(subtotal)}{fee}. Total: {money(record['total'])} ({where}, "
        f"{record['payment_method']}). {record['customer_name']}, {record['contact']}."
    )
    if language == "english":
        return f"Order summary: {details} Reply \"yes\" to confirm."
    return f"Order summary po: {details} Paki-reply po ng \"opo\" para i-confirm."


# ---------------------------------------------------------------------------
# Language detection (small models drift into Tagalog, so we remind them)
# ---------------------------------------------------------------------------

TAGALOG_WORDS = {
    "po", "opo", "ang", "ng", "mga", "sa", "na", "ba", "pa", "ko", "ako", "kayo",
    "ninyo", "nyo", "niyo", "namin", "natin", "magkano", "meron", "mayroon", "wala",
    "salamat", "itlog", "lang", "naman", "yung", "sana", "pwede", "puwede", "paano",
    "saan", "ilan", "ilang", "oo", "hindi", "din", "rin", "kasi", "gusto", "bili",
    "bibili", "pa-deliver", "padeliver", "pa-order", "isang", "dalawa", "tatlo",
    "bandehado", "kaha", "sige", "tig", "ano", "kailan", "pakidala",
}
ENGLISH_WORDS = {
    "the", "i", "i'm", "want", "do", "does", "you", "have", "how", "much", "is",
    "are", "my", "for", "please", "can", "what", "when", "where", "of", "and", "to",
    "would", "like", "it", "yes", "thanks", "thank", "your", "there", "any", "need",
    "will", "much", "price", "deliver", "hi", "hello",
}
LANGUAGE_HINTS = {
    "english": "English only (no Tagalog words, no \"po\").",
    "tagalog": "Tagalog, polite, with \"po\"/\"opo\".",
    "taglish": "Taglish (mix of Tagalog and English), polite, with \"po\".",
}


def detect_language(text):
    words = set(re.findall(r"[a-z][a-z'-]*", (text or "").lower()))
    tl, en = len(words & TAGALOG_WORDS), len(words & ENGLISH_WORDS)
    if tl and en:
        return "taglish" if tl <= en * 2 else "tagalog"
    if tl:
        return "tagalog"
    if en:
        return "english"
    return None  # e.g. just a name and number


def reply_language(session_id):
    """Language of the customer's most recent message that has one."""
    for m in reversed(recent_messages(session_id)):
        if m["role"] == "user":
            language = detect_language(m["content"])
            if language:
                return language
    return None


def order_problem(reason):
    return (
        f"Pasensya na po, hindi ko po ma-save ang order ({reason}). "
        f"Sorry, I couldn't save the order ({reason}). "
        f"You can also contact our staff at {staff_contact()}."
    )


# ---------------------------------------------------------------------------
# The one reusable chat function (web chat and Messenger both call this)
# ---------------------------------------------------------------------------

CJK_CHARS = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def call_ollama(messages):
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE,
        messages=messages,
    )
    return (response.choices[0].message.content or "").strip()


def handle_message(session_id, text):
    """load prices -> build prompt -> call Ollama -> handle order -> reply text."""
    save_message(session_id, "user", text)

    system_prompt = build_system_prompt()
    language = reply_language(session_id)
    if language:
        system_prompt += f"\nREPLY LANGUAGE FOR THIS MESSAGE: {LANGUAGE_HINTS[language]}\n"
    messages = [{"role": "system", "content": system_prompt}] + recent_messages(session_id)

    try:
        raw_reply = call_ollama(messages)
        if CJK_CHARS.search(raw_reply):
            # Qwen sometimes slips into Chinese: retry once with a reminder.
            print("Non-Filipino/English reply, retrying:", raw_reply)
            reminder = {"role": "system", "content":
                        "Reply ONLY in English, Tagalog, or Taglish. Never use Chinese."}
            raw_reply = call_ollama(messages + [reminder])
            if CJK_CHARS.search(raw_reply):
                raw_reply = unsure_message()
    except Exception as e:  # Ollama down, timeout, model missing, etc.
        print("Ollama error:", repr(e))
        return ollama_down_message()

    reply, order, had_tag = extract_order(raw_reply)

    # Check the order the model wrote, using backend prices and rules.
    record, error = None, None
    if order is not None:
        record, error = validate_and_price(order)
        if not error and not details_came_from_customer(order, session_id):
            error = "please send your name and contact number first"
    elif had_tag:
        error = "the order details were incomplete"

    ungrounded = looks_ungrounded(reply, system_prompt, text)
    if ungrounded:
        print("Ungrounded reply replaced:", reply)
        reply = ""

    if had_tag and is_confirmation(text):
        # Customer confirmed: save the order with the backend's total.
        if error:
            reply = (reply + "\n\n" + order_problem(error)).strip()
        else:
            order_id, is_new = save_order(session_id, record, order)
            if is_new or not reply:  # don't repeat the confirmation every turn
                if is_new:
                    print(f"New order #{order_id}: {record}")
                reply = (reply + "\n\n" + order_confirmation(order_id, record)).strip()
    else:
        if had_tag:
            # The model jumped ahead before the customer confirmed: don't save.
            print("Ignored [ORDER] line before customer confirmation:", order)
        if not reply and record and not error:
            # Model's summary was empty or had wrong math: show the backend's.
            reply = order_summary(record, reply_language(session_id))

    if not reply:
        reply = unsure_message()

    save_message(session_id, "assistant", reply)
    return reply


# ---------------------------------------------------------------------------
# Web chat entry point (for your own demo -- J&J has no website, see SRS s.13)
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    question = str(data.get("message", "")).strip()
    # The browser keeps a random session id so each visitor has their own history.
    session_id = re.sub(r"[^A-Za-z0-9_-]", "", str(data.get("session_id", "")))[:64]
    session_id = session_id or uuid.uuid4().hex

    if not question:
        return jsonify({"reply": "Please type a question!", "session_id": session_id})

    try:
        answer = handle_message("web:" + session_id, question)
    except Exception as e:  # NFR-07: fail safely, never crash the chat
        print("Chat error:", repr(e))
        answer = ollama_down_message()

    return jsonify({"reply": answer, "session_id": session_id})


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
                    answer = handle_message(f"messenger:{sender_id}", message["text"])
                except Exception as e:
                    print("Messenger error:", repr(e))
                    answer = ollama_down_message()
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
