"""
J&J Egg Supply AI Chatbot - Backend (Flask)
--------------------------------------------
One backend, two entry points:
  - POST /chat        -> used by the web chat widget (static/index.html)
  - GET/POST /webhook -> used by Facebook Messenger

Both entry points call the exact same handle_message() function
(load prices -> build prompt -> call Gemini -> handle order), so the
knowledge base, prompt rules, and order logic only exist in one place.
Only this backend talks to Gemini; the browser never sees the API key.
"""

import os
import re
import io
import sys
import csv
import json
import glob
import time
import uuid
import sqlite3
from datetime import datetime

import requests
from flask import Flask, request, jsonify, send_from_directory
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Without this, printing "₱" on the Windows console crashes the request.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

app = Flask(__name__, static_folder="static")

# Gemini through its OpenAI-compatible endpoint, so we can use the OpenAI SDK.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Tried in order when the main model is busy, rate-limited, or too slow.
GEMINI_FALLBACK_MODELS = [
    m.strip() for m in
    os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash").split(",")
    if m.strip() and m.strip() != GEMINI_MODEL
]
GEMINI_REASONING_EFFORT = os.environ.get("GEMINI_REASONING_EFFORT", "low")
# A busy model can hang for a minute; give up early and try the next one.
GEMINI_TIMEOUT_SECONDS = 20
TEMPERATURE = 0.3

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set in .env -- the chatbot cannot answer.")

client = OpenAI(
    api_key=GEMINI_API_KEY or "missing",
    base_url=GEMINI_BASE_URL,
    timeout=GEMINI_TIMEOUT_SECONDS,
    max_retries=0,
)

# "Google is busy" errors: worth trying the next model (unlike a bad key).
BUSY_ERRORS = (openai.APITimeoutError, openai.APIConnectionError,
               openai.RateLimitError, openai.InternalServerError)

DB_PATH = os.environ.get("DB_PATH", "jj_egg.db")
HISTORY_LIMIT = 10        # messages of history sent to the model
MAX_MESSAGE_CHARS = 1000  # longer customer messages are cut (protects the Gemini quota)

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
# Products, prices, and stock from a published Google Sheet
# ---------------------------------------------------------------------------

PRICING_SHEET_CSV_URL = os.environ.get("PRICING_SHEET_CSV_URL", "").strip()
# Optional Apps Script web app inside the sheet (see sheet_script.gs): reads the
# sheet live and subtracts stock when an order is saved.
SHEET_SCRIPT_URL = os.environ.get("SHEET_SCRIPT_URL", "").strip()
SHEET_SCRIPT_SECRET = os.environ.get("SHEET_SCRIPT_SECRET", "").strip()
PRICING_CACHE_SECONDS = 60  # re-check the sheet at most once a minute

if not PRICING_SHEET_CSV_URL and not SHEET_SCRIPT_URL:
    print("WARNING: PRICING_SHEET_CSV_URL is not set in .env -- the bot has no prices or stock.")

_pricing_cache = {"data": None, "checked_at": 0.0}


def _format_price(raw):
    """Turns '238', '238.00', '₱238' or '1,238' into '₱238.00'.
    Returns None if the value isn't a number."""
    cleaned = re.sub(r"[^\d.]", "", raw or "")
    try:
        return f"₱{float(cleaned):,.2f}"
    except ValueError:
        return None


def fetch_sheet_rows():
    """The sheet as a list of rows (first row = headers): live from the Apps
    Script if configured, else from the published CSV."""
    if SHEET_SCRIPT_URL:
        resp = requests.get(SHEET_SCRIPT_URL, timeout=10)
        resp.raise_for_status()
        return resp.json()["rows"]
    resp = requests.get(PRICING_SHEET_CSV_URL, timeout=5)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return list(csv.reader(io.StringIO(resp.text)))


def fetch_live_pricing():
    """Downloads the sheet. Returns a list of
    {product, price, unit, size_range, stock} entries, or None on any error."""
    try:
        rows = fetch_sheet_rows()
        headers = [str(h).strip().lower() for h in rows[0]]
        entries = []
        for values in rows[1:]:
            row = {h: str(v).strip() for h, v in zip(headers, values)}
            product = row.get("product")
            price = _format_price(row.get("price"))
            if not product or not price:
                continue
            entry = {"product": product, "price": price}
            for field in ("unit", "size_range", "last_updated"):
                if row.get(field):
                    entry[field] = row[field]
            # Empty cell (or no stock column) = not counted; 0 = out of stock.
            count = re.search(r"\d+", row.get("stock", ""))
            entry["stock"] = int(count.group(0)) if count else None
            entries.append(entry)
        return entries or None
    except Exception as e:
        print("Pricing sheet fetch error:", e)
        return None


def get_pricing():
    """Current products and prices from the Google Sheet (cached). If the sheet
    can't be reached, the last copy that loaded is kept."""
    if not PRICING_SHEET_CSV_URL and not SHEET_SCRIPT_URL:
        return []

    now = time.time()
    # Checked at most once per cache period, even after a failure, so a
    # broken sheet doesn't slow down every message.
    if now - _pricing_cache["checked_at"] >= PRICING_CACHE_SECONDS:
        _pricing_cache["checked_at"] = now
        live = fetch_live_pricing()
        if live:
            _pricing_cache["data"] = live
    return _pricing_cache["data"] or []


def take_stock(items):
    """Subtracts ordered quantities from the sheet's stock column via the
    Apps Script. Returns an error message if the sheet refused (e.g. not
    enough left), else None. Does nothing if no script is configured."""
    if not SHEET_SCRIPT_URL:
        return None
    payload = {"secret": SHEET_SCRIPT_SECRET,
               "items": [{"product": i["product"], "quantity": i["quantity"]} for i in items]}
    try:
        resp = requests.post(SHEET_SCRIPT_URL, json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        # Save the order anyway; staff can fix the count by hand.
        print("Stock update failed (order still saved, update the sheet by hand):", repr(e))
        return None
    _pricing_cache["checked_at"] = 0  # re-read the new stock on the next message
    return result.get("error")


def get_stock():
    """Stock from the sheet's "stock" column (trays for eggs, sacks for rice)."""
    return [{"product": p["product"], "stock": p.get("stock"), "unit": unit_name(p) + "s"}
            for p in get_pricing()]


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


def ai_down_message():
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
        # Older databases had one product per order (product/trays columns).
        if "product" in [r[1] for r in conn.execute("PRAGMA table_info(orders)")]:
            conn.execute("ALTER TABLE orders RENAME TO orders_old")
        old_layout = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'orders_old'"
        ).fetchone() is not None

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
                items_summary TEXT NOT NULL,
                fulfillment TEXT NOT NULL,
                delivery_area TEXT,
                delivery_fee REAL NOT NULL,
                discount REAL NOT NULL DEFAULT 0,
                total REAL NOT NULL,
                customer_name TEXT NOT NULL,
                contact TEXT NOT NULL,
                address TEXT,
                payment_method TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'new',
                raw_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pending_orders (
                session_id TEXT PRIMARY KEY,
                raw_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL REFERENCES orders(id),
                product TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                unit TEXT NOT NULL,
                unit_price REAL NOT NULL,
                line_total REAL NOT NULL
            );
        """)

        if "discount" not in [r[1] for r in conn.execute("PRAGMA table_info(orders)")]:
            conn.execute("ALTER TABLE orders ADD COLUMN discount REAL NOT NULL DEFAULT 0")

        if old_layout:
            for r in conn.execute("SELECT * FROM orders_old").fetchall():
                conn.execute(
                    """INSERT INTO orders (id, session_id, created_at, items_summary, fulfillment,
                       delivery_area, delivery_fee, total, customer_name, contact, address,
                       payment_method, status, raw_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (r["id"], r["session_id"], r["created_at"], f"{r['trays']} tray(s) {r['product']}",
                     r["fulfillment"], r["delivery_area"], r["delivery_fee"], r["total"],
                     r["customer_name"], r["contact"], r["address"], r["payment_method"],
                     r["status"], r["raw_json"]),
                )
                conn.execute(
                    "INSERT INTO order_items (order_id, product, quantity, unit, unit_price, line_total) "
                    "VALUES (?, ?, ?, 'tray', ?, ?)",
                    (r["id"], r["product"], r["trays"], r["unit_price"], r["unit_price"] * r["trays"]),
                )
            conn.execute("DROP TABLE orders_old")


init_db()


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def save_message(session_id, role, content):
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now_iso()),
        )
        return cur.lastrowid


def delete_message(message_id):
    with db() as conn:
        conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))


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
    """Matches 'Medium', 'm', 'Medium Eggs (M)', 'jasmine rice' to a pricing entry."""
    wanted = (name or "").strip().lower()
    if not wanted:
        return None
    pricing = get_pricing()
    for entry in pricing:
        product = entry["product"].lower()
        letter = re.search(r"\((\w+)\)", product)
        first_word = product.split()[0]
        if wanted in (product, first_word, f"{first_word} eggs") or (letter and wanted == letter.group(1)):
            return entry
    # Every word given appears in exactly one product name ("jasmine rice").
    words = set(re.findall(r"\w+", wanted))
    matches = [e for e in pricing if words <= set(re.findall(r"\w+", e["product"].lower()))]
    return matches[0] if len(matches) == 1 else None


def unit_name(product_entry):
    """'sack' or 'tray', from the price list's unit (e.g. 'per sack')."""
    return "sack" if "sack" in product_entry.get("unit", "").lower() else "tray"


def count_units(quantity, unit):
    return f"{quantity} {unit}{'' if quantity == 1 else 's'}"


def stock_for(product_name):
    """Units in stock, or None if the product's stock isn't counted."""
    for entry in get_stock():
        if entry.get("product", "").lower() == product_name.lower():
            return entry.get("stock")
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


def egg_discount_rule():
    """(minimum egg trays, peso discount) from kb/promotions.json, or (None, 0)."""
    for promo in KB.get("promotions", []):
        if promo.get("min_egg_trays") and promo.get("discount"):
            return int(promo["min_egg_trays"]), float(promo["discount"])
    return None, 0.0


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

    lines.append("PRICES:")
    if not get_pricing():
        lines.append("- Prices and stock are unavailable right now. Give the staff contact number.")
    for p in get_pricing():
        size = f" ({p['size_range']})" if p.get("size_range") else ""
        lines.append(f"- {p['product']}{size}: {p['price']} {p.get('unit', '')}".rstrip())

    lines.append("\nSTOCK (available now):")
    for s in get_stock():
        count = s.get("stock")
        if count is None:
            status = "not counted (our staff will confirm availability)"
        else:
            status = "OUT OF STOCK" if not count else f"{count} {s.get('unit', 'trays')}"
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

    lines.append("\nPROMOS (applied to the order total automatically):")
    for p in KB.get("promotions", []):
        lines.append(f"- {p.get('details', '')}")

    return "\n".join(lines)


def build_examples():
    """Few-shot examples built from live data so they never show stale prices."""
    examples = []
    pricing = get_pricing()
    trays = [p for p in pricing if unit_name(p) == "tray"]
    sacks = [p for p in pricing if unit_name(p) == "sack"]
    if trays:
        p = trays[-1]
        examples.append(
            f"Customer: Magkano po ang isang tray ng {p['product'].split()[0]}?\n"
            f"Assistant: Ang {p['product']} po ay {p['price']} {p.get('unit', 'per tray')}. "
            "Ilang tray po ang kailangan ninyo?"
        )
    if sacks:
        p = sacks[0]
        examples.append(
            f"Customer: Magkano po ang isang sako ng {p['product']}?\n"
            f"Assistant: Ang {p['product']} po ay {p['price']} {p.get('unit', 'per sack')}. "
            "Ilang sako po ang kailangan ninyo?"
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
            "Ano po ang order ninyo at ilan po?"
        )
    min_trays, discount = egg_discount_rule()
    if min_trays:
        examples.append(
            "Customer: May discount po ba kung marami ang order?\n"
            f"Assistant: Opo! May {money(discount)} discount po ang buong order kapag "
            f"{min_trays} trays o higit pa ng itlog (kahit iba-ibang size). "
            "Ilang tray po ang order ninyo?"
        )
    examples.append(
        "Customer: Pwede po bang tawaran, ₱200 na lang per tray?\n"
        "Assistant: Pasensya na po, hindi ko po mababago ang presyo. Para po sa ganitong request, "
        f"pakitawagan o i-text po ang aming staff sa {staff_contact()}. Salamat po!"
    )
    examples.append(
        "Customer: What's the weather today?\n"
        f"Assistant: Sorry, I can only help with questions about {store_name()}'s eggs and rice, "
        "prices, delivery, and orders. Is there anything I can help you with?"
    )
    return "\n\n".join(examples)


SYSTEM_PROMPT = """You are the customer service assistant of {store}, a small store selling eggs (by the tray) \
and rice (by the sack). You help customers with prices, stock, delivery, payment, and taking orders.

LANGUAGE
- Look at the customer's LATEST message and reply in that language: English -> English only. Tagalog -> Tagalog. Taglish (mixed) -> Taglish.
- In Tagalog or Taglish, be polite and use "po" and "opo".
- Never translate product names. Write them exactly as listed (e.g. "Medium Eggs (M)", "Jasmine Denorado Rice").
- Always write prices with the ₱ sign (e.g. ₱238.00).

TAGALOG WORDS CUSTOMERS USE
- itlog = egg
- tray / bandehado = one tray of 30 eggs
- kaha = case/box (ask how many trays they mean)
- bigas = rice
- sako = sack
- magkano = how much
- meron pa ba = is there still any / is it in stock
- pa-deliver = please deliver

FACTS (the ONLY information you may use)
{facts}

RULES
1. Use ONLY the facts above. Never invent prices, discounts, stock, fees, delivery schedules, or any other detail. If the answer is not in the facts, say you are not sure and give the staff contact number {contact}.
2. If a product is OUT OF STOCK, say so and suggest one that is in stock. Never accept more than is in stock. If a product's stock is "not counted", you may take the order but say our staff will confirm availability.
3. You may tell customers about the PROMOS above; the system applies them to the order total automatically. Never promise or invent any other discount. If the customer asks for a different discount, tries to negotiate the price, or has a complaint (broken or rotten eggs, late or wrong delivery, refunds), do not decide it yourself: say our staff will help and give the number {contact}.
4. Only talk about the store and its products. Politely redirect unrelated questions.
5. Keep replies short: 1 to 4 sentences.
6. For delivery, ask whether the address is within Angeles City, then tell the customer that area's delivery fee and minimum order (if any) before continuing. Do not accept a delivery order below the minimum. Follow the DELIVERY facts for rice.

TAKING AN ORDER
One order can have several products (e.g. eggs and rice together). Collect these details, asking for only one or two at a time:
each product and its quantity (trays for eggs, sacks for rice), delivery or pickup, customer name, contact number, delivery address (delivery only; ask whether it is within Angeles City), payment method.
When you have ALL the details, do NOT calculate the total yourself. Write one short sentence (e.g. "Narito po ang inyong order:") and end your reply with this line, on its own line, as valid JSON, filled in ONLY with details the customer actually gave you:
[ORDER] {{"items": [{{"product": "<exact product name>", "quantity": <number>}}], "fulfillment": "<delivery or pickup>", "delivery_area": "<within_angeles or outside_angeles, null for pickup>", "name": "<customer's name>", "contact": "<customer's number>", "address": "<delivery address, empty for pickup>", "payment": "<payment method>"}}
The system then shows the customer the order summary with the correct total and asks them to confirm.
- If the customer confirms (e.g. "yes", "opo", "confirm"), thank them and write the same [ORDER] line again.
- If they change something, write the updated [ORDER] line.
List every product in "items". Never mention the [ORDER] line to the customer.
After the customer confirms, the order still has to be confirmed by our staff. Never say the order is confirmed, final, or ready; say our staff will confirm it and contact them on Messenger or by phone. If the customer asks about their order's status, say it is waiting for staff confirmation and give the staff number {contact}.

AFTER AN ORDER IS RECEIVED (a message starting with "⏳ ... Order #")
- That order is closed. Never change it, add to it, or repeat its items. Do not offer to "add to your order".
- If the customer wants more products, that is a NEW, separate order: its [ORDER] line lists ONLY the new products. Reuse the name, contact number, delivery details, and payment method they already gave (ask only if they want to change them).
- To change or cancel an order that was already received, give the staff number {contact}.

EXAMPLES
{examples}
"""


def build_system_prompt():
    return SYSTEM_PROMPT.format(
        store=store_name(),
        facts=build_facts(),
        contact=staff_contact(),
        examples=build_examples(),
    )


# ---------------------------------------------------------------------------
# Grounding check (AI-07): reject peso amounts that aren't from the facts
# ---------------------------------------------------------------------------

def _peso_amounts(text):
    return [float(m.replace(",", "")) for m in re.findall(r"₱\s?(\d[\d,]*(?:\.\d+)?)", text or "")]


def looks_ungrounded(answer, facts, user_message=""):
    """True if the answer mentions a ₱ amount that is neither in the facts /
    the customer's message nor a single order line (price x quantity,
    fee x trays, or (price + fee) x trays). Order totals are written by the
    backend (order_summary), not the model."""
    amounts = _peso_amounts(answer)
    if not amounts:
        return False
    known = _peso_amounts(facts) + _peso_amounts(user_message)
    prices = [peso_to_float(p["price"]) for p in get_pricing()]
    fees = [f for f in (delivery_rule("within_angeles")[0], delivery_rule("outside_angeles")[0]) if f]
    # (unit amount, max quantity). Fee-only lines get a tight cap; otherwise
    # a ₱5 fee would make any multiple of 5 "valid".
    max_qty = max([20] + [s.get("stock") or 0 for s in get_stock()])
    units = [(u, max_qty) for u in prices + [p + f for p in prices for f in fees] if u]
    units += [(f, 20) for f in fees]

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


def order_lines(order):
    """The [ORDER] line's items; also accepts the old one-product format."""
    items = order.get("items")
    if items is None and order.get("size"):
        items = [{"product": order.get("size"), "quantity": order.get("trays")}]
    return items if isinstance(items, list) else []


def validate_and_price(order):
    """Recalculates the total from backend prices. Returns (record, error)."""
    items = {}
    for line in order_lines(order):
        if not isinstance(line, dict):
            return None, "invalid order items"
        product = find_product(str(line.get("product", "")))
        if not product:
            return None, f"unknown product: {line.get('product')}"
        try:
            quantity = int(line.get("quantity"))
        except (TypeError, ValueError):
            return None, "invalid quantity"
        if quantity < 1:
            return None, "invalid quantity"
        name = product["product"]
        if name in items:
            items[name]["quantity"] += quantity
        else:
            items[name] = {"product": name, "quantity": quantity, "unit": unit_name(product),
                           "unit_price": peso_to_float(product["price"])}
    if not items:
        return None, "no products in the order"

    items = list(items.values())
    for item in items:
        stock = stock_for(item["product"])
        if stock is not None and item["quantity"] > stock:
            return None, f"only {count_units(stock, item['unit'])} of {item['product']} in stock"
        item["line_total"] = item["unit_price"] * item["quantity"]

    egg_trays = sum(i["quantity"] for i in items if i["unit"] == "tray")
    has_sacks = any(i["unit"] == "sack" for i in items)

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
        if delivery_area == "outside_angeles" and has_sacks:
            return None, "rice delivery outside Angeles City is arranged by our staff"
        fee_per_tray, minimum = delivery_rule(delivery_area)
        if fee_per_tray is None:
            return None, "delivery is not available for that area"
        if minimum > 1 and egg_trays < minimum:
            return None, f"delivery to that area needs at least {minimum} trays of eggs"
        fee_total = fee_per_tray * egg_trays

    min_trays, discount = egg_discount_rule()
    if not (min_trays and egg_trays >= min_trays):
        discount = 0.0

    record = {
        "items": items,
        "fulfillment": fulfillment,
        "delivery_area": delivery_area,
        "delivery_fee": fee_total,
        "discount": discount,
        "total": sum(i["line_total"] for i in items) + fee_total - discount,
        "customer_name": name,
        "contact": contact,
        "address": address,
        "payment_method": match or payment,
    }
    return record, None


def describe_items(items):
    """'5 trays Small Eggs (S) x ₱230.00 = ₱1,150.00; 1 sack ...'"""
    return "; ".join(
        f"{count_units(i['quantity'], i['unit'])} {i['product']} x {money(i['unit_price'])} "
        f"= {money(i['line_total'])}"
        for i in items
    )


def recent_order_id(session_id, record):
    """Id of the same order saved in the last 30 minutes, if any (the model
    sometimes repeats the [ORDER] line after the order is saved)."""
    with db() as conn:
        row = conn.execute(
            "SELECT id FROM orders WHERE session_id = ? AND items_summary = ? "
            "AND contact = ? AND created_at >= ?",
            (session_id, describe_items(record["items"]), record["contact"],
             datetime.fromtimestamp(time.time() - 1800).isoformat(timespec="seconds")),
        ).fetchone()
    return row["id"] if row else None


def set_pending_order(session_id, order):
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO pending_orders (session_id, raw_json, updated_at) VALUES (?, ?, ?)",
            (session_id, json.dumps(order, ensure_ascii=False), now_iso()),
        )


def get_pending_order(session_id):
    with db() as conn:
        row = conn.execute("SELECT raw_json FROM pending_orders WHERE session_id = ?",
                           (session_id,)).fetchone()
    return json.loads(row["raw_json"]) if row else None


SUMMARY_MARKER = "Order summary"


def last_bot_message(session_id):
    return next((m["content"] for m in reversed(recent_messages(session_id))
                 if m["role"] == "assistant"), "")


def summary_was_shown(session_id, record):
    """True if the bot's previous message was the summary of this exact order."""
    last = last_bot_message(session_id)
    return describe_items(record["items"]) in last and money(record["total"]) in last


def save_order(session_id, record, raw_order):
    summary = describe_items(record["items"])
    with db() as conn:
        conn.execute("DELETE FROM pending_orders WHERE session_id = ?", (session_id,))
        cur = conn.execute(
            """INSERT INTO orders (session_id, created_at, items_summary, fulfillment,
               delivery_area, delivery_fee, discount, total, customer_name, contact, address,
               payment_method, raw_json, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
            (session_id, now_iso(), summary, record["fulfillment"], record["delivery_area"],
             record["delivery_fee"], record["discount"], record["total"], record["customer_name"],
             record["contact"], record["address"], record["payment_method"],
             json.dumps(raw_order, ensure_ascii=False)),
        )
        order_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO order_items (order_id, product, quantity, unit, unit_price, line_total) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(order_id, i["product"], i["quantity"], i["unit"], i["unit_price"], i["line_total"])
             for i in record["items"]],
        )
        return order_id


def fees_and_discount(record):
    """' + delivery fee ₱30.00 - discount ₱50.00' (empty parts left out)."""
    text = f" + delivery fee {money(record['delivery_fee'])}" if record["delivery_fee"] else ""
    if record.get("discount"):
        text += f" - discount {money(record['discount'])}"
    return text


def order_received(order_id, record, language):
    """Shown after the customer confirms: the order now waits for staff."""
    details = (
        f"{describe_items(record['items'])}{fees_and_discount(record)}. "
        f"Total: {money(record['total'])}, {record['fulfillment']}, {record['payment_method']}."
    )
    if language == "english":
        return (
            f"⏳ Order #{order_id} received and waiting for staff confirmation: {details} "
            f"Your order is not final yet. Please wait for our staff to confirm it; "
            f"they will contact you on Messenger or at {record['contact']}. Thank you!"
        )
    return (
        f"⏳ Natanggap na po ang Order #{order_id} at hinihintay na lang po ang kumpirmasyon "
        f"ng aming staff: {details} Hindi pa po final ang order. Pakihintay po ang "
        f"kumpirmasyon ng aming staff; kokontakin po nila kayo sa Messenger o sa "
        f"{record['contact']}. Salamat po!"
    )


def order_summary(record, language):
    """Order summary with the backend's total, shown before the customer confirms."""
    where = f"delivery to {record['address']}" if record["fulfillment"] == "delivery" else "pickup"
    details = (
        f"{describe_items(record['items'])}{fees_and_discount(record)}. "
        f"Total: {money(record['total'])} ({where}, "
        f"{record['payment_method']}). {record['customer_name']}, {record['contact']}."
    )
    if language == "english":
        return f"{SUMMARY_MARKER}: {details} Reply \"yes\" to confirm."
    return f"{SUMMARY_MARKER} po: {details} Paki-reply po ng \"opo\" para i-confirm."


# ---------------------------------------------------------------------------
# Language detection (tells the model which language to reply in)
# ---------------------------------------------------------------------------

TAGALOG_WORDS = {
    "po", "opo", "ang", "ng", "mga", "sa", "na", "ba", "pa", "ko", "ako", "kayo",
    "ninyo", "nyo", "niyo", "namin", "natin", "magkano", "meron", "mayroon", "wala",
    "salamat", "itlog", "lang", "naman", "yung", "sana", "pwede", "puwede", "paano",
    "saan", "ilan", "ilang", "oo", "hindi", "din", "rin", "kasi", "gusto", "bili",
    "bibili", "pa-deliver", "padeliver", "pa-order", "isang", "dalawa", "tatlo",
    "bandehado", "kaha", "sige", "tig", "ano", "kailan", "pakidala", "bigas", "sako",
}
ENGLISH_WORDS = {
    "the", "i", "i'm", "want", "do", "does", "you", "have", "how", "much", "is",
    "are", "my", "for", "please", "can", "what", "when", "where", "of", "and", "to",
    "would", "like", "it", "yes", "thanks", "thank", "your", "there", "any", "need",
    "will", "price", "deliver", "hi", "hello",
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
        f"Pasensya na po, may problema po sa order ({reason}). "
        f"Sorry, there's a problem with the order ({reason}). "
        f"You can also contact our staff at {staff_contact()}."
    )


# ---------------------------------------------------------------------------
# The one reusable chat function (web chat and Messenger both call this)
# ---------------------------------------------------------------------------

def ask_gemini(model, messages):
    response = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        messages=messages,
        # extra_body so it works with the pinned openai==1.51.0 SDK
        extra_body={"reasoning_effort": GEMINI_REASONING_EFFORT},
    )
    return (response.choices[0].message.content or "").strip()


def call_model(messages):
    """Asks the main model; if Google says it's busy, tries each fallback model."""
    models = [GEMINI_MODEL] + GEMINI_FALLBACK_MODELS
    for i, model in enumerate(models):
        try:
            return ask_gemini(model, messages)
        except BUSY_ERRORS as e:
            if i == len(models) - 1:
                raise
            print(f"{model} busy ({type(e).__name__}), trying {models[i + 1]}")


def handle_message(session_id, text):
    """load prices -> build prompt -> call Gemini -> handle order -> reply text."""
    text = text[:MAX_MESSAGE_CHARS]
    user_message_id = save_message(session_id, "user", text)

    system_prompt = build_system_prompt()
    language = reply_language(session_id)
    if language:
        system_prompt += f"\nREPLY LANGUAGE FOR THIS MESSAGE: {LANGUAGE_HINTS[language]}\n"
    messages = [{"role": "system", "content": system_prompt}] + recent_messages(session_id)

    try:
        raw_reply = call_model(messages)
    except Exception as e:  # Gemini down, bad API key, quota, etc.
        print("Gemini error:", repr(e))
        # Unanswered messages left in the history confuse the next reply.
        delete_message(user_message_id)
        return ai_down_message()

    reply, order, had_tag = extract_order(raw_reply)
    if (order is None and not had_tag and is_confirmation(text) and len(text.split()) <= 4
            and SUMMARY_MARKER in last_bot_message(session_id)):
        # A short "yes" right after the summary, but the model didn't repeat
        # the [ORDER] line: use the order waiting for confirmation.
        order = get_pending_order(session_id)

    record, error = None, None
    if order is not None:
        record, error = validate_and_price(order)
        if not error and not details_came_from_customer(order, session_id):
            error = "please send your name and contact number first"
    elif had_tag:
        error = "the order details were incomplete"

    if looks_ungrounded(reply, system_prompt, text):
        print("Ungrounded reply replaced:", reply)
        reply = ""

    if error:
        reply = (reply + "\n\n" + order_problem(error)).strip()
    elif record:
        if recent_order_id(session_id, record):
            pass  # already saved; the model just repeated the [ORDER] line
        elif is_confirmation(text) and summary_was_shown(session_id, record):
            stock_error = take_stock(record["items"])
            if stock_error:  # someone else bought the last ones in the meantime
                reply = (reply + "\n\n" + order_problem(stock_error)).strip()
            else:
                order_id = save_order(session_id, record, order)
                print(f"New order #{order_id} (waiting for staff): {record}")
                reply = (reply + "\n\n" + order_received(order_id, record, language)).strip()
        else:
            # Show the backend's summary; the order is saved only after the
            # customer confirms this exact summary.
            set_pending_order(session_id, order)
            reply = (reply + "\n\n" + order_summary(record, language)).strip()

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
    session_id = re.sub(r"[^A-Za-z0-9_-]", "", str(data.get("session_id", "")))[:64]
    session_id = session_id or uuid.uuid4().hex

    if not question:
        return jsonify({"reply": "Please type a question!", "session_id": session_id})

    try:
        answer = handle_message("web:" + session_id, question)
    except Exception as e:  # NFR-07: fail safely, never crash the chat
        print("Chat error:", repr(e))
        answer = ai_down_message()

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
    data = request.get_json(silent=True) or {}

    for entry in data.get("entry", []):
        for event in entry.get("messaging", []):
            sender_id = event.get("sender", {}).get("id")
            message = event.get("message", {})

            # is_echo = the Page's own reply coming back; answering it would loop.
            if sender_id and "text" in message and not message.get("is_echo"):
                try:
                    answer = handle_message(f"messenger:{sender_id}", message["text"])
                except Exception as e:
                    print("Messenger error:", repr(e))
                    answer = ai_down_message()
                send_message(sender_id, answer)

    return "ok", 200


def send_message(recipient_id, text):
    url = f"https://graph.facebook.com/v21.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    payload = {"recipient": {"id": recipient_id}, "message": {"text": text}}
    resp = requests.post(url, json=payload, timeout=10)
    if resp.status_code != 200:
        print("Messenger send error:", resp.status_code, resp.text)


if __name__ == "__main__":
    # FLASK_DEBUG=1 in .env: auto-reload on code changes. Off by default because
    # the debug page would be reachable by anyone on the same network.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
