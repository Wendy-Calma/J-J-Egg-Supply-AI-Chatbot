# J&J Egg Supply AI Chatbot -- Starter Code

This is a working skeleton for the chatbot described in the project SRS.
One backend, two entry points (web chat + Facebook Messenger), both calling
the same retrieval + grounding + Luna logic.

## Run it locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# install Ollama (ollama.com) and pull the model once:
ollama pull qwen2.5:7b
python app.py
```

Open http://localhost:5000 in your browser to use the web chat.

## Files

- `app.py` -- the whole backend: KB loading, retrieval, grounding, /chat, /webhook
- `kb/*.json` -- one file per knowledge category (edit these directly, no database needed)
- `static/index.html` -- the web chat widget (your team's testing/demo tool)
- `.env.example` -- copy to `.env` and fill in real secrets (never commit `.env` itself)

## Before your Messenger webhook will work

1. Get `PAGE_ACCESS_TOKEN` from the Meta Developer Console (requires J&J granting
   your team a Page role first -- see SRS section 2.1).
2. Deploy this app somewhere with a public HTTPS URL (Render, Railway, etc.) --
   Messenger's webhook cannot reach `localhost`.
3. In the Meta Developer Console, set the webhook URL to
   `https://<your-deployed-url>/webhook` and the verify token to whatever you
   set as `MESSENGER_VERIFY_TOKEN` in `.env`.

## Editing the knowledge base

Just edit the JSON files in `kb/`. No restart needed for the web chat during
local development (files are re-read... actually note: currently the KB is
loaded once at startup -- restart `python app.py` after editing KB files,
or extend `load_knowledge_base()` to reload on each request if you want
live updates without a restart).

## How the bot works

- The model runs locally in Ollama (`OLLAMA_URL`, `OLLAMA_MODEL` in `.env`).
  Only the Flask backend talks to Ollama; the browser never does.
- `handle_message(session_id, text)` in `app.py` is the one chat function
  (load prices -> build prompt -> call Ollama -> handle order). `/chat` and
  `/webhook` (Messenger) both call it.
- The system prompt is rebuilt on every message from live prices, stock,
  delivery, payment, and store info, so the bot only uses current facts.
- Chat history and orders are saved in SQLite (`jj_egg.db`). Only the last
  10 messages of a conversation are sent to the model.
- When a customer confirms an order, the model writes a hidden `[ORDER] {...}`
  line. The backend removes it, re-checks size, stock, delivery rules, and
  payment, recalculates the total from its own prices (never the model's
  math), saves it to the `orders` table, and shows the customer the
  confirmed total.
- To see orders: `sqlite3 jj_egg.db "SELECT * FROM orders ORDER BY id DESC;"`
  (or open the file with DB Browser for SQLite).
- `kb/promotions.json` is not shown to the bot: discounts, bulk orders (12+
  trays), price negotiation, and complaints are always referred to staff.

## Updating egg prices (Google Sheet)

Prices change often, so they can be managed in a Google Sheet instead of
editing code or JSON. The bot re-reads the sheet at most every 5 minutes.

One-time setup:

1. Create a Google Sheet with this header row (first row):
   `product | price | unit | size_range | last_updated`
2. Add one row per egg size, e.g.
   `Medium Eggs (M) | 238 | per tray of 30 pcs | 55-59 grams | 2026-11-05`
   (price can be `238`, `238.00` or `₱238`).
3. File -> Share -> **Publish to web** -> pick that sheet -> **Comma-separated
   values (.csv)** -> Publish, and copy the link.
4. Put the link in `.env` (and in your host's environment variables):
   `PRICING_SHEET_CSV_URL=https://docs.google.com/spreadsheets/d/e/.../pub?output=csv`
5. Restart the app once.

After that, just edit the price in the sheet. The bot picks it up within about
5 minutes (Google's publish delay plus the bot's cache), with no restart and
no git. If the sheet can't be reached, the bot falls back to `kb/pricing.json`,
so update that file once in a while too.

## Before you run this

Two fields in `kb/business.json` and `kb/ordering.json` are placeholders --
`INSERT REAL ADDRESS HERE` and `INSERT REAL NUMBER HERE`. These were redacted
for privacy when shared with the AI assistant that helped build this. Fill in
the real values yourself before using this for real customers.

## A real lesson already learned from testing this KB

Keyword retrieval only finds what's actually *worded* in the KB. The first
version of `business.json` only had `"operating_hours": "7:00 AM - 5:00 PM"`,
so "What time do you open?" returned nothing -- the words "time" and "open"
never appeared anywhere in the data. Adding a natural-language `hours_note`
field fixed it. If a real customer question isn't matching, this is almost
always why: check whether the words they'd actually use appear anywhere in
your JSON, not just the structured field names.

## Facebook Page access

The team now has Meta Business Suite access to J&J's existing Page. Next
step: create the Meta Developer App and generate a Page Access Token (see
the Development & Implementation Guide, Section 6) -- then fill in
`PAGE_ACCESS_TOKEN` in `.env`.
