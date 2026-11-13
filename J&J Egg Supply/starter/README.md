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
# put your Gemini API key (from aistudio.google.com) in .env as GEMINI_API_KEY
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

Edit the JSON files in `kb/`, then restart `python app.py` (the KB is loaded
once at startup). Egg prices are better managed in the Google Sheet (below),
which updates without a restart.

## How the bot works

- The model is Google Gemini (`GEMINI_API_KEY`, `GEMINI_MODEL` in `.env`),
  called through its OpenAI-compatible API. Only the Flask backend talks to
  Gemini; the API key never reaches the browser. Chat messages (including
  customer names, addresses, and phone numbers in orders) are sent to Google.
- `handle_message(session_id, text)` in `app.py` is the one chat function
  (load prices -> build prompt -> call Gemini -> handle order). `/chat` and
  `/webhook` (Messenger) both call it.
- The system prompt is rebuilt on every message from live prices, stock,
  delivery, payment, and store info, so the bot only uses current facts.
- Chat history and orders are saved in SQLite (`jj_egg.db`). Only the last
  10 messages of a conversation are sent to the model.
- One order can have several products (e.g. eggs and rice). When the model
  has all the order details it writes a hidden `[ORDER] {...}` line. The
  backend removes it, re-checks products, stock, delivery rules, and payment,
  calculates the total from its own prices (never the model's math), and
  shows the customer the order summary. The order is saved only when the
  customer replies "yes"/"opo" to that exact summary.
- A saved order has status `pending`: the customer is told it is waiting for
  staff confirmation and that staff will contact them. After staff confirm
  with the customer, mark it, e.g.
  `sqlite3 jj_egg.db "UPDATE orders SET status = 'confirmed' WHERE id = 12;"`
- Delivery rules: free within Angeles City (eggs and rice). Outside Angeles
  City: ₱5 per tray of eggs, 6 trays minimum; rice outside Angeles City is
  referred to staff.
- Orders are saved in the `orders` table (one row per order, with an
  `items_summary`) and `order_items` (one row per product). To see them:
  `sqlite3 jj_egg.db "SELECT id, created_at, customer_name, contact, items_summary, total, fulfillment FROM orders ORDER BY id DESC;"`
  (or open the file with DB Browser for SQLite).
- Discount: ₱50 off the whole order for 12 or more trays of eggs (all sizes
  combined). The rule lives in `kb/promotions.json` (`min_egg_trays`,
  `discount`); the backend applies it to the total. Other discounts, price
  negotiation, and complaints are referred to staff.
- Orders are limited only by the stock in the sheet (no bulk limit).

## Products, prices, and stock (Google Sheet)

The Google Sheet is the only place products, prices, and stock are kept.
To add a product, add a row; no code change or restart is needed. Use
`per tray ...` or `per sack` in the `unit` column: products with "sack" in the
unit are counted in sacks, everything else in trays. The bot re-reads the
sheet at most once a minute.

One-time setup:

1. Create a Google Sheet with this header row (first row):
   `product | price | unit | size_range | last_updated | stock`
2. Add one row per product, e.g.
   `Medium Eggs (M) | 238 | per tray of 30 pcs | 55-59 grams | 2026-11-05 | 20`
   `Jasmine Denorado Rice | 1200 | per sack | 25 kg | 2026-11-13 | 8`
   (price can be `238`, `238.00`, `₱238` or `₱1,200`).
   `stock` is trays for eggs and sacks for rice: `0` = out of stock, an empty
   cell = not counted (the bot says staff will confirm availability).
3. File -> Share -> **Publish to web** -> pick that sheet -> **Comma-separated
   values (.csv)** -> Publish, and copy the link.
4. Put the link in `.env` (and in your host's environment variables):
   `PRICING_SHEET_CSV_URL=https://docs.google.com/spreadsheets/d/e/.../pub?output=csv`
5. Restart the app once.

After that, just edit prices or stock in the sheet. The bot picks it up within
about 5-6 minutes (mostly Google's publish delay, plus up to 1 minute of the
bot's cache), with no restart and no git.

If the sheet can't be reached, the bot keeps using the last copy it loaded.
If it can't load the sheet at all (e.g. no internet when the app starts), it
tells customers prices are unavailable and gives the staff number.

## Automatic stock updates (optional)

With this set up, confirmed orders subtract from the sheet's `stock` column
(e.g. 2 trays ordered: 20 -> 18), and the bot reads the sheet live, so price
and stock edits show up within a minute instead of 5.

1. Open the Google Sheet -> **Extensions -> Apps Script**. Delete what's there
   and paste the contents of `sheet_script.gs`.
2. In the script, change `SECRET` to a long random string. Save.
3. **Deploy -> New deployment** -> gear icon -> **Web app**.
   Execute as: **Me**. Who has access: **Anyone**. Deploy, allow the
   permissions, and copy the **Web app URL**.
4. In `.env`:
   `SHEET_SCRIPT_URL=<the Web app URL>`
   `SHEET_SCRIPT_SECRET=<the same secret as in the script>`
5. Restart the app.

Only the first tab of the sheet is read. The read link is public (like the
published CSV), but stock can only be changed with the secret. If you edit
the script later, use **Deploy -> Manage deployments -> Edit -> New version**
so the URL stays the same. If the script can't be reached when an order is
confirmed, the order is still saved and the server log says to update the
stock by hand.

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
