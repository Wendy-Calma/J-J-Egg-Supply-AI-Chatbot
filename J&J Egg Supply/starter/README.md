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
# edit .env and paste in your real NVIDIA_API_KEY (from build.nvidia.com)
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
