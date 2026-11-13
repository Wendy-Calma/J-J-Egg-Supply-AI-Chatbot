/**
 * J&J Egg Supply -- Google Apps Script for the products/prices/stock sheet.
 *
 * Lets the chatbot read the sheet live (no "Publish to web" delay) and
 * subtract stock when a customer confirms an order.
 *
 * Setup (see README.md, "Automatic stock updates"):
 *   1. In the Google Sheet: Extensions -> Apps Script, paste this whole file.
 *   2. Change SECRET below to a long random string, and put the same value in
 *      .env as SHEET_SCRIPT_SECRET.
 *   3. Deploy -> New deployment -> type "Web app", Execute as: Me,
 *      Who has access: Anyone. Copy the Web app URL into .env as SHEET_SCRIPT_URL.
 */

const SECRET = 'change-this-to-a-long-random-string';

function sheet_() {
  // The first tab of the spreadsheet (product | price | unit | ... | stock).
  return SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}

// Read: returns every row as text, exactly as shown in the sheet.
function doGet() {
  return json_({ rows: sheet_().getDataRange().getDisplayValues() });
}

// Write: {secret, items: [{product, quantity}]} -> subtracts stock.
// Checks every item first, so an order is never half-subtracted.
function doPost(e) {
  const body = JSON.parse(e.postData.contents);
  if (body.secret !== SECRET) return json_({ error: 'not allowed' });

  const lock = LockService.getScriptLock();
  lock.waitLock(10000);  // two orders at once can't both take the last tray
  try {
    const sheet = sheet_();
    const values = sheet.getDataRange().getValues();
    const headers = values[0].map(h => String(h).trim().toLowerCase());
    const productCol = headers.indexOf('product');
    const stockCol = headers.indexOf('stock');
    if (productCol < 0 || stockCol < 0) return json_({ error: 'sheet needs product and stock columns' });

    const updates = [];
    for (const item of body.items) {
      const row = values.findIndex((r, i) =>
        i > 0 && String(r[productCol]).trim().toLowerCase() === String(item.product).trim().toLowerCase());
      if (row < 0) return json_({ error: 'unknown product: ' + item.product });
      const stock = values[row][stockCol];
      if (stock === '' || stock === null) continue;  // stock not counted
      if (Number(stock) < item.quantity) {
        return json_({ error: 'only ' + stock + ' left of ' + item.product });
      }
      updates.push([row, Number(stock) - item.quantity]);
    }
    updates.forEach(([row, left]) => sheet.getRange(row + 1, stockCol + 1).setValue(left));
    return json_({ ok: true });
  } finally {
    lock.releaseLock();
  }
}
