"""
keep_alive.py
=============
Supabase Keep-Alive Script for the Vision-Language Pipeline.

Purpose:
    Supabase pauses free-tier projects after 7 days of inactivity.
    This script inserts a single realistic mock record into `vlp_extractions`
    every time it runs to keep the database active and demonstrate real-world
    data variety.

Run manually:
    python scripts/keep_alive.py

Run via GitHub Actions:
    Triggered automatically every 3 days (see .github/workflows/keep-alive.yml).

Required environment variables:
    SUPABASE_URL   - Your project URL (e.g. https://xxxx.supabase.co)
    SUPABASE_KEY   - Your service_role or anon key
"""

import os
import sys
import uuid
import random
from datetime import datetime, timezone

# ── Supabase client ───────────────────────────────────────────────────────────
try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] supabase package not found. Run: pip install supabase")
    sys.exit(1)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("[ERROR] SUPABASE_URL and SUPABASE_KEY environment variables must be set.")
    sys.exit(1)

client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print(f"[OK] Connected to Supabase: {SUPABASE_URL}")


# ── Invoice mock data ─────────────────────────────────────────────────────────
VENDORS = ["AWS", "Vercel", "OpenAI", "Google Cloud", "Stripe", "Twilio", "Cloudflare"]

INVOICE_LINE_ITEMS = [
    {"description": "Compute (EC2 / Cloud Run)", "unit": "hours"},
    {"description": "Storage (S3 / GCS)", "unit": "GB"},
    {"description": "API Inference Tokens", "unit": "1M tokens"},
    {"description": "Bandwidth Egress", "unit": "GB"},
    {"description": "Managed Database (RDS / Supabase Pro)", "unit": "months"},
    {"description": "CDN Requests", "unit": "1M requests"},
    {"description": "Serverless Function Invocations", "unit": "1M calls"},
]


def make_invoice_payload() -> dict:
    vendor = random.choice(VENDORS)
    num_items = random.randint(1, 4)
    line_items = []
    subtotal = 0.0

    for item in random.sample(INVOICE_LINE_ITEMS, num_items):
        qty = round(random.uniform(1, 500), 2)
        unit_price = round(random.uniform(0.002, 0.85), 4)
        total = round(qty * unit_price, 2)
        subtotal += total
        line_items.append({
            "description": item["description"],
            "quantity": qty,
            "unit": item["unit"],
            "unit_price_usd": unit_price,
            "line_total_usd": total,
        })

    tax_rate = random.choice([0.0, 0.05, 0.08, 0.10])
    tax = round(subtotal * tax_rate, 2)
    grand_total = round(subtotal + tax, 2)

    return {
        "vendor": vendor,
        "invoice_number": f"INV-{random.randint(10000, 99999)}",
        "invoice_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "due_date": f"2026-0{random.randint(8, 9)}-{random.randint(1, 28):02d}",
        "currency": "USD",
        "line_items": line_items,
        "subtotal_usd": round(subtotal, 2),
        "tax_rate_pct": tax_rate * 100,
        "tax_usd": tax,
        "grand_total_usd": grand_total,
        "payment_terms": random.choice(["Net 30", "Net 15", "Due on Receipt"]),
        "status": random.choice(["unpaid", "paid", "overdue"]),
    }


# ── Video mock data ───────────────────────────────────────────────────────────
VIDEO_EVENTS = [
    "person_detected",
    "forklift_moving",
    "vehicle_parked",
    "crowd_forming",
    "package_dropped",
    "door_opened",
    "fire_detected",
    "unauthorized_zone_entry",
]

OBJECT_LABELS = [
    "person", "forklift", "car", "truck", "box",
    "fire", "door", "pallet", "bicycle", "dog",
]


def make_bbox() -> list[float]:
    """Generate a random normalized bounding box [x1, y1, x2, y2]."""
    x1 = round(random.uniform(0.0, 0.6), 4)
    y1 = round(random.uniform(0.0, 0.6), 4)
    x2 = round(x1 + random.uniform(0.1, 0.4), 4)
    y2 = round(y1 + random.uniform(0.1, 0.4), 4)
    return [min(x1, 1.0), min(y1, 1.0), min(x2, 1.0), min(y2, 1.0)]


def make_video_payload() -> dict:
    event = random.choice(VIDEO_EVENTS)
    num_objects = random.randint(1, 5)
    tracked_objects = []

    for _ in range(num_objects):
        tracked_objects.append({
            "label": random.choice(OBJECT_LABELS),
            "confidence": round(random.uniform(0.72, 0.99), 4),
            "timestamp_seconds": round(random.uniform(0.0, 120.0), 2),
            "box_2d": make_bbox(),
        })

    return {
        "event_type": event,
        "event_detected": True,
        "summary": f"Automated keep-alive: {event.replace('_', ' ').title()} observed at {datetime.now(timezone.utc).strftime('%H:%M UTC')}.",
        "camera_id": f"CAM-{random.randint(1, 16):02d}",
        "duration_seconds": round(random.uniform(5.0, 300.0), 1),
        "tracked_objects": tracked_objects,
        "frame_count": random.randint(30, 9000),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    job_type = random.choice(["invoice", "video"])
    print(f"[INFO] Generating mock '{job_type}' record...")

    if job_type == "invoice":
        payload = make_invoice_payload()
        prompt_used = f"Extract all billing details for {payload['vendor']} invoice."
    else:
        payload = make_video_payload()
        prompt_used = f"Detect and track all objects. Focus on: {payload['event_type']}."

    record = {
        "id": str(uuid.uuid4()),
        "job_type": job_type,
        "prompt_used": prompt_used,
        "parsed_data": payload,
        "status": "completed",
        "document_id": str(uuid.uuid4()),  # must be a valid UUID to match DB column type
    }

    print(f"[INFO] Inserting record: id={record['id']}  job_type={job_type}")
    response = client.table("vlp_extractions").insert(record).execute()

    if response.data:
        print(f"[SUCCESS] Record inserted into vlp_extractions.")
        print(f"          id         : {record['id']}")
        print(f"          job_type   : {job_type}")
        print(f"          prompt_used: {prompt_used}")
    else:
        print(f"[ERROR] Insert returned no data. Response: {response}")
        sys.exit(1)


if __name__ == "__main__":
    main()
