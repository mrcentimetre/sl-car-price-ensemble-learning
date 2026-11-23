"""
Scrape public used-car listings from ikman.lk (Sri Lanka) for the
Ensemble Learning assignment.

Usage:
    python3 scraper/scrape_ikman.py --pages 220 --out data/ikman_cars_raw.jsonl

Notes
- Only fetches public category pages (/en/ads/sri-lanka/cars?page=N) and public
  ad pages (/en/ad/<slug>) which robots.txt does not disallow.
- Rate limited (small worker pool + per-request delay). Resumable: slugs already
  in the output file are skipped.
- Code in this file was written with the help of Claude (Anthropic) — AI-assisted.
"""
import argparse, json, random, re, sys, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://ikman.lk"
UA = "Mozilla/5.0 (compatible; UoM-ML-Assignment/1.0; educational use)"
DATA_RE = re.compile(r"window\.initialData\s*=\s*(\{.*?\})\s*;?\s*</script>", re.S)
DELAY = 0.6  # seconds between requests per worker


def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en"})
            with urllib.request.urlopen(req, timeout=120) as r:
                html = r.read().decode("utf-8", "replace")
            m = DATA_RE.search(html)
            if not m:
                raise ValueError("no initialData")
            return json.loads(m.group(1))
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            time.sleep(2 * (attempt + 1))
            last = e
    print(f"  ! failed {url}: {last}", file=sys.stderr)
    return None


def list_page(page):
    d = fetch(f"{BASE}/en/ads/sri-lanka/cars?page={page}")
    time.sleep(DELAY)
    if not d:
        return []
    return [a for a in d["serp"]["ads"]["data"]["ads"] if a.get("category", {}).get("id") == 392]


def detail(slug):
    d = fetch(f"{BASE}/en/ad/{slug}")
    time.sleep(DELAY)
    if not d:
        return None
    ad = d["adDetail"]["data"]["ad"]
    props = {p["key"]: p["value"] for p in ad.get("properties", [])}
    loc = ad.get("location") or {}
    area = ad.get("area") or {}
    return {
        "id": ad["id"],
        "slug": slug,
        "title": ad.get("title"),
        "price_raw": (ad.get("money") or {}).get("amount"),
        "money": ad.get("money"),
        "brand": props.get("brand"),
        "model": props.get("model"),
        "year": props.get("model_year"),
        "condition": props.get("condition"),
        "transmission": props.get("transmission"),
        "body": props.get("body"),
        "fuel": props.get("fuel_type"),
        "engine": props.get("engine_capacity"),
        "mileage": props.get("mileage"),
        "location": loc.get("name"),
        "area": area.get("name") if isinstance(area, dict) else area,
        "posted": ad.get("adDate"),
        "is_member": ad.get("isMember"),
        "is_auth_dealer": ad.get("isAuthDealer"),
        "membership_level": ad.get("membershipLevel"),
        "is_verified": ad.get("isVerified"),
        "description_len": len(ad.get("description") or ""),
        "n_images": len((ad.get("images") or {}).get("ids", [])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=200)
    ap.add_argument("--page-step", type=int, default=1, help="fetch every Nth list page (spreads the sample over the whole catalogue)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="data/ikman_cars_raw.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="max ads to fetch details for (0 = all)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            try:
                done.add(json.loads(line)["slug"])
            except Exception:
                pass
    print(f"resuming with {len(done)} ads already saved")

    # 1) collect slugs from list pages
    slugs = {}
    with ThreadPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(list_page, p) for p in range(1, args.pages + 1, args.page_step)]):
            for a in fut.result():
                slugs[a["slug"]] = a
    todo = [s for s in slugs if s not in done]
    random.Random(42).shuffle(todo)  # random sample across pages, not just the newest ads
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(slugs)} unique slugs from list pages, {len(todo)} to fetch")

    # 2) detail pages
    lock = threading.Lock()
    n = 0
    with ThreadPoolExecutor(args.workers) as ex, out.open("a") as f:
        for fut in as_completed([ex.submit(detail, s) for s in todo]):
            row = fut.result()
            if row:
                row["list_price"] = slugs[row["slug"]].get("price")
                row["list_details"] = slugs[row["slug"]].get("details")
                with lock:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    n += 1
                    if n % 100 == 0:
                        print(f"  {n}/{len(todo)} saved", flush=True)
    print(f"done: {n} new ads written to {out}")


if __name__ == "__main__":
    main()
