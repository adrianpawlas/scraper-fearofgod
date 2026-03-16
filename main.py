"""
Full pipeline: scrape Fear of God products, compute SigLIP embeddings, upsert to Supabase.
"""
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
from scraper import scrape_all_products
from embeddings import (
    load_siglip,
    image_embedding_from_url,
    text_embedding,
    build_info_text,
)
from db import (
    get_existing_products,
    upsert_products_batch,
    delete_stale_products,
    update_missing_runs,
)
from config import SOURCE, BRAND, BATCH_SIZE


def run(dry_run: bool = False, limit: int | None = None):
    """Scrape, embed, and upsert with smart logic."""
    print("Loading SigLIP model...")
    processor, tokenizer, model, device = load_siglip()
    
    print("Fetching existing products from database...")
    existing_products = get_existing_products(SOURCE)
    print(f"Found {len(existing_products)} existing products in database.")
    
    print("Scraping products...")
    products = list(scrape_all_products())
    if limit:
        products = products[:limit]
    print(f"Found {len(products)} products from scraper.")
    
    if dry_run:
        for i, p in enumerate(products[:5]):
            print(f"  {i+1}. {p.get('title')} | {p.get('product_url')}")
        return
    
    products = [p for p in products if (p.get("image_url") or "").strip()]
    print(f"Processing {len(products)} products with images...")
    
    new_count = 0
    updated_count = 0
    unchanged_count = 0
    
    rows_to_upsert = []
    seen_urls = set()
    
    for i, row in enumerate(products):
        product_url = row.get("product_url", "")
        seen_urls.add(product_url)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processing {i+1}/{len(products)}: {row.get('title', '')[:50]}...")
        
        existing = existing_products.get(product_url)
        
        if existing is None:
            new_count += 1
            _generate_embeddings(row, processor, tokenizer, model, device, i)
            rows_to_upsert.append(row)
        else:
            image_changed = row.get("image_url") != existing.get("image_url")
            other_changed = (
                row.get("title") != existing.get("title") or
                row.get("price") != existing.get("price") or
                row.get("sale") != existing.get("sale") or
                row.get("description") != existing.get("description") or
                row.get("category") != existing.get("category") or
                row.get("size") != existing.get("size") or
                str(row.get("tags")) != str(existing.get("tags"))
            )
            if image_changed:
                updated_count += 1
                _generate_embeddings(row, processor, tokenizer, model, device, i)
                rows_to_upsert.append(row)
            elif other_changed:
                updated_count += 1
                row["image_embedding"] = existing.get("image_embedding")
                row["info_embedding"] = existing.get("info_embedding")
                rows_to_upsert.append(row)
            else:
                unchanged_count += 1
    
    print(f"\nSummary: {new_count} new, {updated_count} updated, {unchanged_count} unchanged")
    
    if rows_to_upsert:
        print(f"Upserting {len(rows_to_upsert)} products to Supabase...")
        success, errors = upsert_products_batch(rows_to_upsert, batch_size=BATCH_SIZE)
        print(f"Upserted {success}/{len(rows_to_upsert)} products.")
        if errors:
            for e in errors[:20]:
                print(f"  Error: {e}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more errors.")
    
    print("Updating missing runs...")
    update_missing_runs(seen_urls, SOURCE)
    
    print("Cleaning up stale products...")
    deleted = delete_stale_products(seen_urls, SOURCE, max_missing_runs=2)
    print(f"Deleted {deleted} stale products.")
    
    print("\n" + "="*50)
    print("RUN COMPLETE")
    print("="*50)
    print(f"  New products added:      {new_count}")
    print(f"  Products updated:        {updated_count}")
    print(f"  Products unchanged:      {unchanged_count}")
    print(f"  Stale products deleted:  {deleted}")
    print("="*50)
    
    return new_count, updated_count, unchanged_count, deleted


def _generate_embeddings(row, processor, tokenizer, model, device, index):
    """Generate embeddings for a product with staggered delay."""
    if index > 0:
        time.sleep(0.5)
    
    image_url = row.get("image_url")
    if image_url:
        emb = image_embedding_from_url(image_url, processor, model, device)
        row["image_embedding"] = emb
    else:
        row["image_embedding"] = None
    
    info_text = build_info_text(row)
    if info_text:
        txt_emb = text_embedding(info_text, tokenizer, model, device)
        row["info_embedding"] = txt_emb
    else:
        row["info_embedding"] = None


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    limit = None
    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=")[1])
    run(dry_run=dry, limit=limit)
