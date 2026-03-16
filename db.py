"""Supabase operations for products table."""
import logging
import time
from datetime import datetime
from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_ANON_KEY, SOURCE, EMBEDDING_DIM

logger = logging.getLogger(__name__)


def get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def _valid_vector(vec):
    """Return 768-dim list for pgvector if valid; else None."""
    if vec is None or not isinstance(vec, (list, tuple)):
        return None
    if len(vec) != EMBEDDING_DIM:
        return None
    try:
        return [float(x) for x in vec]
    except (TypeError, ValueError):
        return None


def _row_to_payload(row: dict, include_embeddings: bool = True) -> dict:
    """Convert row dict to payload for upsert."""
    payload = {
        "id": row["id"],
        "source": row["source"],
        "product_url": row["product_url"],
        "affiliate_url": row.get("affiliate_url"),
        "image_url": row["image_url"],
        "brand": row["brand"],
        "title": row["title"],
        "description": row.get("description"),
        "category": row.get("category"),
        "gender": row.get("gender"),
        "metadata": row.get("metadata"),
        "size": row.get("size"),
        "second_hand": row.get("second_hand", False),
        "country": row.get("country"),
        "compressed_image_url": row.get("compressed_image_url"),
        "tags": row.get("tags"),
        "other": row.get("other"),
        "price": row.get("price"),
        "sale": row.get("sale"),
        "additional_images": row.get("additional_images"),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    if "created_at" in row:
        payload["created_at"] = row["created_at"]
    else:
        payload["created_at"] = datetime.utcnow().isoformat() + "Z"
    
    if include_embeddings:
        img_emb = _valid_vector(row.get("image_embedding"))
        if img_emb is not None:
            payload["image_embedding"] = img_emb
        info_emb = _valid_vector(row.get("info_embedding"))
        if info_emb is not None:
            payload["info_embedding"] = info_emb
    return payload


def _has_changed(existing: dict, new_row: dict) -> bool:
    """Check if any meaningful field has changed."""
    fields = ["title", "description", "price", "sale", "image_url", 
              "additional_images", "category", "size", "tags"]
    for field in fields:
        existing_val = existing.get(field)
        new_val = new_row.get(field)
        if str(existing_val or "") != str(new_val or ""):
            return True
    return False


def get_existing_products(source: str = SOURCE) -> dict[str, dict]:
    """Fetch all existing products for a source. Returns dict keyed by product_url."""
    client = get_client()
    try:
        response = client.table("products").select("*").eq("source", source).execute()
        return {p["product_url"]: p for p in response.data}
    except Exception as e:
        logger.warning("Failed to fetch existing products: %s", e)
        return {}


def get_product_missing_runs(source: str = SOURCE) -> dict[str, int]:
    """Get products and their consecutive missing run counts."""
    client = get_client()
    try:
        response = client.table("products").select("product_url, missing_runs").eq("source", source).execute()
        return {p["product_url"]: p.get("missing_runs", 0) or 0 for p in response.data}
    except Exception as e:
        logger.warning("Failed to fetch missing runs: %s", e)
        return {}


def upsert_products_batch(
    rows: list[dict],
    batch_size: int = 50,
    max_retries: int = 3,
) -> tuple[int, list[str]]:
    """
    Upsert product rows in batches.
    Returns (success_count, list of error messages).
    """
    if not rows:
        return 0, []
    
    total_success = 0
    all_errors = []
    
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        payload = [_row_to_payload(row) for row in batch]
        
        for retry in range(max_retries):
            try:
                client = get_client()
                client.table("products").upsert(
                    payload,
                    on_conflict="source,product_url",
                ).execute()
                total_success += len(batch)
                break
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(1)
                    continue
                for row in batch:
                    all_errors.append(f"{row.get('id', '?')}: {e}")
    
    return total_success, all_errors


def upsert_products(
    rows: list[dict],
    batch_size: int = 50,
    max_retries: int = 3,
) -> tuple[int, list[str]]:
    """Legacy wrapper for backwards compatibility."""
    return upsert_products_batch(rows, batch_size, max_retries)


def delete_stale_products(
    seen_urls: set[str],
    source: str = SOURCE,
    max_missing_runs: int = 2,
) -> int:
    """
    Delete products that have been missing for 2+ consecutive runs.
    Returns count of deleted products.
    """
    client = get_client()
    
    try:
        response = client.table("products").select("product_url, missing_runs").eq("source", source).execute()
        to_delete = []
        for p in response.data:
            url = p.get("product_url")
            missing = p.get("missing_runs", 0) or 0
            if url and url not in seen_urls and missing >= max_missing_runs:
                to_delete.append(url)
        
        if to_delete:
            client.table("products").delete().in_("product_url", to_delete).execute()
            return len(to_delete)
    except Exception as e:
        logger.warning("Failed to delete stale products: %s", e)
    return 0


def update_missing_runs(seen_urls: set[str], source: str = SOURCE) -> None:
    """Increment missing_runs for unseen products, reset for seen products."""
    client = get_client()
    try:
        all_products = client.table("products").select("product_url, missing_runs").eq("source", source).execute()
        
        for p in all_products.data:
            url = p.get("product_url")
            if not url:
                continue
            if url in seen_urls:
                client.table("products").update({"missing_runs": 0}).eq("product_url", url).execute()
            else:
                current_missing = p.get("missing_runs", 0) or 0
                client.table("products").update({"missing_runs": current_missing + 1}).eq("product_url", url).execute()
    except Exception as e:
        logger.warning("Failed to update missing runs: %s", e)
