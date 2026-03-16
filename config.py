"""Configuration for Fear of God scraper."""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://fearofgod.com"
LOCALE = "en-be"
BASE_WITH_LOCALE = f"{BASE_URL}/{LOCALE}"

COLLECTION_URLS = [
    f"{BASE_WITH_LOCALE}/collections/mens-all",
    f"{BASE_WITH_LOCALE}/collections/womens-all",
]

PRODUCTS_PER_PAGE = 50
SOURCE = "scraper-fearofgod"
BRAND = "Fear of God"

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://yqawmzggcgpeyaaynrjk.supabase.co")
SUPABASE_ANON_KEY = os.getenv(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4",
)

SIGLIP_MODEL = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768
BATCH_SIZE = 50
