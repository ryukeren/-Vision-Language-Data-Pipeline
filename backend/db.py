"""
Supabase Cloud client — initialized once at module load using service-role key.
Import `supabase_client` anywhere in the application to perform table operations.
"""
import os
from supabase import create_client, Client
from config import settings

# Bulletproof Supabase Initialization
supabase_url = os.environ.get("SUPABASE_URL", settings.supabase_url)
# The user's .env has a placeholder for the anon key, but a real token for the service role key.
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", settings.supabase_service_role_key)

if supabase_url and supabase_key and supabase_key != "your-anon-key-here":
    supabase_client: Client = create_client(supabase_url, supabase_key)
else:
    supabase_client = None
    print("[WARNING] Supabase credentials missing or invalid. Database writes will fail.")

def get_supabase() -> Client:
    """Backwards-compatible accessor — returns the global client instance."""
    return supabase_client
