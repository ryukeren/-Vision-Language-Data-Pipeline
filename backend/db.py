"""
Supabase Cloud client — initialized once at module load using service-role key.
Import `supabase_client` anywhere in the application to perform table operations.
"""
from supabase import create_client, Client
from config import settings

# Eagerly initialize a single global client instance at startup.
# The service-role key bypasses Row Level Security — safe for server-side writes only.
supabase_client: Client = create_client(
    settings.supabase_url,
    settings.supabase_service_role_key,
)


def get_supabase() -> Client:
    """Backwards-compatible accessor — returns the global client instance."""
    return supabase_client
