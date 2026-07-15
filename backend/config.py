"""
Application configuration using Pydantic Settings (v2)
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # silently skip undeclared env vars
    )

    # Gemini
    gemini_api_key: str = ""

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""

    # Database (Transaction Pooler — IPv4-compatible)
    database_url: str = ""

    # LangSmith / LangGraph
    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_project: str = "vLP-Agentic-Workflow"
    langsmith_endpoint: str = "https://apac.api.smith.langchain.com"

    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    app_api_key: str = ""  # Set this in .env as APP_API_KEY=your-secret-key


settings = Settings()
