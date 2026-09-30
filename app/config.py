"""Application Configuration using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    # Customize the Settings
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Defining Variables and their Types

    # OpenAI Configuration
    openai_api_key : str

    # Qdrant Cloud Configuration
    qdrant_url : str
    qdrant_api_key : str

    # Collection Settings
    colllection_name : str = "rag_documents"

    # Document Processing Settings
    chunk_size : int = 1000
    chunk_overlap : int = 200

    # Model Configuration
    embedding_model : str = "text-embedding-3-small"
    llm_model : str = "gpt-4o-mini"
    llm_temperature : float = 0.0

    # Retrieval Settings
    retrieval_k : int = 5

    # Logging Level
    log_level : str = "INFO"


def get_settings() -> Settings:

    # Get Settings instance
    return Settings()