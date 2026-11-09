from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    port: int = 8000
    client_url: str
    local_client_url: str = "http://localhost:3000"
    localhost: str = "http://127.0.0.1:8000"
    environment: str = "development"

    nextauth_secret: str

    database_url: str

    ollama_host: str = "https://ollama.com"
    ollama_api_key: str
    llm_model: str = "gpt-oss:120b"
    models_cache_ttl_seconds: int = 3600
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    embed_device: str = "auto"  # auto | cuda | cpu
    chunk_size: int = 500  # tokens
    chunk_overlap: int = 50  # tokens

    # Original uploaded PDFs: S3-compatible storage (Cloudflare R2)
    # may include the bucket: https://<account>.r2.cloudflarestorage.com/<bucket>
    s3_endpoint: str | None = None
    s3_bucket: str | None = None  # overrides the bucket from S3_ENDPOINT
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_region: str = "auto"  # R2 uses "auto"
    file_url_ttl_s: int = 3600  # lifetime of presigned links to view a file
    default_system_prompt: str = "You are a helpful coding assistant."

    # RAG quality
    rewrite_model: str | None = None  # model for query rewriting; None = llm_model
    rewrite_timeout_s: float = 8.0
    rag_min_score: float = 0.4  # drop retrieved chunks less similar than this (cosine)
    # documents up to this size go to the model whole (summaries, "what is it about");
    # bigger ones are searched. 0 = always search.
    rag_full_context_max_tokens: int = 12000
    history_max_tokens: int = 6000  # token budget for the chat history sent to the LLM

    @property
    def cors_origins(self) -> list[str]:
        return [self.client_url, self.local_client_url]


settings = Settings()
