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

    deepseek_base_url: str = "https://openrouter.ai/api/v1"
    deepseek_api_key: str

    groq_api_key: str
    gemini_api_key: str

    @property
    def cors_origins(self) -> list[str]:
        return [self.client_url, self.local_client_url]


settings = Settings()
