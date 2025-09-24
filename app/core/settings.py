# File: src/wal_fact_checker/core/settings.py
"""Application settings and configuration."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    GoogleSecretManagerSettingsSource,
    PydanticBaseSettingsSource,
)

load_dotenv(".env", override=True)


class AppSettings(BaseSettings):
    """Application-wide settings and configuration."""

    scrape_do_token: str = Field(escription="API key for scrape.do service")

    # General application settings
    app_name: str = Field(default="WAL Fact Checker", description="Application name")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # HTTP client settings
    default_timeout: float = Field(
        default=60.0, description="Default HTTP request timeout in seconds"
    )
    max_retries: int = Field(default=3, description="Maximum number of HTTP retries")

    # Content processing settings
    max_content_length: int = Field(
        default=10000, description="Maximum content length for processing"
    )

    groq_api_key: str = Field(default="", description="Groq API key")

    google_api_key: str = Field(alias="gcp_genai_key", default="", description="Google API key")

    langfuse_host: str = Field(default="", description="Langfuse host")
    langfuse_public_key: str = Field(default="", description="Langfuse public key")
    langfuse_secret_key: str = Field(default="", description="Langfuse secret key")
    langfuse_tracing_environment: str = Field(default="", description="Langfuse tracing environment")

    # Model configuration
    GEMINI_2_5_FLASH_MODEL: str = Field(
        default="gemini-2.5-flash", description="Gemini 2.5 Flash model name"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        project_id = os.getenv("GCP_PROJECT_ID")
        gcp_settings = GoogleSecretManagerSettingsSource(
            settings_cls,
            project_id=project_id,
        )
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            gcp_settings,
            file_secret_settings,
        )


# Create application settings instance
settings = AppSettings()

