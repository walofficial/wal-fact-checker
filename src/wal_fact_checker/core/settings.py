# File: src/wal_fact_checker/core/settings.py
"""Application settings and configuration."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application-wide settings and configuration."""

    # Scrape.do API settings
    scrape_api_url: str = Field(
        default="https://api.scrape.do", description="Base URL for scrape.do API"
    )
    scrape_api_key: str = Field(..., description="API key for scrape.do service")

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

    model_config = SettingsConfigDict(
        env_prefix="WAL_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )


# Create application settings instance
settings = AppSettings()
