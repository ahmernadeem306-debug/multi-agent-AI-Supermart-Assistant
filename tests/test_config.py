from __future__ import annotations

import pytest
from pydantic import SecretStr

import app.config as config_module
from app.core.exceptions import ConfigError
from pydantic_settings import SettingsConfigDict


def test_valid_settings_load():
    settings = config_module.get_settings()
    assert settings.groq_model
    assert isinstance(settings.groq_api_key, SecretStr)
    assert settings.database_url


def test_missing_key_raises_config_error(monkeypatch):
    # Ignore any real .env file on disk so this test is isolated from the
    # developer's local configuration.
    monkeypatch.setattr(
        config_module.Settings,
        "model_config",
        SettingsConfigDict(env_file=None, env_prefix="", extra="ignore"),
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "_settings", None)

    with pytest.raises(ConfigError):
        config_module.get_settings()


def test_key_never_exposed_by_repr_or_str():
    settings = config_module.get_settings()
    secret_value = settings.groq_api_key.get_secret_value()

    assert secret_value not in repr(settings.groq_api_key)
    assert secret_value not in str(settings.groq_api_key)
    assert secret_value not in repr(settings)
