"""Tests for the shared application configuration boundary."""

import pytest

from algo_trading.config import BotConfig


def test_bot_config_uses_independent_strategy_params():
    first = BotConfig()
    second = BotConfig()

    first.strategy_params["fast"] = 10

    assert second.strategy_params == {}


def test_bot_config_normalizes_runtime_values():
    config = BotConfig(mode="PAPER", exchange="OKX")

    assert config.mode == "paper"
    assert config.exchange == "okx"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "unsafe"),
        ("exchange", "unsupported"),
        ("risk_per_trade", 0),
        ("risk_per_trade", 1.1),
        ("history_limit", 1),
        ("max_dca_orders", 0),
    ],
)
def test_bot_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        BotConfig(**{field: value})


def test_okx_adapter_does_not_import_bot_orchestrator():
    import algo_trading.live.okx_client as okx_module

    assert okx_module.BotConfig is BotConfig
