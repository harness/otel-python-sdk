"""Boolean SDK flag resolution: is_env_flag_enabled across prefix aliases."""
import os

import pytest

from harness_sdk.env import is_env_flag_enabled

_KEY = "ENABLE_CONSOLE_SPAN_EXPORTER"
_PREFIXES = ("HARNESS_", "HA_", "AT_", "TA_")


@pytest.fixture(autouse=True)
def clear_flag():
    for prefix in _PREFIXES:
        os.environ.pop(f"{prefix}{_KEY}", None)
    yield
    for prefix in _PREFIXES:
        os.environ.pop(f"{prefix}{_KEY}", None)


def test_unset_flag_is_disabled():
    assert is_env_flag_enabled(_KEY) is False


def test_true_enables_flag():
    os.environ[f"HARNESS_{_KEY}"] = "true"
    assert is_env_flag_enabled(_KEY) is True


def test_explicit_false_disables_flag():
    os.environ[f"HARNESS_{_KEY}"] = "false"
    assert is_env_flag_enabled(_KEY) is False


def test_arbitrary_value_does_not_enable_flag():
    os.environ[f"HARNESS_{_KEY}"] = "1"
    assert is_env_flag_enabled(_KEY) is False
    os.environ[f"HARNESS_{_KEY}"] = "yes"
    assert is_env_flag_enabled(_KEY) is False


def test_empty_value_does_not_enable_flag():
    os.environ[f"HARNESS_{_KEY}"] = ""
    assert is_env_flag_enabled(_KEY) is False


def test_legacy_aliases_enable_flag():
    for prefix in ("HA_", "AT_", "TA_"):
        os.environ[f"{prefix}{_KEY}"] = "true"
        assert is_env_flag_enabled(_KEY) is True
        del os.environ[f"{prefix}{_KEY}"]


def test_harness_prefix_wins_over_legacy_alias():
    os.environ[f"HARNESS_{_KEY}"] = "false"
    os.environ[f"HA_{_KEY}"] = "true"
    assert is_env_flag_enabled(_KEY) is False
