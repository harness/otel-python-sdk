"""Tests for harness-sdk sitecustomize autoinstrumentation bootstrap."""

import os
import sys

import pytest

from harness_sdk.config.config import Config
from harness_sdk.plugins.control import get_control_registry


def _clear_ha_env():
    for key in list(os.environ):
        if key.startswith("HA_"):
            del os.environ[key]
    Config._instance = None
    get_control_registry().clear()


def test_agent_initializes_from_sitecustomize():
    if sys.platform == 'darwin':
        pytest.skip('sitecustomize integration test skipped on darwin in CI')
    _clear_ha_env()
    os.environ['HA_ENABLE_CONSOLE_SPAN_EXPORTER'] = 'true'
    from harness_sdk.autoinstrumentation import sitecustomize  # pylint:disable=C0415,W0611
    from harness_sdk.agent import Agent
    agent = Agent()
    assert agent.is_initialized() or not agent.is_enabled()
    _clear_ha_env()
