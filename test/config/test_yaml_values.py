import os

import pytest

from harness_sdk.config.config import Config, build_config
from harness_sdk.config.yaml_values import coerce_bool, coerce_string, normalize_config_dict


def prepare():
    if hasattr(Config, '_instance'):
        del Config._instance


def reset():
    keys_to_delete = [key for key in os.environ if key.startswith('HA_')]
    for key in keys_to_delete:
        del os.environ[key]
    if hasattr(Config, '_instance'):
        Config._instance = None


@pytest.mark.parametrize(
    'value,expected',
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ('true', True),
        ('TRUE', True),
        ('yes', True),
        ('no', False),
        ('on', True),
        ('off', False),
        ('1', True),
        ('0', False),
    ],
)
def test_coerce_bool(value, expected):
    assert coerce_bool(value) is expected


def test_coerce_bool_invalid():
    with pytest.raises(ValueError):
        coerce_bool('maybe')


def test_coerce_string():
    assert coerce_string(123) == '123'
    assert coerce_string(1.5) == '1.5'
    assert coerce_string('hello') == 'hello'


def test_normalize_config_dict_service_name():
    config = {'service_name': 123}
    normalize_config_dict(config, {'service_name': 'otel-sdk'})
    assert config['service_name'] == '123'


def test_build_config_service_name_int(tmp_path):
    prepare()
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('service_name: 123\n', encoding='utf-8')
    os.environ['HA_CONFIG_FILE'] = str(config_file)

    agent_config, *_ = build_config()
    assert agent_config.service_name.value == '123'
    reset()


@pytest.mark.parametrize(
    'yaml_text,field,expected',
    [
        ('gen_ai:\n  enabled: 1\n', 'enabled', True),
        ('gen_ai:\n  enabled: "yes"\n', 'enabled', True),
        ('gen_ai:\n  enabled: "no"\n', 'enabled', False),
        ('gen_ai:\n  payload_capture_enabled: 1\n', 'payload_capture_enabled', True),
        ('gen_ai:\n  payload_evaluation_enabled: "yes"\n', 'payload_evaluation_enabled', True),
        ('gen_ai:\n  payload_evaluation_enabled: 1\n', 'payload_evaluation_enabled', True),
    ],
)
def test_build_config_gen_ai_bool_coercion(tmp_path, yaml_text, field, expected):
    prepare()
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml_text, encoding='utf-8')
    os.environ['HA_CONFIG_FILE'] = str(config_file)

    agent_config, *_ = build_config()
    assert getattr(agent_config.gen_ai, field).value is expected
    reset()
