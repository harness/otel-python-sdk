import copy
import json
import os

import yaml

from google.protobuf import json_format

from harness_sdk.config import config_pb2
from harness_sdk.config.default import DEFAULT, SDK_CONFIG_KEYS
from harness_sdk.config.environment import overwrite_with_environment
from harness_sdk.otlp_reporting import normalize_reporting_dict
from harness_sdk.custom_logger import get_custom_logger

logger = get_custom_logger(__name__)


def _parse_plugin_env(env_key: str) -> list | None:
    raw = os.environ.get(env_key)
    if not raw:
        return None
    return [name.strip() for name in raw.split(',') if name.strip()]


def _ordered_plugin_names_from_section(section) -> list:
    if section is None:
        return []
    if isinstance(section, list):
        return [str(name).strip() for name in section if str(name).strip()]
    if isinstance(section, dict):
        return [
            name for name, cfg in section.items()
            if _is_plugin_entry_enabled(cfg)
        ]
    return []


def _is_plugin_entry_enabled(cfg) -> bool:
    if isinstance(cfg, dict):
        return cfg.get('enabled', True)
    return bool(cfg)


def _filter_sdk_config(file_dict):
    if not file_dict:
        return {}
    return {key: file_dict[key] for key in SDK_CONFIG_KEYS if key in file_dict}


class Config:  # pylint:disable=R0903
    _instance = None
    _singleton_lock = __import__('threading').Lock()

    def __new__(cls):
        if getattr(cls, '_instance', None) is None:
            with cls._singleton_lock:
                if getattr(cls, '_instance', None) is None:
                    cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            (
                self.config,
                self.plugins_config,
                self.enabled_control_plugins,
                self.enabled_observability_plugins,
                self.reporting_encoding,
            ) = build_config()

    def get_plugin_option(self, plugin_name: str, option: str, default=None, plugin_type="control"):
        section = self.plugins_config.get(plugin_type, {})
        if plugin_name in section and isinstance(section[plugin_name], dict):
            return section[plugin_name].get(option, default)
        return default


def merge_config(base_config, overriding_config):
    for key in overriding_config:
        if key in base_config and isinstance(base_config[key], dict):
            base_config[key] = merge_config(base_config[key], overriding_config[key])
        else:
            base_config[key] = overriding_config[key]
    return base_config


def build_config():
    agent_config = config_pb2.AgentConfig()
    config_dict = copy.deepcopy(DEFAULT)
    file_dict = read_from_file()
    if file_dict is not None:
        merge_config(config_dict, _filter_sdk_config(file_dict))

    plugins_config = config_dict.pop('plugins', {})

    enabled_control_plugins = _parse_plugin_env('HA_CONTROL_PLUGINS')
    if enabled_control_plugins is None:
        enabled_control_plugins = _parse_plugin_env('AT_CONTROL_PLUGINS')
    if enabled_control_plugins is None:
        enabled_control_plugins = _ordered_plugin_names_from_section(
            plugins_config.get('control')
        )
    if not enabled_control_plugins:
        enabled_control_plugins = []

    enabled_observability_plugins = _parse_plugin_env('HA_OBSERVABILITY_PLUGINS')
    if enabled_observability_plugins is None:
        enabled_observability_plugins = _parse_plugin_env('AT_OBSERVABILITY_PLUGINS')
    if enabled_observability_plugins is None:
        enabled_observability_plugins = _ordered_plugin_names_from_section(
            plugins_config.get('observability')
        )
    if not enabled_observability_plugins:
        enabled_observability_plugins = ['builtin_pipeline', 'builtin_span_attributes']

    reporting_encoding = None
    if 'reporting' in config_dict:
        reporting_encoding = normalize_reporting_dict(config_dict['reporting'])

    json_string = json.dumps(config_dict)
    json_format.Parse(json_string, agent_config, ignore_unknown_fields=True)

    agent_config, reporting_encoding = overwrite_with_environment(
        agent_config, reporting_encoding
    )
    logger.debug(json_string)
    return (
        agent_config,
        plugins_config,
        enabled_control_plugins,
        enabled_observability_plugins,
        reporting_encoding,
    )


def read_from_file():
    config_path = _config_file_path()
    if config_path is None or not os.path.exists(config_path):
        logger.debug("HA_CONFIG_FILE path not set")
        return None

    with open(config_path, 'r', encoding="UTF-8") as config_file:
        try:
            return yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            logger.debug(exc)
            return None


def _config_file_path():
    return (
        os.environ.get('HA_CONFIG_FILE')
        or os.environ.get('AT_CONFIG_FILE')
        or os.environ.get('TA_CONFIG_FILE')
    )
