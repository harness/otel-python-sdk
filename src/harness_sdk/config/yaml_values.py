"""Coerce YAML-loaded values to protobuf-compatible JSON types."""

_TRUE_STRINGS = frozenset({'true', 'yes', 'on', '1'})
_FALSE_STRINGS = frozenset({'false', 'no', 'off', '0'})


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    raise ValueError(f'cannot coerce {value!r} to bool')


def coerce_string(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return value


def normalize_config_dict(config_dict, template):
    if not isinstance(config_dict, dict) or not isinstance(template, dict):
        return config_dict
    for key, value in list(config_dict.items()):
        if key not in template:
            continue
        template_value = template[key]
        if isinstance(template_value, dict) and isinstance(value, dict):
            normalize_config_dict(value, template_value)
        elif isinstance(template_value, bool):
            config_dict[key] = coerce_bool(value)
        elif isinstance(template_value, str) and not isinstance(value, str):
            config_dict[key] = coerce_string(value)
    return config_dict
