import os

# Prefix precedence for resolving SDK settings: new HARNESS_ first, then the
# legacy HA_/AT_/TA_ aliases for backwards compatibility.
_PREFIXES = ("HARNESS_", "HA_", "AT_", "TA_")


def get_env_value(target_key):
    """Read SDK env vars honoring prefix precedence: HARNESS_, then legacy HA_/AT_/TA_."""
    for prefix in _PREFIXES:
        env_var_key = f"{prefix}{target_key}"
        if env_var_key in os.environ:
            return os.environ[env_var_key]
    return None


def is_env_flag_enabled(target_key):
    """Boolean SDK flag under any supported prefix (HARNESS_ > HA_ > AT_ > TA_).

    Returns True only when the resolved value is case-insensitively 'true'.
    """
    value = get_env_value(target_key)
    return value is not None and value.strip().lower() == "true"


def is_harness_flag_enabled(env_var_name):
    """Opt-in flag under HARNESS_ or HA_ (HARNESS_ wins when both are set).

    Enable flags never existed under AT_/TA_, so only those two prefixes are
    honored. Returns True only when the resolved value is present and
    case-insensitively 'true'.
    """
    for key in _flag_keys(env_var_name):
        if key in os.environ:
            return os.environ[key].strip().lower() == "true"
    return False


def is_enable_flag_present(env_var_name):
    """Presence check for an opt-in flag under HARNESS_ or HA_."""
    return any(key in os.environ for key in _flag_keys(env_var_name))


_FLAG_PREFIXES = ("HARNESS_", "HA_")


def _flag_keys(env_var_name):
    suffix = (
        env_var_name[len("HARNESS_"):]
        if env_var_name.startswith("HARNESS_")
        else env_var_name
    )
    return [f"{prefix}{suffix}" for prefix in _FLAG_PREFIXES]
