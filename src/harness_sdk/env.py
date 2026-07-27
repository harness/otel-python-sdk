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


def is_env_var_present(target_key):
    """Return True if the key is set under any supported prefix (presence check)."""
    return any(f"{prefix}{target_key}" in os.environ for prefix in _PREFIXES)


def is_harness_flag_enabled(env_var_name):
    """Strict opt-in flag under the HARNESS_ prefix only (no legacy aliases).

    Returns True only when the value is present and case-insensitively 'true'.
    """
    value = os.environ.get(env_var_name)
    return value is not None and value.strip().lower() == "true"
