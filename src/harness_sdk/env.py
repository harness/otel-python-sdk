import os


def get_env_value(target_key):
    """Read SDK env vars: HA_<key>, then legacy AT_/TA_ aliases."""
    for prefix in ("HA_", "AT_", "TA_"):
        env_var_key = f"{prefix}{target_key}"
        if env_var_key in os.environ:
            return os.environ[env_var_key]
    return None
