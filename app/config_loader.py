"""
NeuralRP Configuration Loader

Loads configuration from config.yaml with environment variable overrides.
Environment variables override YAML settings with format:
  NEURALRP_{SECTION}_{KEY}

Example:
  NEURALRP_SERVER_PORT=9000
  NEURALRP_KOBOLD_URL=http://localhost:5001
"""

import os
import yaml
from typing import Dict, Any, Optional

# Default configuration (fallback if config.yaml missing)
DEFAULT_CONFIG = {
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "fallback_ports": [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010],
        "cors_origins": ["*"],
        "log_level": "INFO"
    },
    "kobold": {
        "url": "http://127.0.0.1:5001"
    },
    "stable_diffusion": {
        "url": "http://127.0.0.1:7861",
        "timeout": 10.0
    },
    "context": {
        "max_context": 8192,
        "summarize_threshold": 0.90,
        "summarize_trigger_turn": 10,
        "history_window": 5,
        "max_exchanges_per_scene": 15,
        "world_info_reinforce_freq": 3
    },
    "retention": {
        "backup_enabled": True,
        "backup_schedule": "daily",
        "backup_retention_days": 30,
        "unnamed_chat_days": 30,
        "autosaved_chat_days": 7,
        "change_log_days": 30,
        "performance_metrics_days": 7,
        "summarized_messages_days": 90,
        "log_retention_days": 30,
        "vacuum_interval_days": 7
    },
    "features": {
        "performance_mode_enabled": True
    },
    "system_prompt": "Write a highly detailed, creative, and immersive response. Stay in character at all times."
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from config.yaml with environment variable overrides.

    Args:
        config_path: Path to config.yaml file (optional, auto-detected if not provided)

    Returns:
        Complete configuration dictionary with nested sections
    """
    # Start with defaults
    config = DEFAULT_CONFIG.copy()

    # Load from YAML file if it exists
    if config_path is None:
        # Auto-detect config.yaml location
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config.yaml")

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    # Deep merge: YAML overrides defaults
                    config = _deep_merge(config, yaml_config)
        except Exception as e:
            print(f"[CONFIG WARNING] Failed to load config.yaml: {e}")
            print(f"[CONFIG] Using default configuration")
    else:
        print(f"[CONFIG] config.yaml not found at {config_path}")
        print(f"[CONFIG] Using default configuration")
        print(f"[CONFIG] Create config.yaml to customize settings")

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary (values take precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.

    Format: NEURALRP_{SECTION}_{KEY}

    Examples:
        NEURALRP_SERVER_PORT=9000
        NEURALRP_KOBOLD_URL=http://localhost:5001
        NEURALRP_CONTEXT_MAX_CONTEXT=4096
    """
    env_prefix = "NEURALRP_"

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(env_prefix):
            continue

        # Parse env key: NEURALRP_SECTION_KEY -> ["SECTION", "KEY"]
        parts = env_key[len(env_prefix):].lower().split('_')

        if len(parts) < 2:
            continue  # Skip malformed env vars

        # Navigate to nested config section
        target = config
        for i, part in enumerate(parts[:-1]):
            if part not in target:
                break  # Invalid path
            target = target[part]

        # Set final key
        final_key = parts[-1]

        # Type conversion based on existing config value
        if final_key in target:
            if isinstance(target[final_key], bool):
                target[final_key] = env_value.lower() in ('true', '1', 'yes')
            elif isinstance(target[final_key], int):
                target[final_key] = int(env_value)
            elif isinstance(target[final_key], float):
                target[final_key] = float(env_value)
            else:
                target[final_key] = env_value

    return config


# Global config instance (loaded once on import)
CONFIG = load_config()
