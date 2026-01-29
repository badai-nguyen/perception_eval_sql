import json
import os
from typing import Any, Callable, Dict, Optional

CONFIG_FILE = os.path.expanduser("./configs/autoware_evaluator_dl_config.json")


def load_user_config(config_file: str = CONFIG_FILE) -> Dict[str, Any]:
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_user_config(
    config: Dict[str, Any],
    config_file: str = CONFIG_FILE,
    warning_fn: Optional[Callable[[str], None]] = None,
) -> None:
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        if warning_fn is not None:
            warning_fn(f"Could not save config: {e}")


class UserConfig:
    def __init__(
        self,
        *,
        config_file: str = CONFIG_FILE,
        warning_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._config_file = config_file
        self._warning_fn = warning_fn
        self._config = load_user_config(config_file)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if self._config.get(key) != value:
            self._config[key] = value
            save_user_config(
                self._config,
                config_file=self._config_file,
                warning_fn=self._warning_fn,
            )
