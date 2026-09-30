from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    return cfg
