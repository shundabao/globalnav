"""Load API keys from GLOBALNAV or VELMA .env files."""

import os
from pathlib import Path


def _parse_env_file(path: Path) -> None:
    if not path.exists():
        return
    values = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value:
            values[key] = value
    for key, value in values.items():
        if key not in os.environ:
            os.environ[key] = value


def load_env() -> None:
    """Load .env from GLOBALNAV first, then VELMA (shared key location)."""
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / '.env',
        root.parent / 'VELMA' / '.env',
        Path('/data/shzheng/VELMAFLAME/VELMA/.env'),
        Path('/home/shzheng/Data/VELMAFLAME/VELMA/.env'),
    ]
    for path in candidates:
        _parse_env_file(path)


def get_openai_api_key() -> str:
    load_env()
    key = os.environ.get('OPENAI_API_KEY', '')
    if not key:
        raise ValueError(
            'OPENAI_API_KEY not set. Copy VELMA/.env.example to VELMA/.env and add your key.'
        )
    return key


DEFAULT_MODEL = 'gpt-4o-mini'


def get_google_maps_api_key() -> str:
    load_env()
    return (
        os.environ.get('GOOGLE_MAPS_API_KEY')
        or os.environ.get('GOOGLE_MAPS_KEY')
        or os.environ.get('GOOGLE_API_KEY')
        or ''
    )
