from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def data_directory():
    root = get_project_root()
    return root / 'data' / 'dataset'
