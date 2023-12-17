from pathlib import Path

from dotenv import load_dotenv


def get_project_root() -> Path:
    import git

    return Path(git.Repo(".", search_parent_directories=True).working_tree_dir)  # type: ignore


ROOT = get_project_root()
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


def load_env():
    return load_dotenv(ROOT / ".env")
