"""Where data lives.

Code lives in the git repo; data lives outside it (Google Drive when running in
Colab), so downloads survive Colab restarts and never end up in git.
"""
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data_store"


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def cache_dir(self) -> Path:
        """FastF1's HTTP cache (raw API responses)."""
        return self.root / "fastf1_cache"

    @property
    def quali_dir(self) -> Path:
        """One parquet file per qualifying session."""
        return self.root / "raw" / "quali"

    def ensure(self) -> "Paths":
        for d in (self.cache_dir, self.quali_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(data_dir: str | Path | None = None) -> Paths:
    """Resolve the data directory: argument > F1SIM_DATA_DIR env var > ./data_store."""
    root = data_dir or os.environ.get("F1SIM_DATA_DIR") or DEFAULT_DATA_DIR
    return Paths(Path(root)).ensure()
