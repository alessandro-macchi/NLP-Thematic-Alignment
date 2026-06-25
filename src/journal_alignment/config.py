"""Project configuration for the journal alignment analysis."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Central configuration for the thematic journal alignment project.

    Default paths are relative to the project root. The defaults assume commands
    are run from the top-level ``nlp/`` directory.
    """

    articles_path: Path = field(
        default_factory=lambda: Path("../data/raw/articles_pubmed.csv")
    )
    aims_scope_path: Path = field(
        default_factory=lambda: Path("../data/raw/aims_scope.txt")
    )
    processed_dir: Path = field(default_factory=lambda: Path("../data/processed"))
    results_dir: Path = field(default_factory=lambda: Path("../data/results"))
    figures_dir: Path = field(default_factory=lambda: Path("../reports/figures"))
    tables_dir: Path = field(default_factory=lambda: Path("../reports/tables"))
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_n: int = 10
    min_abstract_length: int = 50
    outlier_method: str = "z_score"
    z_threshold: float = 2.0
    random_seed: int = 42
