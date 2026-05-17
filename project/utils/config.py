from dataclasses import dataclass
from pathlib import Path


@dataclass
class DriftConfig:
    train_end: int = 2005
    gap_end: int = 2010

    alpha: float = 0.05
    drift_fraction_threshold: float = 0.05

    latent_dim: int = 64
    hidden_1: int = 512
    hidden_2: int = 384
    hidden_3: int = 256
    hidden_4: int = 128
    hidden_5: int = 96
    n_epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3

    swd_n_projections: int = 512
    swd_bootstrap_runs: int = 200
    swd_max_samples: int = 10_000
    min_samples_per_year: int = 10

    seed: int = 42
    threshold_std_multiplier: float = 0.5

    output_dir: Path = Path("outputs")

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    def create_output_dirs(self) -> None:
        for directory in [
            self.figures_dir,
            self.tables_dir,
            self.models_dir,
            self.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
