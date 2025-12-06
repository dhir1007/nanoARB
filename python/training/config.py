"""Training configuration for NanoARB models."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: Path = Path("data/processed")
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    sequence_length: int = 100
    num_levels: int = 10
    features_per_level: int = 4
    prediction_horizons: list[int] = field(default_factory=lambda: [10, 50, 100])
    batch_size: int = 256
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = "mamba"  # mamba, transformer, tft
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.1
    # Mamba specific
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    # Transformer specific
    num_heads: int = 8
    ff_dim: int = 512


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    device: str = "cuda"
    mixed_precision: bool = True
    seed: int = 42


@dataclass
class RLConfig:
    """Reinforcement learning configuration."""
    algorithm: str = "iql"  # iql, dt, cql
    gamma: float = 0.99
    tau: float = 0.005
    expectile: float = 0.7  # IQL specific
    beta: float = 3.0  # IQL temperature
    context_length: int = 20  # Decision Transformer
    return_scale: float = 1000.0


@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    experiment_name: str = "nanoarb_v1"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            data=DataConfig(**data.get("data", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            rl=RLConfig(**data.get("rl", {})),
            experiment_name=data.get("experiment_name", "nanoarb_v1"),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# Default configuration
DEFAULT_CONFIG = Config()

