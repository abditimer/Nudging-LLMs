from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataConfig:
    data_folder_name: str = None
    min_word_count: int = 30
    categories: List[str] = None
    batch_size: int = 32

    def __post_init__(self):
        if self.categories is None:
            self.categories = ["songs"]
        if self.data_folder_name is None:
            self.data_folder_name = "data"
        
@dataclass
class ModelConfig:
    name: str = "qwen3:0.6b"
    temperature: float = 0.7
    endpoint: str = "http://localhost:11434"

@dataclass
class ExperimentConfig:
    name: str = "memorisation_study"
    random_seed: int = 42
    model_config: ModelConfig = field(default_factory=ModelConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    context_percentages: List[int] = None
    max_samples: Optional[int] = None

    def __post_init__(self):
        if self.context_percentages is None:
            self.context_percentages = [20,60]

# Pre-defined configs for common experiments
EXPERIMENT_BASELINE = ExperimentConfig(
    name="memorisation_baseline",
    context_percentages=[40],
    max_samples=3,
    model_config=ModelConfig(),
    data_config=DataConfig()
)

EXPERIMENT_BASELINE_MULTIPLE = ExperimentConfig(
    name="memorisation_baseline",
    context_percentages=[20, 40, 60],
    max_samples=3,
    model_config=ModelConfig(),
    data_config=DataConfig()
)

EXPERIMENT_EXTENDED = ExperimentConfig(
    name="memorisation_extended", 
    context_percentages=[0, 5, 10, 20, 30, 50, 75, 90, 98],
    model_config=ModelConfig(),
    data_config=DataConfig()
)
