from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DataConfig:
    min_word_count: int = 30
    categories: List[str] = None
    batch_size: int = 32

    def __post_init__(self):
        if self.categories is None:
            self.categories = ["songs"]
        
@dataclass
class ModelConfig:
    name: str = "qwen3:0.6b"
    temperature: float = 0.7
    endpoint: str = "http://localhost:11434"

@dataclass
class ExperimentConfig:
    name: str = "memorisation_study"
    context_percentages: List[int] = None
    random_seed: int = 42

    def __post_init__(self):
        if self.context_percentages is None:
            self.context_percentages = [20,60]

# Pre-defined configs for common experiments
MEMORISATION_BASELINE = ExperimentConfig(
    name="memorisation_baseline",
    context_percentages=[20, 40, 60]
)

MEMORISATION_EXTENDED = ExperimentConfig(
    name="memorisation_extended", 
    context_percentages=[0, 5, 10, 20, 30, 50, 75, 90, 98]
)