from dataclasses import dataclass, field
from typing import List, Optional

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
class PromptConfig:
    """
    Track prompts
    """
    version: str
    prompt: str

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
    
    def start_logging(self):
        logger.info(f"Running experiment: {self.name}")
        logger.info(f"Contexted to run: {self.context_percentages}")


# Pre-defined configs for common experiments
EXPERIMENT_BASELINE_SAMPLED = ExperimentConfig(
    name="memorisation_baseline",
    context_percentages=[40],
    max_samples=3,
    model_config=ModelConfig(),
    data_config=DataConfig()
)

EXPERIMENT_BASELINE_ONLY_SONGS = ExperimentConfig(
    name="memorisation_baseline",
    context_percentages=[0, 25, 60, 90],
    model_config=ModelConfig(),
    data_config=DataConfig(
        categories=["songs"]
    )
)

EXPERIMENT_BASELINE = ExperimentConfig(
    name="memorisation_baseline",
    context_percentages=[40],
    model_config=ModelConfig(),
    data_config=DataConfig()
)

EXPERIMENT_BASELINE_MULTIPLE = ExperimentConfig(
    name="memorisation_baseline",
    context_percentages=[20, 40, 60],
    #max_samples=3,
    model_config=ModelConfig(),
    data_config=DataConfig()
)

EXPERIMENT_EXTENDED = ExperimentConfig(
    name="memorisation_extended", 
    context_percentages=[5, 25, 50, 75, 90],
    model_config=ModelConfig(),
    data_config=DataConfig(
        categories=["songs"]
    )
)
