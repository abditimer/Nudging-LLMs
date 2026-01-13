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
    max_tokens: Optional[int] = None

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
    context_delay_seconds: float = 0.0

    def __post_init__(self):
        if self.context_percentages is None:
            self.context_percentages = [20,60]
    
    def start_logging(self):
        logger.info(f"Running experiment: {self.name}")
        logger.info(f"Contexted to run: {self.context_percentages}")


# builder function that lets for quicker experimentation
def baseline(
        model:str = "qwen3:0.6b",
        categories: Optional[List[str]] = None,
        context_pcts: Optional[List[int]] = None,
        max_samples: Optional[int] = None,
        context_delay_seconds: float = 0.0,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
) -> ExperimentConfig:
    """Create a baseline memorisation exp with custom params"""
    return ExperimentConfig(
        name="memorisation_baseline",
        context_percentages=context_pcts or [40],
        max_samples=max_samples,
        context_delay_seconds=context_delay_seconds,
        model_config=ModelConfig(
            name=model,
            temperature=temperature,
            max_tokens=max_tokens
        ),
        data_config=DataConfig(
            categories=categories
        ),
        **kwargs
    )

def extended(
        model:str = "qwen3:0.6b",
        categories: Optional[List[str]] = None,
        context_pcts: Optional[List[int]] = None,
        max_samples: Optional[int] = None,
        context_delay_seconds: float = 0.0,
        temperature: float = 0.7,
        max_tokens: optional[int] = None,
        **kwargs
) -> ExperimentConfig:
    """Create a extnded memorisation exp with custom params"""
    return ExperimentConfig(
        name="memorisation_extended",
        context_percentages=context_pcts or [5,25,50,75,90],
        max_samples=max_samples,
        context_delay_seconds=context_delay_seconds,
        model_config=ModelConfig(
            name=model,
            temperature=temperature,
            max_tokens=max_tokens
        ),
        data_config=DataConfig(
            categories=categories or ["songs"]
        ),
        **kwargs
    )

EXPERIMENT_BASELINE_ONLY_SONGS_QWEN = baseline(
    model="qwen2.5:0.5b-instruct",
    categories=["songs"],
    context_pcts=[0, 25, 60, 90],
    context_delay_seconds=5.0
)

# ---------------------------
# Removing the below, but keeping for reproducability
# ---------------------------


# Pre-defined configs for common experiments
EXPERIMENT_BASELINE_SAMPLED = ExperimentConfig(
    name="memorisation_baseline",
    context_percentages=[40],
    max_samples=3,
    model_config=ModelConfig(),
    data_config=DataConfig()
)

EXPERIMENT_BASELINE_ONLY_SONGS_QWEN_old = ExperimentConfig(
    name="memorisation_baseline",
    context_percentages=[0, 25, 60, 90],
    model_config=ModelConfig(name="qwen2.5:0.5b-instruct"),
    data_config=DataConfig(
        categories=["songs"]
    ),
    context_delay_seconds=5.0,
)

EXPERIMENT_BASELINE_ONLY_SONGS_LLAMA = ExperimentConfig(
    name="memorisation_baseline",
    context_percentages=[0, 25, 60, 90],
    model_config=ModelConfig(name="llama3.2:1b-instruct-q4_K_M"),
    data_config=DataConfig(
        categories=["songs"]
    ),
    context_delay_seconds=5.0,
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
