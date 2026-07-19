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
    endpoint: str = "http://localhost:11434"

@dataclass
class ExperimentConfig:
    name: str = "memorisation_study"
    models: list[ModelConfig] = field(default_factory=list)
    data_config: DataConfig = field(default_factory=DataConfig)
    context_percentages: list[int] = field(
        default_factory=lambda: [0, 25, 50, 75, 90]
    )
    temperatures: list[float] = field(
        default_factory=lambda: [0.0, 0.7]
    )
    random_seed: int = 42
    prompt_version: str = "v4"
    token_multiplier: float = 1.5
    include_semantic: bool = False
    selected_text_ids: list[str] = field(default_factory=list)
    output_filename: str = "pilot_600_v4.csv"
    context_delay_seconds: float = 0.0

# Configuration track 1: lightweight, one-model notebook experiments.
def experimental(
        model: str = "qwen3:0.6b",
        categories: Optional[List[str]] = None,
        context_pcts: Optional[List[int]] = None,
        context_delay_seconds: float = 0.0,
        temperature: float = 0.7,
        random_seed: int = 42,
        prompt_version: str = "v4",
        token_multiplier: float = 1.5,
        include_semantic: bool = False,
) -> ExperimentConfig:
    """Create a one-model configuration for exploratory notebook work."""
    return ExperimentConfig(
        name="memorisation_experimental",
        models=[ModelConfig(name=model)],
        data_config=DataConfig(categories=categories or ["songs"]),
        context_percentages=context_pcts or [5,25,50,75,90],
        temperatures=[temperature],
        random_seed=random_seed,
        prompt_version=prompt_version,
        token_multiplier=token_multiplier,
        include_semantic=include_semantic,
        context_delay_seconds=context_delay_seconds,
    )

# Configuration track 2: the frozen, resumable batch pilot.
PILOT_SMOKE = ExperimentConfig(
    name="pilot_smoke_v4",
    models=[ModelConfig(name="qwen2.5:0.5b-instruct")],
    data_config=DataConfig(categories=["songs"]),
    context_percentages=[0, 25, 50, 75, 90],
    temperatures=[0.0, 0.7],
    random_seed=42,
    prompt_version="v4",
    token_multiplier=1.5,
    include_semantic=False,
    selected_text_ids=["songs::taylor_swift::the_fate_of_ophelia"],
    output_filename="pilot_smoke_v4.csv",
    context_delay_seconds=5.0,
)


PILOT_600 = ExperimentConfig(
    name="pilot_600_v4",
    models=[
        ModelConfig(name="qwen2.5:0.5b-instruct"),
        ModelConfig(name="llama3.2:1b-instruct-q4_K_M"),
    ],
    data_config=DataConfig(categories=["songs"]),
    context_percentages=[0, 25, 50, 75, 90],
    temperatures=[0.0, 0.7],
    random_seed=42,
    prompt_version="v4",
    token_multiplier=1.5,
    include_semantic=False,
    selected_text_ids=[
        # Add your exact 30 dataset IDs here.
        "songs::taylor_swift::the_fate_of_ophelia"
    ],
    output_filename="pilot_600_v4.csv",
    context_delay_seconds=5.0,
)


# A bounded two-song run using the same model, prompt, and decoding settings as
# the eventual full pilot: 2 texts × 5 contexts × 2 models × 2 temperatures.
PILOT_SONGS_40 = ExperimentConfig(
    name="pilot_songs_40_v4",
    models=[
        ModelConfig(name="qwen2.5:0.5b-instruct"),
        ModelConfig(name="llama3.2:1b-instruct-q4_K_M"),
    ],
    data_config=DataConfig(categories=["songs"]),
    context_percentages=[0, 25, 50, 75, 90],
    temperatures=[0.0, 0.7],
    random_seed=42,
    prompt_version="v4",
    token_multiplier=1.5,
    include_semantic=False,
    selected_text_ids=[
        "songs::taylor_swift::the_fate_of_ophelia",
        "songs::taylor_swift::shake_it_off",
    ],
    output_filename="pilot_songs_40_v4.csv",
    context_delay_seconds=5.0,
)


# Terminal-facing names. Add future named experiment configurations here.
EXPERIMENT_CONFIGS = {
    "smoke": PILOT_SMOKE,
    "songs-40": PILOT_SONGS_40,
    "pilot-600": PILOT_600,
}
