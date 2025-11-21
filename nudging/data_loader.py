"""
Data Loader for text content organised by:
category/owner/name structure

Loads and preprocesses text files from a structured directory hierarchy,
applying category-specific cleaning (e.g. removing timestamps from podcasts).

"""

from pathlib import Path
import re
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable

logger = logging.getLogger(__name__)

__all__ = ['load_data', 'preprocess_text']

# --- Regex patterns for podcast transcript cleaning ---
_TS_INLINE = re.compile(r'\b(?:\d{1,2}:)?\d{1,2}:\d{2}\b')   # hh:mm:ss or m:ss or mm:ss
_TS_LINE   = re.compile(r'^\s*(?:\d{1,2}:)?\d{1,2}:\d{2}\s*$') # timestamp-only lines
_SPEAKER   = re.compile(r'^\s*[A-Z][A-Z\s.\'-]{2,}:\s+')       # ALLCAPS NAME:
_BRACKETED = re.compile(r'\s*\[(?:MUSIC|APPLAUSE|LAUGHTER|SFX)[^\]]*\]\s*', re.I)

    
def preprocess_text(category: str, text: str) -> str:
    """
    Apply category-aware text preprocessing.

    for podcast:
        - remove timestamps (e.g. "0:05", "01:12:23")
        - strip inline timestamps within sentences
        - remove speaker tags (e.g. "ABDI TIMER: ")
        - remove bracketed stage cues (e.g. "[INTRO MUSIC PLAYING]")

    for all categories:
        - normalise line breaks
        - collapse excessive whitespaces

    args:
        category: content category ('podcast', 'song')
        text: Raw text content to preprocess

    returns:
        Cleaned and normalised text
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if category.lower() == "podcasts":
        cleaned_lines = []
        for line in text.split("\n"):
            # Drop pure timestamp lines (e.g., "0:05", "01:12:33")
            if _TS_LINE.match(line):
                continue
            # Remove bracketed cues like [MUSIC PLAYING], [APPLAUSE]
            line = _BRACKETED.sub(" ", line)
            # Strip inline timestamps inside the sentence
            line = _TS_INLINE.sub(" ", line)
            # Remove ALLCAPS speaker labels at start of line: "ANDREW HUBERMAN: ..."
            line = _SPEAKER.sub("", line)
            # Collapse leftover whitespace
            line = re.sub(r'\s+', ' ', line).strip()
            if line:
                cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)
    
    # all other texts
    text = re.sub(r'\n{3,}', "\n\n", text).strip()
    return text

def _load_contents_by_structure(
        base_dir: str | Path = "data",
        exts: Tuple[str, ...] = (".txt",),
        min_words: int=30,
        max_samples: Optional[int] = None,
        custom_preprocessor: Optional[Callable[[str, str], str]] = None,
) -> dict:
    """
    Internal function to load texts from structured directory.

    Directory structure expected: base_dir/category/owner/name.ext

    Args:
        base_dir: Root dir containing the structured data
        exts: Tuple of file ext to process
        min_words: min word count threshold
        custom_preprocessor: optional function
        max_sample: % to truncate by characterss

    returns:
        contents_dict: dict returns a dict with the shape
        {{category}::{owner}::{name} : the_actual_text}
    """

    base = Path(base_dir)
    contents = {}

    logger.info(f"Scanning directory: {base}")
    logger.debug(f"File extension: {exts}, min_words: {min_words}")

    for p in base.rglob('*'):
        if not p.is_file():
            logger.info(f"Skipping non-file: {p}")
            continue
        if p.suffix.lower() not in exts:
            logger.debug(f"Skipping {p}: extension {p.suffix} not in {exts}")
            continue

        try:
            rel = p.relative_to(base)
        except ValueError:
            continue

        parts = rel.parts

        if len(parts) <3:
            continue

        category, owner = parts[0], parts[1]
        name = p.stem

        raw = p.read_text(encoding="utf-8", errors="ignore")

        # use custom preprocessor if provided, otherwise use default
        if custom_preprocessor is not None:
            text = custom_preprocessor(category, raw)
        else:
            text = preprocess_text(category, raw)

        words = len(text.split())
        kept = words >= min_words
        key = f"{category}::{owner}::{name}"

        if kept:
            if max_samples is not None and max_samples > 0 :
                content_testing_amount = int(len(text) * (max_samples / 100))
                contents[key] = text[:content_testing_amount]
                logger.info(f"TEST MODE - Kept {key}: {words} words, limit : {content_testing_amount}")
            else:
                contents[key] = text
                logger.info(f"Kept {key}: {words} words")
        else:
            logger.debug(f"filtered {key}: only {words} words (min: {min_words})")

    logger.info(f"Loaded {len(contents)} files")
    

    return contents

def load_data(
        base_dir: str | Path = "data",
        exts: Tuple[str, ...] = (".txt",),
        min_words: int = 30,
        max_samples: Optional[int] = None,
        custom_preprocessor: Optional[Callable[[str, str], str]] = None,
) -> dict:
    """
    Load and preprocess text data from structured directory hierarchy.

    Scans the directory structure (category/owner/name) and applies
    category-specific preprocessing to each file. Returns both the
    processed content and metadata inventory.

    Args:
        base_dir: Root directory containing the data (default: "data")
        exts: File extensions to process (default: (".txt",))
        min_words: Minimum word count to include a file (default: 30)
        custom_preprocessor: Optional custom preprocessing function that takes
                           (category, text) and returns processed text.
                           If None, uses default preprocess_text function.
        max_sample: % to truncate by characterss

    Returns:
        contents: Dict mapping 'category::owner::name' to preprocessed text
          
    Raises:
        FileNotFoundError: If base_dir does not exist
        NotADirectoryError: If base_dir is a file, not a directory

    Example:
        >>> # Basic usage
        >>> result = load_data()
        >>> print(result.summary)
        >>> print(f"Loaded {len(result.contents)} files")
        >>> print(result.inventory.head())
        >>> text = result.contents['podcasts::huberman::episode_1']
        >>>
        >>> # With custom preprocessing
        >>> def my_preprocessor(category, text):
        ...     return text.lower().strip()
        >>> result = load_data(custom_preprocessor=my_preprocessor)
    """

    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {base_dir}")
    if not base.is_dir():
        raise NotADirectoryError(f"Expected a directory, got a file: {base_dir}")

    logger.info(f"Starting data load from: {base_dir}")

    contents = _load_contents_by_structure(
        base_dir=base_dir, 
        exts=exts, 
        min_words=min_words, 
        custom_preprocessor=custom_preprocessor, 
        max_samples=max_samples
    )
    logger.info(f"Load complete.")

    return contents