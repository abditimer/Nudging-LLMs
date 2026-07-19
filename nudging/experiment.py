from collections import defaultdict
from typing import Dict
from math import ceil
from nudging.models import OllamaClient
from nudging.metrics import exact_match_score, fuzzy_match_score, token_overlap_score, semantic_similarity_score
from nudging.prompt import build_continuation_prompt

import logging
logger = logging.getLogger(__name__)

def _get_split_text(text: str, percentage: float) -> dict:
    """Split text into test portion and remaining portion"""
    logger.info("splitting text.")
    d = defaultdict(str)
    words = text.split()
    chunk_size = int(len(words) * (percentage / 100))
    d['test_words'] = " ".join(words[:chunk_size])
    d['remaining_words'] = " ".join(words[chunk_size:])
    return d

def _trim_to_n_words(text: str, max_words: int) -> str:
    """Trim generated text to the same word span as the withheld target."""
    words = text.strip().split()
    return " ".join(words[:max_words])


def _get_num_predict_for_target(target_word_count: int, token_multiplier: float = 1.5) -> int:
    """Convert the target word count into an approximate Ollama token budget."""
    if target_word_count <= 0:
        return 1
    return ceil(target_word_count * token_multiplier)

def _generate_response(
    *,
    content: str,
    percentage: float,
    model_client: OllamaClient,
    prompt_version: str,
    temperature: float,
    seed: int | None,
    token_multiplier: float,
):
    '''
    it connects to our model and sends it the text.
    
    :param content: precontext string for the model to generate from
    :param percentage: how much content the model is seeing
    :param model_client: the model we are connecting to
    '''
    logger.info("generating a response via model client.")
    split_text = _get_split_text(content, percentage)
    context = split_text["test_words"]
    target = split_text["remaining_words"]
    target_word_count = len(target.split())

    prompt = build_continuation_prompt(
        version=prompt_version,
        context_text=context,
        target_word_count=target_word_count,
    )

    num_predict = _get_num_predict_for_target(
        target_word_count=target_word_count,
        token_multiplier=token_multiplier,
    )

    raw_generated_response = model_client.generate(
        prompt=prompt,
        temperature=temperature,
        seed=seed,
        num_predict=num_predict,
    )

    generated_response = _trim_to_n_words(
        raw_generated_response,
        target_word_count,
    )
    raw_generated_words = len(raw_generated_response.split())
    generated_words = len(generated_response.split())
    metadata = {
        "prompt_version": prompt_version,
        "temperature": temperature,
        "seed": seed,
        "num_predict": num_predict,
        "raw_generated_words": raw_generated_words,
        "generated_words": generated_words,
        "raw_length_ratio": raw_generated_words / target_word_count if target_word_count else 0.0,
        "scored_length_ratio": generated_words / target_word_count if target_word_count else 0.0,
        "length_controlled": True,
        "trimmed_to_target_words": True,
    }
    return generated_response, context, target, metadata

def run_single_experiment(
    *,
    content: str,
    percentage: float,
    model_client: OllamaClient,
    prompt_version: str,
    temperature: float,
    seed: int | None,
    token_multiplier: float,
) -> Dict:
    """Run one experiment, return metrics dict"""

    generated_response, context, target, generation_metadata = _generate_response(
        content=content,
        percentage=percentage,
        model_client=model_client,
        prompt_version=prompt_version,
        temperature=temperature,
        seed=seed,
        token_multiplier=token_multiplier,
    )

    # Calculate exact_match metric only (for now)
    exact_match = exact_match_score(generated=generated_response, target=target)

    # Return dict with: percentage, context_words, target_words, 
    #                   generated_words, exact_match
    return {
        "percentage": percentage,
        "context_words": len(context.split()),
        "target_words": len(target.split()),
        "exact_match": exact_match,
        **generation_metadata,
    }

def run_experiments(
    *,
    title: str,
    content: str,
    percentage: float,
    model_client: OllamaClient,
    prompt_version: str,
    temperature: float,
    seed: int | None,
    token_multiplier: float,
    include_semantic: bool = False,
) -> Dict:
    """
    we first generate the response and then calculate all the metrics.
    
    :param content: this is the text we are experiment on
    :type content: str
    :param percentage: how much of the text we want to analyse in this run
    :type percentage: float
    :param model_client: model client e.g. ollama model
    :type model_client: OllamaClient
    :param verbose: Description
    :type verbose: bool
    :return: data and all experimental results
    :rtype: Dict
    """
    logger.info("running all experiments")
    generated_response, context, target, generation_metadata = _generate_response(
        content=content,
        percentage=percentage,
        model_client=model_client,
        prompt_version=prompt_version,
        temperature=temperature,
        seed=seed,
        token_multiplier=token_multiplier,
    )

    # Calculate metrics
    metrics = {
        "content": title,
        "percentage": percentage,
        "context_words": len(context.split()),
        "target_words": len(target.split()),
        "exact_match": exact_match_score(generated_response, target),
        "fuzzy_match": fuzzy_match_score(generated_response, target),
        "token_overlap": token_overlap_score(generated_response, target),
        **generation_metadata,
    }
    metrics["semantic_similarity"] = (
        semantic_similarity_score(generated_response, target)
        if include_semantic
        else None
    )
    return metrics
