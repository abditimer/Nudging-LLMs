from collections import defaultdict
from typing import Dict
from math import ceil
from nudging.models import OllamaClient
from nudging.metrics import exact_match_score, fuzzy_match_score, token_overlap_score, semantic_similarity_score

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
    content,
    percentage,
    model_client,
    control_length: bool = True,
    trim_to_target: bool = True,
):
    '''
    it connects to our model and sends it the text.
    
    :param content: precontext string for the model to generate from
    :param percentage: how much content the model is seeing
    :param model_client: the model we are connecting to
    '''
    logger.info("generating a response via model client.")
    #TODO: add input validation

    # Split text
    split_text = _get_split_text(content, percentage)
    context = split_text['test_words']
    target = split_text['remaining_words']
    target_word_count = len(target.split())
    # Create prompt
    prompt = f"""Continue this text:<StartText>
{context}
</StartText>
Continue:"""
    # Generate with model
    generation_options = {}
    if control_length:
        generation_options["num_predict"] = _get_num_predict_for_target(
            target_word_count=target_word_count,
            token_multiplier=model_client.words_to_token_multiplier,
        )

    raw_generated_response = model_client.generate(prompt=prompt, **generation_options)
    generated_response = raw_generated_response
    if trim_to_target:
        generated_response = _trim_to_n_words(
            text=raw_generated_response,
            max_words=target_word_count,
        )

    metadata = {
        "raw_generated_words": len(raw_generated_response.split()),
        "trimmed_to_target_words": trim_to_target,
        "length_controlled": control_length,
        "num_predict": generation_options.get("num_predict"),
    }
    return generated_response, context, target, metadata

def run_single_experiment(
    content: str,
    percentage: float, 
    model_client: OllamaClient,
    verbose: bool = False
) -> Dict:
    """Run one experiment, return metrics dict"""

    generated_response, context, target, generation_metadata = _generate_response(
        content, percentage, model_client
    )

    # Calculate exact_match metric only (for now)
    exact_match = exact_match_score(generated=generated_response, target=target)

    # Return dict with: percentage, context_words, target_words, 
    #                   generated_words, exact_match
    return {
        "percentage": percentage,
        "context_words": len(context.split()),
        "target_words": len(target.split()),
        "generated_words": len(generated_response.split()),
        "exact_match": exact_match,
        **generation_metadata,
    }

def run_experiments(
    title: str,
    content: str,
    percentage: float, 
    model_client: OllamaClient,
    verbose: bool = False
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
        content, percentage, model_client
    )

    # Calculate metrics
    metrics = {
        "content": title,
        "percentage": percentage,
        "context_words": len(context.split()),
        "target_words": len(target.split()),
        "generated_words": len(generated_response.split()),
        "exact_match": exact_match_score(generated_response, target),
        "fuzzy_match": fuzzy_match_score(generated_response, target),
        "token_overlap": token_overlap_score(generated_response, target),
        "semantic_similarity": semantic_similarity_score(generated_response, target),
        **generation_metadata,
    }
    return metrics
