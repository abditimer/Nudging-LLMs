from collections import defaultdict
from typing import Dict
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

def _generate_response(content, percentage, model_client):
    logger.info("generating a response via model client.")
    #TODO: add input validation

    # Split text
    split_text = _get_split_text(content, percentage)
    context = split_text['test_words']
    target = split_text['remaining_words']
    # Create prompt
    prompt = f"""Continue this text:<StartText>
{context}
</StartText>
Continue:"""
    # Generate with model
    generated_response = model_client.generate(prompt=prompt)
    return generated_response, context, target

def run_single_experiment(
    content: str,
    percentage: float, 
    model_client: OllamaClient,
    verbose: bool = False
) -> Dict:
    """Run one experiment, return metrics dict"""

    generated_response, context, target = _generate_response(
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
    }

def run_experiments(
    title: str,
    content: str,
    percentage: float, 
    model_client: OllamaClient,
    verbose: bool = False
) -> Dict:
    """
    Docstring for run_experiments
    
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
    generated_response, context, target = _generate_response(
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
    }
    return metrics