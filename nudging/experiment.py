from collections import defaultdict
from typing import Dict
from nudging.models import OllamaClient
from nudging.metrics import exact_match_score


def _get_split_text(text: str, percentage: float) -> dict:
    """Split text into test portion and remaining portion"""
    d = defaultdict(str)
    words = text.split()
    chunk_size = int(len(words) * (percentage / 100))
    d['test_words'] = " ".join(words[:chunk_size])
    d['remaining_words'] = " ".join(words[chunk_size:])
    return d

def run_single_experiment(
    content: str,
    percentage: float, 
    model_client: OllamaClient,
    verbose: bool = False
) -> Dict:
    """Run one experiment, return metrics dict"""
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