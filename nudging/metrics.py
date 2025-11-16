from rapidfuzz import fuzz
import numpy as np

def exact_match_score(generated:str,target:str) -> float:
    """Simple exact character match"""
    generated_cleaned = generated.lower().strip()
    target_cleaned = target.lower().strip()
    if not target_cleaned:
        return 0.0
    matches = sum(1 for a, b in zip(generated_cleaned, target_cleaned) if a==b)
    return matches/len(target_cleaned)

def fuzzy_match_score(generated:str, target:str) -> float:
    """compare 2 texts using fuzzy matching (Levenshtein distance).
    Returns between 0.0 -> 1.0 (the higher, the more similar)"""
    return fuzz.ratio(generated.lower(), target.lower()) / 100.0

def token_overlap_score(generated:str, target:str) -> float:
    generated_tokens = set(generated.lower().split())
    target_tokens = set(target.lower().split())
    if not target_tokens:
        return 0.0
    intersection = len(generated_tokens & target_tokens)
    union = len(generated_tokens | target_tokens)
    return intersection / union if union > 0 else 0.0

def semantic_similarity_score(generated:str, target:str, model) -> float:
    "cosine similarity of embeddings"
    if not generated.strip() or not target.strip():
        return 0.0
    embeddings = model.encode([generated, target])
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(similarity)
