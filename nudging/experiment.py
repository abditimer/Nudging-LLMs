def run_experiment(content:str, percentage:float, model_name:str="qwen3:0.6b"):
    """Run one experiment at a given percentage"""
    split_text = get_split_text(content, percentage)
    context = split_text['test_words']
    target = split_text['remaining_words']

    prompt = f"""Continue this text:

{context}
        
Continue:"""
    
    print(f"\n{'='*60}")
    print(f"Testing at {percentage}% context")
    print(f"Context: {len(context.split())} words")
    print(f"Target: {len(target.split())} words")

    url="http://localhost:11434/api/generate"
    data={
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }

    response = requests.post(url, json=data)
    generated = response.json()["response"]

    print(f"\nGenerated ({len(generated.split())} words):")
    print(generated[:200], "...")

    # Calculate metrics
    metrics = {
        "percentage": percentage,
        "context_words": len(context.split()),
        "target_words": len(target.split()),
        "generated_words": len(generated.split()),
        "exact_match": exact_match_score(generated, target),
        "fuzzy_match": fuzzy_match_score(generated, target),
        "token_overlap": token_overlap_score(generated, target),
        "semantic_similarity": semantic_similarity_score(generated, target, semantic_model),
    }
    
    print(f"\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, float) and k != "percentage":
            print(f"  {k}: {v:.3f}")
    
    return metrics

# Cell 6: Run Multiple Experiments
percentages = [0, 5, 10, 20, 30, 50, 75, 90]
results = []

for pct in tqdm(percentages, desc="Running experiments"):
    result = run_experiment(SAMPLE_CONTENT, pct,'qwen3:0.6b')
    results.append(result)

# Convert to DataFrame
df_results = pd.DataFrame(results)
df_results