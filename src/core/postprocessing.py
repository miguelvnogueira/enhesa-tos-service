import numpy as np
from typing import Dict, List, Any

def compute_bayesian_posterior(
    prior_prob: float, 
    search_results: List[Dict[str,Any]], 
    min_sim_score: float = 0.5,
    baseline_unfair_rate: float = 0.11,
) -> float:
    """
    Updates the model's prior probability using an empirical Bayesian likelihood 
    derived from the semantic search neighborhood.
    
    :param prior_prob: float, The SVC model's P(Unfair) output [0, 1]
    :param search_results: list of dicts, Results from search_engine.search()
    :param baseline_unfair_rate: float, The global corpus base-rate (11% in CLAUDETTE)
    :return: float, The updated context-aware Posterior Probability [0, 1]
    """
    prior_prob = np.clip(prior_prob, 0.01, 0.99)
    
    # 1. Convert Prior Probability to Prior Odds
    prior_odds = prior_prob / (1.0 - prior_prob)
    
    # 2. Compute the Likelihood Ratio (Lambda) from Semantic Neighbors
    # We weight each neighbor's vote by its continuous cosine similarity score
    total_weight = 0.0
    unfair_weight = 0.0
    
    for result in search_results:
        sim_score = result.get('similarity_score', 0.0)
        # Enforce a minimum floor for similarity to prevent irrelevant matches from counting
        if sim_score < min_sim_score: 
            continue
            
        weight = sim_score **2
        total_weight += weight
        
        if result.get('is_unfair') == 1:
            unfair_weight += weight

    # If no valid semantic neighbors were found, return the model prior unchanged
    if total_weight == 0:
        return prior_prob

    # Empirical proportion of unfairness in this local cluster
    local_unfair_ratio = unfair_weight / total_weight
    
    # Avoid zero/one extremes in the local ratio to keep the multiplier stable
    local_unfair_ratio = np.clip(local_unfair_ratio, 0.05, 0.95)
    
    # 3. Formulate the Likelihood Ratio: (Local Ratio / Global Base Rate)
    # If the local cluster has a higher density of unfair clauses than the 11% global average,
    # this ratio acts as a positive multiplier (> 1.0).
    likelihood_ratio = (local_unfair_ratio / baseline_unfair_rate) / ((1.0 - local_unfair_ratio) / (1.0 - baseline_unfair_rate))
    
    # 4. Compute Posterior Odds and convert back to Posterior Probability
    posterior_odds = prior_odds * likelihood_ratio
    posterior_prob = posterior_odds / (1.0 + posterior_odds)
    
    return posterior_prob