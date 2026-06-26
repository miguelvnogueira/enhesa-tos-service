import logging
from typing import Union, List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logging.info(f"Project root added to execution runtime path: {PROJECT_ROOT}")

from src.core.feature_engine import SimpleEmbeddingEngine, LegalBertEmbeddingEngine
from src.core.classifier import ClauseClassifier

def process_user_sentence(
    search_engine: Union[SimpleEmbeddingEngine, LegalBertEmbeddingEngine], 
    clause_clf: ClauseClassifier, 
    input_sentence: str, 
    top_k: int=3
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Orchestrates the complete NLP analysis pipeline for a target ToS sentence.
    
    1. Executes a spatial cosine proximity search against the cached corpus index.
    2. Encodes the raw text query via the Legal-BERT transformer layers.
    3. Infers whether the target sentence is legally compliant or an unfair clause.

    Args:
        search_engine (LegalBertEmbeddingEngine): Initialized vector search engine 
            containing precomputed dense legal vector spaces and data metadata frames.
        clause_clf (ClauseClassifier): Pre-trained classification model 
            (e.g., SVC RBF Pipeline) mapping vectors to binary fairness indicators.
        input_sentence (str): The raw text sequence extracted from the user input.
        top_k (int, optional): The total number of similar historical matches to return. 
            Defaults to 3.

    Returns:
        Tuple[List[Dict[str, Any]], np.ndarray]: A composite tuple containing:
            - search_results: A ranked list of matching metadata dictionaries, 
              including historical ground truths and calculated cosine similarity scores.
            - prediction: A numpy array containing the model's binary compliance verdict 
              (e.g., 1 for Unfair, 0 for Fair).
    """
    logging.info(f"Processing input clause: '{input_sentence}'")

    logging.info(f"Searching for top {top_k} similarity matches...")
    search_results = search_engine.search(input_sentence, top_k=top_k)

    logging.info(f"Classifying sentence as fair or unfair...")
    query_vector = search_engine.transform(input_sentence)

    prediction = clause_clf.predict(query_vector)


    return search_results, prediction

