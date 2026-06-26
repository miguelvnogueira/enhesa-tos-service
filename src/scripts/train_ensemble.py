import logging
import joblib
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


EXPERIMENT_NAME = "simple_embeddings"
#EXPERIMENT_NAME = "legal_bert_embeddings" 
ARTIFACTS_DIR = PROJECT_ROOT / "models" / EXPERIMENT_NAME
CLASSIFIERS_DIR = ARTIFACTS_DIR / "classifiers"

from src.core.feature_engine import LegalBertEmbeddingEngine
from src.core.classifier import MODELS_TO_EXPERIMENT

def load_base_estimators():
    """
    Loads and unwraps the pre-tuned ensemble base estimators from disk storage.

    1. Iterates through the configured model experiment registry keys.
    2. Dynamically resolves serialization paths for frozen joblib binary artifacts.
    3. Safely extracts the core underlying model instance from any wrapper abstractions.
    4. Compiles a named tuple-like list structure required by downstream ensemble meta-classifiers.

    Returns:
        list[tuple[str, object]]: A collection of named estimator pairs, where each tuple 
            contains the string identifier key and the unmarshalled concrete model instance.
            Returns an empty list if no frozen artifacts are successfully resolved.
    """
    estimators = []
    
    model_names = list(MODELS_TO_EXPERIMENT.keys())
    
    for name in model_names:
        model_path = CLASSIFIERS_DIR / name / "clf_estimator.joblib" # Adjust if your save_model uses a different filename
        if model_path.exists():
            logging.info(f"Loading tuned base estimator: {name}")
            clf_wrapper = joblib.load(model_path)
            
            actual_estimator = getattr(clf_wrapper, "model", clf_wrapper)
            
            estimators.append((name, actual_estimator))
        else:
            logging.warning(f"Could not find frozen weights for {name} at {model_path}")
            
    return estimators

def main():
    logging.info("Loading feature embeddings and metadata labels...")
    engine = LegalBertEmbeddingEngine()
    engine.load_artifacts(ARTIFACTS_DIR)
    
    X = engine.corpus_embeddings
    y = engine.metadata_df['is_unfair'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    base_estimators = load_base_estimators()
    if not base_estimators:
        raise ValueError("No base estimators were successfully loaded. Check your paths.")
        
    logging.info(f"Successfully assembled {len(base_estimators)} base estimators into stacking pipeline.")
    
    meta_learner = LogisticRegression(
        max_iter=5000,         
        solver="lbfgs",        
        class_weight="balanced", 
        random_state=42
    )    
    
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,            
        n_jobs=-1,       
        passthrough=False
    )    
    logging.info("Training Meta-Ensemble Stacking Classifier (generating out-of-fold predictions)...")
    stacking_clf.fit(X_train, y_train)
    
    logging.info("Evaluating Stacking Classifier against validation holdout...")
    y_pred = stacking_clf.predict(X_test)
    metrics_payload = classification_report(y_test, y_pred, output_dict=True)
    
    ensemble_metrics = {
        "Model": "MetaEnsemble_Stacking",
        "macro_f1": metrics_payload["macro avg"]["f1-score"],
        "macro_precision": metrics_payload["macro avg"]["precision"],
        "macro_recall": metrics_payload["macro avg"]["recall"],
        "accuracy": metrics_payload["accuracy"]
    }
    
    
    ensemble_output_dir = CLASSIFIERS_DIR / "MetaEnsemble_Stacking"
    ensemble_output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(stacking_clf, ensemble_output_dir / "model.joblib")
    joblib.dump(ensemble_metrics, ensemble_output_dir / "evaluation_report.joblib")
    logging.info("Meta-Ensemble serialized to disk.")

if __name__ == "__main__":
    main()