import logging
import joblib
from pathlib import Path
import sys

# Setup structural logging output formats
logging.basicConfig(
    format="%(asctime)s.%(msecs)d %(levelname)s %(filename)s:%(lineno)d %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Resolve project pathing topologies (moves two steps out from src/scripts/ to find root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logging.info(f"Project root added to execution runtime path: {PROJECT_ROOT}")


from src.core.dataset_parser import parse_claudette_zipfile
from src.core.feature_engine import SimpleEmbeddingEngine
from src.core.classifier import ClauseClassifier, MODELS_TO_EXPERIMENT, TUNING_GRIDS


def main():
    logging.info("Instantiate a fresh engine instance")
    search_engine = SimpleEmbeddingEngine()

    artifacts_dir = PROJECT_ROOT / "models" / "search_artifacts"
    logging.info(f"Loading precomputed embeddings from {artifacts_dir}...")
    search_engine.load_artifacts(artifacts_dir)

    logging.info(f"Getting predictor variables -> embeddings...")
    X = search_engine.corpus_embeddings
    logging.info(f"Getting target variable 'is_unfair'...")
    y = search_engine.metadata_df['is_unfair'].values

    # Establish explicit directory mapping for optimized classifiers
    models_output_dir = PROJECT_ROOT / "models" / "classifiers"
    models_output_dir.mkdir(parents=True, exist_ok=True)

    # Naming alignment map to connect MODELS_TO_EXPERIMENT keys with TUNING_GRIDS keys
    grid_name_alignment = {
        "LogisticRegression": "LogisticRegression",
        "LinearSVC": "LinearSVC",
        "RandomForest": "RandomForestClassifier",
        "HistGradientBoosting": "HistGradientBoostingClassifier",
        "MLPClassifier": "MLPClassifier"
    }

    # Tracking metrics dictionary for a final summary report
    leaderboard = {}

    for model_name, estimator in MODELS_TO_EXPERIMENT.items():
        logging.info(f"🚀 Starting automated hyperparameter tuning for: {model_name}")
                    
        param_space = TUNING_GRIDS[model_name]
        
        # Instantiate your custom wrapper with the base estimator configuration
        clause_clf = ClauseClassifier(estimator=estimator)
        
        try:
            # 1. Automated tuning updates clause_clf.model to the best internal parameter structure
            metrics_payload = clause_clf.tune_and_evaluate(
                X, y, 
                param_distributions=param_space, 
                test_size=0.2, 
                n_iter=15,  # Explores 15 random parameter combinations per model family
                cv=5,       # 5-Fold Stratified Cross-Validation
                random_state=42
            )
            

            best_fit_metrics = {
                "model_family": model_name,
                "best_params": clause_clf.model.get_params(), # Captures the winning hyperparameters
                "macro_f1": metrics_payload["macro avg"]["f1-score"],
                "macro_precision": metrics_payload["macro avg"]["precision"],
                "macro_recall": metrics_payload["macro avg"]["recall"],
                "accuracy": metrics_payload["accuracy"]
            }

            # Log the performance
            logging.info(f"✅ Optimized {model_name} Best-Fit Test Macro F1: {best_fit_metrics['macro_f1']:.4f}")
            leaderboard[model_name] = best_fit_metrics["macro_f1"]

            # Unique subdirectory
            model_save_path = models_output_dir / model_name
            model_save_path.mkdir(parents=True, exist_ok=True)
            
            # Save the best-fit model weights
            logging.info(f"💾 Freezing best-fit weights for {model_name}...")
            clause_clf.save_model(model_save_path)
            
            # Save ONLY the streamlined best-fit metrics dictionary
            report_file_path = model_save_path / "evaluation_report.joblib"
            logging.info(f"📊 Saving clean best-fit metrics report at: {report_file_path}")
            joblib.dump(best_fit_metrics, report_file_path)

        except Exception as e:
            logging.error(f"❌ Tuning or execution pipeline crashed on {model_name}: {str(e)}")


    logging.info("🎉 All 5 model families have been tuned. Their unique best-fit states are frozen to disk!")


if __name__ == "__main__":
    main()