import logging
import joblib
from pathlib import Path
import sys

logging.basicConfig(
    format="%(asctime)s.%(msecs)d %(levelname)s %(filename)s:%(lineno)d %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logging.info(f"Project root added to execution runtime path: {PROJECT_ROOT}")


from src.core.dataset_parser import parse_claudette_zipfile
from src.core.feature_engine import SimpleEmbeddingEngine, LegalBertEmbeddingEngine
from src.core.classifier import ClauseClassifier, MODELS_TO_EXPERIMENT, TUNING_GRIDS


ARTIFACT_EXPERIMENTS_DICT = {
    "simple_embeddings":SimpleEmbeddingEngine(),
    "legal_bert_embeddings":LegalBertEmbeddingEngine(),
}

def main():
    logging.info("Instantiate a fresh engine instance")

    for artifacts_experiment, artifacts_engine in ARTIFACT_EXPERIMENTS_DICT.items():

        search_engine = artifacts_engine

        artifacts_dir = PROJECT_ROOT / "models" / artifacts_experiment
        logging.info(f"Loading precomputed embeddings from {artifacts_dir}...")
        search_engine.load_artifacts(artifacts_dir)

        logging.info(f"Getting predictor variables -> embeddings...")
        X = search_engine.corpus_embeddings
        logging.info(f"Getting target variable 'is_unfair'...")
        y = search_engine.metadata_df['is_unfair'].values

        # Establish explicit directory mapping for optimized classifiers
        models_output_dir = artifacts_dir / "classifiers"
        models_output_dir.mkdir(parents=True, exist_ok=True)


        leaderboard = {}

        for model_name, estimator in MODELS_TO_EXPERIMENT.items():
            logging.info(f"Starting automated hyperparameter tuning for: {model_name}")
                        
            param_space = TUNING_GRIDS[model_name]
            
            clause_clf = ClauseClassifier(estimator=estimator)
            
            try:
                metrics_payload = clause_clf.tune_and_evaluate(
                    X, y, 
                    param_distributions=param_space, 
                    test_size=0.2, 
                    n_iter=15,  
                    cv=5,       
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

                logging.info(f"Optimized {model_name} Best-Fit Test Macro F1: {best_fit_metrics['macro_f1']:.4f}")
                leaderboard[model_name] = best_fit_metrics["macro_f1"]

                model_save_path = models_output_dir / model_name
                model_save_path.mkdir(parents=True, exist_ok=True)
                
                logging.info(f"Freezing best-fit weights for {model_name}...")
                clause_clf.save_model(model_save_path)
                
                report_file_path = model_save_path / "evaluation_report.joblib"
                logging.info(f"Saving clean best-fit metrics report at: {report_file_path}")
                joblib.dump(best_fit_metrics, report_file_path)

            except Exception as e:
                logging.error(f"Tuning or execution pipeline crashed on {model_name}: {str(e)}")


        logging.info("All 5 model families have been tuned. Their unique best-fit states are frozen to disk!")


if __name__ == "__main__":
    main()