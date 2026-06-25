import logging
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_TO_EXPERIMENT = {
    "LogisticRegression": LogisticRegression(
        class_weight="balanced", 
        max_iter=5000, 
        random_state=42
    ),
    "LinearSVC": LinearSVC(
        class_weight="balanced", 
        dual="auto", 
        max_iter=5000, 
        random_state=42
    ),
    "SVC_RBF_Pipeline": Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf", 
            class_weight="balanced", 
            random_state=42
        ))
    ]),
    "RandomForest": RandomForestClassifier(
        class_weight="balanced", 
        n_estimators=200, 
        random_state=42, 
        n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        random_state=42
    ),
    "MLPClassifier": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True, 
        validation_fraction=0.1
    )
}

# Map of the search space structures per estimator family
TUNING_GRIDS = {
    "LogisticRegression": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],         
        "solver": ["saga", "lbfgs"],         
        "l1_ratio": [0.0, 1.0] 
    },
    "LinearSVC": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "loss": ["squared_hinge"],
        "tol": [1e-4, 1e-3, 1e-2]
    },
    "SVC_RBF_Pipeline": {
        # Prefixed with 'svc__' to target the estimator inside the Pipeline
        "svc__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "svc__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
    },
    "RandomForest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    },
    "HistGradientBoosting": {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [10, 20, 50],
        "l2_regularization": [0.1, 1.0, 10.0, 50.0] 
    },
    "MLPClassifier": {
        "hidden_layer_sizes": [(128, 64, 32), (128, 64), (64, 32)],
        "alpha": [0.001, 0.01, 0.1, 1.0], 
        "learning_rate_init": [0.001, 0.01],
        "solver": ["adam"]
    }
}


class ClauseClassifier:
    """
    Handles training, automated hyperparameter tuning, evaluation, and serialization 
    of modular machine learning classifiers over dense semantic embedding spaces.
    """
    
    def __init__(self, estimator=None) -> None:
        if estimator is None:
            self.model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            logging.info("No estimator specified. Defaulting to Balanced Logistic Regression.")
        else:
            self.model = estimator
            logging.info(f"Initialized wrapper with custom estimator: {estimator.__class__.__name__}")

    def tune_and_evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        param_distributions: dict, 
        test_size: float = 0.2, 
        n_iter: int = 15, 
        cv: int = 5,
        random_state: int = 42
    ) -> dict:
        """
        Runs a randomized search cross-validation space over the training split, 
        updates the underlying estimator to the best found instance, and evaluates on test.
        """
        logging.info(f"Splitting dataset (Test size: {test_size * 100}%)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        model_name = self.model.__class__.__name__
        logging.info(f"Starting automated tuning for {model_name} (n_iter={n_iter}, cv={cv})...")
        
        # Core automated search optimization block
        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="f1_macro",  # Balanced metric optimal for text classification tasks
            n_jobs=-1,           # Parallelize across all execution CPU cores
            random_state=random_state
        )
        
        search.fit(X_train, y_train)
        
        logging.info(f"Optimal parameters found: {search.best_params_}")
        logging.info(f"Best cross-validation macro F1 score: {search.best_score_:.4f}")
        
        # Override the current tracking instance model with the optimized, fitted estimator
        self.model = search.best_estimator_
        
        logging.info("Executing test partition inference evaluation...")
        predictions = self.model.predict(X_test)
        
        return classification_report(y_test, predictions, output_dict=True)

    def predict(self, query_embedding: np.ndarray) -> int:
        if self.model is None:
            raise ValueError("Classifier states are uninitialized. Train or load a model first.")
        return int(self.model.predict(query_embedding)[0])

    def save_model(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "clf_estimator.joblib"
        joblib.dump(self.model, model_path)
        logging.info(f"Model estimator successfully frozen at: {model_path}")

    def load_model(self, output_dir: Path) -> None:
        model_path = output_dir / "clf_estimator.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No serial structures found at target pathway: {model_path}")
        self.model = joblib.load(model_path)
        logging.info(f"Model framework successfully restored from {model_path}")