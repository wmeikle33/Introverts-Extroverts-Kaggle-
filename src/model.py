from __future__ import annotations
from pathlib import Path
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .features import auto_preprocess, split_features_label
from .metrics import ctr_metrics

def get_param_distributions(model_name : str):
    if model_name == 'logreg':
        return {
            "clf__C" : [0.01, 0.1, 1, 10],
            "clf__penalty": ["l2"],
        }
    if model_name == "xgb":
         return {
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__subsample": [0.7, 0.8, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 1.0],
        }

    raise ValueError(f"Unknown model_name={model_name!r}")
        


def build_pipeline(
    X: pd.DataFrame,
    model_name: str,
    random_state: int = 42,
) -> Pipeline:
    preprocessor = auto_preprocess(X)

    if model_name == "logreg":
        classifier = LogisticRegression(
            max_iter=200,
            random_state=random_state,
        )
    elif model_name == "xgb":
        if XGBClassifier is None:
            raise ImportError(
                "XGBoost is not installed. Run: pip install -e '.[xgb]'"
            )
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_name={model_name!r}")

    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", classifier),
        ]
    )

def train_eval_save(
    df: pd.DataFrame,
    label: str,
    model_path: str,
    model_name= str,
    random_state: int = 42,
    test_size: float = 0.2,
    param_search: dict = None,
    n_iter: int = 50,
    cv: int = 5,
) -> dict[str, float]:
    X, y = split_features_label(df, label)

    pipe = build_pipeline(X, model_name=model_name, random_state=random_state)

    stratify = y if y.nunique() <= 20 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if param_search:
        search = RandomizedSearchCV(
            pipe,
            param_distributions=get_param_distributions(model_name),
            n_iter=n_iter,
            scoring="neg_log_loss",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
    else:
        pipe.fit(X_train, y_train)

    metrics: dict[str, float] = {}

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, model_path)

    return metrics



def load_model(path: str):
    return load(path)
