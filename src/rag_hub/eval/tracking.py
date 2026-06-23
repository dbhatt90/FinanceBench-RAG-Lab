"""
MLflow tracking helpers for the unified eval framework.

Uses a local SQLite backend (./mlflow.db) so no server is required — MLflow 3.x
deprecated the plain file store. View with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

Kept dependency-light: mlflow is imported lazily inside the helpers so importing
this module never forces the dependency on callers that don't track.

Typical use:

    from rag_hub.eval.tracking import mlflow_run, log_metrics

    with mlflow_run(day=6, config_name="bge+crag", params={"k": 5}):
        log_metrics({"recall@5": 0.62, "mrr": 0.51})
"""

from contextlib import contextmanager
from typing import Dict, Optional

from rag_hub.config.settings import ROOT_DIR

EXPERIMENT_NAME = "financebench_retrieval"
TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
ARTIFACT_LOCATION = f"file:{ROOT_DIR / 'mlruns'}"


def _mlflow():
    import mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow


@contextmanager
def mlflow_run(day: int, config_name: str, params: Optional[Dict] = None):
    """Open one MLflow run tagged with the day + config, logging params.

    The run is named "day{N}-{config}" and tagged day=N, config=name so the
    dashboard can group/sort by day.
    """
    mlflow = _mlflow()
    with mlflow.start_run(run_name=f"day{day}-{config_name}"):
        mlflow.set_tags({"day": day, "config": config_name})
        if params:
            mlflow.log_params(params)
        yield


def log_metrics(metrics: Dict[str, float]) -> None:
    """Log a flat dict of metric_name -> value to the active run.

    Metric names are sanitised (MLflow disallows characters like '@')."""
    import mlflow
    clean = {_sanitize(k): float(v) for k, v in metrics.items() if v is not None}
    mlflow.log_metrics(clean)


def _sanitize(name: str) -> str:
    # MLflow metric keys allow alphanumerics, _-. / space; '@' is not allowed.
    return name.replace("@", "_at_")
