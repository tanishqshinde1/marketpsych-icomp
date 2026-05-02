"""StageProxy models and dynamic state summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover - compatibility for older scikit-learn
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import feature_columns


@dataclass
class FittedStageModel:
    name: str
    pipeline: Pipeline
    numeric_features: list[str]
    train_min: float
    train_max: float

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        x = prepare_model_frame(data, self.numeric_features)
        raw = self.pipeline.predict(x)
        if self.train_max > self.train_min:
            scaled = (raw - self.train_min) / (self.train_max - self.train_min)
        else:
            scaled = raw
        return np.clip(scaled, 0.0, 1.0)


def train_stage_model(
    weekly: pd.DataFrame,
    model_name: str = "ridge",
    random_state: int = 7,
) -> tuple[FittedStageModel, dict[str, float]]:
    """Train a StageProxy model against the future composite target."""
    train = weekly.dropna(subset=["target_stage_proxy"]).copy()
    if train.empty:
        raise ValueError("No rows with future target are available for model training.")

    numeric_features = feature_columns(train)
    x = prepare_model_frame(train, numeric_features)
    y = train["target_stage_proxy"].astype(float).to_numpy()

    split_idx = max(1, int(len(train) * 0.8))
    if split_idx >= len(train):
        split_idx = len(train) - 1
    train_idx = np.arange(len(train)) < split_idx

    pipeline = build_pipeline(model_name, numeric_features, random_state=random_state)
    pipeline.fit(x.iloc[train_idx], y[train_idx])

    raw_train_pred = pipeline.predict(x)
    fitted = FittedStageModel(
        name=model_name,
        pipeline=pipeline,
        numeric_features=numeric_features,
        train_min=float(np.min(raw_train_pred)),
        train_max=float(np.max(raw_train_pred)),
    )
    pred = fitted.predict(train)
    metrics = regression_metrics(y, pred, y[~train_idx], pred[~train_idx])
    return fitted, metrics


def compare_models(weekly: pd.DataFrame, random_state: int = 7) -> tuple[dict[str, FittedStageModel], pd.DataFrame]:
    fitted: dict[str, FittedStageModel] = {}
    rows = []
    for model_name in ["ridge", "mlp"]:
        model, metrics = train_stage_model(weekly, model_name=model_name, random_state=random_state)
        fitted[model_name] = model
        rows.append({"model": model_name, **metrics})
    return fitted, pd.DataFrame(rows)


def build_pipeline(model_name: str, numeric_features: list[str], random_state: int = 7) -> Pipeline:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility for older scikit-learn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("source", encoder, ["source_type"]),
        ],
        remainder="drop",
    )

    if model_name == "ridge":
        estimator = Ridge(alpha=1.0, random_state=random_state)
    elif model_name == "elasticnet":
        estimator = ElasticNet(alpha=0.001, l1_ratio=0.25, random_state=random_state, max_iter=5000)
    elif model_name == "mlp":
        estimator = MLPRegressor(
            hidden_layer_sizes=(24,),
            activation="relu",
            alpha=0.001,
            learning_rate_init=0.01,
            max_iter=2000,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.2,
        )
    else:
        raise ValueError("model_name must be one of: ridge, elasticnet, mlp")

    return Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])


def prepare_model_frame(data: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    frame = data.copy()
    for col in numeric_features:
        if col not in frame.columns:
            frame[col] = 0.0
    if "source_type" not in frame.columns:
        raise ValueError("source_type column is required for StageProxy prediction.")
    return frame[numeric_features + ["source_type"]]


def regression_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    y_val: np.ndarray,
    pred_val: np.ndarray,
) -> dict[str, float]:
    metrics = {
        "rmse_all": float(root_mean_squared_error(y_true, pred)),
        "mae_all": float(mean_absolute_error(y_true, pred)),
        "r2_all": float(r2_score(y_true, pred)) if len(y_true) > 1 else float("nan"),
    }
    if len(y_val) > 0:
        metrics.update(
            {
                "rmse_holdout": float(root_mean_squared_error(y_val, pred_val)),
                "mae_holdout": float(mean_absolute_error(y_val, pred_val)),
                "r2_holdout": float(r2_score(y_val, pred_val)) if len(y_val) > 1 else float("nan"),
            }
        )
    else:
        metrics.update({"rmse_holdout": float("nan"), "mae_holdout": float("nan"), "r2_holdout": float("nan")})
    return metrics


def attach_stage_predictions(weekly: pd.DataFrame, model: FittedStageModel) -> pd.DataFrame:
    out = weekly.copy()
    out["StageProxy"] = model.predict(out)
    out = out.sort_values(["source_type", "week"])
    out["DeltaStage"] = out.groupby("source_type")["StageProxy"].diff().fillna(0.0)
    return out


def global_stage(source_state: pd.DataFrame) -> pd.DataFrame:
    global_df = (
        source_state.groupby("week", as_index=False)
        .agg(
            GlobalStage=("StageProxy", "mean"),
            active_source_count=("active_source_count", "max"),
            diffusion_breadth=("diffusion_breadth", "max"),
        )
        .sort_values("week")
    )
    global_df["GlobalDeltaStage"] = global_df["GlobalStage"].diff().fillna(0.0)
    return global_df


def detect_emergence_events(
    global_df: pd.DataFrame,
    source_state: pd.DataFrame,
    percentiles: list[int] | None = None,
    activation_threshold: float = 0.05,
) -> pd.DataFrame:
    percentiles = percentiles or [75, 80, 90]
    rows = []
    active_source_state = source_state[source_state["intensity"] > activation_threshold]
    source_accel = (
        active_source_state.groupby("week", as_index=False)["DeltaStage"]
        .max()
        .rename(columns={"DeltaStage": "max_active_source_delta"})
    )
    merged = global_df.merge(source_accel, on="week", how="left")
    merged["max_active_source_delta"] = merged["max_active_source_delta"].fillna(float("-inf"))
    source_delta_threshold = active_source_state["DeltaStage"].quantile(0.90)
    for pct in percentiles:
        threshold = merged["GlobalDeltaStage"].quantile(pct / 100.0)
        flagged = merged[
            (merged["GlobalDeltaStage"] >= threshold)
            & (merged["GlobalDeltaStage"] > 0)
            & (
                (merged["active_source_count"] >= 2)
                | (merged["max_active_source_delta"] >= source_delta_threshold)
            )
        ].copy()
        flagged["threshold_percentile"] = pct
        flagged["threshold_value"] = float(threshold)
        rows.append(flagged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def source_activation_table(
    source_state: pd.DataFrame,
    activation_threshold: float = 0.05,
    absolute_infected_threshold: float = 0.65,
) -> pd.DataFrame:
    rows = []
    for source_type, group in source_state.groupby("source_type", observed=True):
        group = group.sort_values("week")
        active = group[group["intensity"] > activation_threshold]
        quantile_threshold = float(group["StageProxy"].quantile(0.75))
        infected_q = group[group["StageProxy"] >= quantile_threshold]
        infected_abs = group[group["StageProxy"] >= absolute_infected_threshold]
        rows.append(
            {
                "source_type": source_type,
                "first_activation_week": _first_week(active),
                "infected_threshold_quantile_75": quantile_threshold,
                "first_infected_proxy_week_quantile": _first_week(infected_q),
                "infected_threshold_absolute": absolute_infected_threshold,
                "first_infected_proxy_week_absolute": _first_week(infected_abs),
            }
        )
    return pd.DataFrame(rows)


def lead_lag_matrix(activation: pd.DataFrame, week_col: str = "first_infected_proxy_week_quantile") -> pd.DataFrame:
    table = activation.set_index("source_type")[week_col].dropna()
    sources = activation["source_type"].tolist()
    matrix = pd.DataFrame(index=sources, columns=sources, dtype=float)
    for row_source in sources:
        for col_source in sources:
            if row_source in table.index and col_source in table.index:
                delta = (pd.Timestamp(table[col_source]) - pd.Timestamp(table[row_source])).days / 7.0
                matrix.loc[row_source, col_source] = delta
    return matrix


def _first_week(frame: pd.DataFrame):
    if frame.empty:
        return pd.NaT
    return pd.Timestamp(frame.iloc[0]["week"])
