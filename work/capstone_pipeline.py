"""Leakage-aware capstone pipeline for refresh-review prioritization.

Public artifacts contain aggregates only. Pseudonymous IDs are retained only in the
gitignored private queue used for human review.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
RELEASE = "flyrank_pseudonymized_warehouse_release_v20260703"
FACT_ROOT = "hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance"
FEATURE_COLUMNS = [
    "log_impressions", "ctr", "avg_position", "active_days",
    "click_days", "position_days", "impression_momentum", "click_momentum",
]
ID_COLUMNS = ["client_hash_id", "content_hash_id"]

# Each anchor is feature month -> strictly later outcome month.
WINDOWS = {
    "train_2026_01": ("2026-01", "2026-02"),
    "train_2026_02": ("2026-02", "2026-03"),
    "development_2026_03": ("2026-03", "2026-04"),
    "refit_2026_04": ("2026-04", "2026-05"),
    "sealed_test_2026_05": ("2026-05", "2026-06"),
}


def _month_relation(month: str) -> str:
    return (
        "read_parquet('" + FACT_ROOT + f"/month={month}/*.parquet', "
        "hive_partitioning=true)"
    )


def _canonical_daily_sql(month: str) -> str:
    """Return valid, de-duplicated GSC observations at page-day grain.

    The release schema permits nullable keys. Those rows cannot represent a usable
    page-day and are excluded. DISTINCT is intentionally limited to the GSC fields
    used below: it removes repeated copies of the same observation but does not
    combine conflicting observations. ``verify_partition`` rejects conflicts first.
    """
    relation = _month_relation(month)
    return f"""
        SELECT DISTINCT
               report_date, client_hash_id, content_hash_id,
               gsc_data_available, gsc_impressions, gsc_clicks,
               gsc_sum_position, gsc_avg_position
        FROM {relation}
        WHERE report_date IS NOT NULL
          AND client_hash_id IS NOT NULL
          AND content_hash_id IS NOT NULL
          AND gsc_data_available IS TRUE
    """


def _aggregate_sql(month: str, prefix: str) -> str:
    """Aggregate canonical page-days to client-page-month grain.

    Clicks, impressions, and sum-position are additive. CTR is calculated later as
    total clicks / total impressions. Average position is impression-weighted via
    total sum-position / total impressions; daily averages are never summed.
    """
    daily = _canonical_daily_sql(month)
    return f"""
        SELECT client_hash_id, content_hash_id,
               SUM(gsc_impressions) AS {prefix}_impressions,
               SUM(gsc_clicks) AS {prefix}_clicks,
               SUM(gsc_sum_position) / NULLIF(SUM(gsc_impressions), 0) AS {prefix}_avg_position,
               COUNT(*) FILTER (WHERE gsc_data_available IS TRUE) AS {prefix}_available_days,
               COUNT(*) FILTER (WHERE gsc_impressions > 0) AS {prefix}_active_days,
               COUNT(*) FILTER (WHERE gsc_clicks > 0) AS {prefix}_click_days,
               COUNT(*) FILTER (WHERE gsc_impressions > 0 AND gsc_avg_position IS NOT NULL)
                   AS {prefix}_position_days,
               SUM(gsc_impressions) FILTER (WHERE DAY(report_date) <= 14)
                   AS {prefix}_impressions_early,
               SUM(gsc_impressions) FILTER (WHERE DAY(report_date) > 14)
                   AS {prefix}_impressions_late,
               SUM(gsc_clicks) FILTER (WHERE DAY(report_date) <= 14)
                   AS {prefix}_clicks_early,
               SUM(gsc_clicks) FILTER (WHERE DAY(report_date) > 14)
                   AS {prefix}_clicks_late
        FROM ({daily}) AS daily
        GROUP BY client_hash_id, content_hash_id
    """


def connect_warehouse(token: str | None = None):
    """Create a DuckDB connection without ever logging the credential."""
    import duckdb

    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set; authorized warehouse extraction was not run.")
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs")
    con.execute(
        "CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN ?)", [token]
    )
    return con


def _audit_relation(con, relation: str, month: str) -> dict:
    """Audit nullable keys and valid-key GSC variants without exposing row data."""
    row = con.sql(f"""
        WITH source AS (
            SELECT * FROM {relation}
        ),
        valid AS (
            SELECT * FROM source
            WHERE report_date IS NOT NULL
              AND client_hash_id IS NOT NULL
              AND content_hash_id IS NOT NULL
        ),
        key_groups AS (
            SELECT report_date, client_hash_id, content_hash_id,
                   COUNT(*) AS physical_rows,
                   COUNT(DISTINCT (
                       gsc_data_available, gsc_impressions, gsc_clicks,
                       gsc_sum_position, gsc_avg_position
                   )) AS gsc_variants
            FROM valid
            GROUP BY report_date, client_hash_id, content_hash_id
        )
        SELECT
            (SELECT COUNT(*) FROM source) AS rows,
            (SELECT MIN(report_date) FROM source) AS min_date,
            (SELECT MAX(report_date) FROM source) AS max_date,
            (SELECT COUNT(*) FROM source WHERE report_date IS NULL) AS null_date_rows,
            (SELECT COUNT(*) FROM source WHERE client_hash_id IS NULL) AS null_client_rows,
            (SELECT COUNT(*) FROM source WHERE content_hash_id IS NULL) AS null_content_rows,
            (SELECT COUNT(*) FROM source
             WHERE report_date IS NULL OR client_hash_id IS NULL OR content_hash_id IS NULL)
                AS invalid_key_rows,
            (SELECT COALESCE(SUM(physical_rows - 1), 0) FROM key_groups
             WHERE physical_rows > 1) AS repeated_valid_key_rows,
            (SELECT COUNT(*) FROM key_groups
             WHERE physical_rows > 1 AND gsc_variants = 1) AS exact_gsc_duplicate_groups,
            (SELECT COUNT(*) FROM key_groups
             WHERE gsc_variants > 1) AS conflicting_gsc_groups,
            (SELECT COALESCE(MAX(physical_rows), 1) FROM key_groups) AS max_key_multiplicity,
            (SELECT COUNT(*) FROM key_groups) AS valid_page_days
    """).df().iloc[0]
    return {
        "month": month,
        "rows": int(row["rows"]),
        "min_date": str(row["min_date"]),
        "max_date": str(row["max_date"]),
        "null_date_rows": int(row["null_date_rows"]),
        "null_client_rows": int(row["null_client_rows"]),
        "null_content_rows": int(row["null_content_rows"]),
        "invalid_key_rows": int(row["invalid_key_rows"]),
        "repeated_valid_key_rows": int(row["repeated_valid_key_rows"]),
        "exact_gsc_duplicate_groups": int(row["exact_gsc_duplicate_groups"]),
        "conflicting_gsc_groups": int(row["conflicting_gsc_groups"]),
        "max_key_multiplicity": int(row["max_key_multiplicity"]),
        "valid_page_days": int(row["valid_page_days"]),
    }


def verify_partition(con, month: str) -> dict:
    """Audit a warehouse partition and return public-safe aggregate diagnostics."""
    return _audit_relation(con, _month_relation(month), month)


def extract_snapshot(con, feature_month: str, outcome_month: str) -> pd.DataFrame:
    """Create one temporally ordered feature/outcome snapshot."""
    if feature_month >= outcome_month:
        raise ValueError("Feature month must be strictly before outcome month.")
    feature_sql = _aggregate_sql(feature_month, "f")
    outcome_sql = _aggregate_sql(outcome_month, "y")
    frame = con.sql(f"""
        WITH f AS ({feature_sql}), y AS ({outcome_sql})
        SELECT f.*, y.y_impressions, y.y_available_days, y.y_active_days
        FROM f INNER JOIN y USING (client_hash_id, content_hash_id)
        WHERE f.f_available_days >= 20
          AND y.y_available_days >= 20
          AND f.f_impressions >= 100
    """).df()
    return engineer_snapshot(frame, feature_month, outcome_month)


def engineer_snapshot(
    frame: pd.DataFrame, feature_month: str, outcome_month: str
) -> pd.DataFrame:
    """Engineer pre-outcome features and the independently observed proxy label."""
    required = {
        *ID_COLUMNS, "f_impressions", "f_clicks", "f_avg_position",
        "f_available_days", "f_active_days", "f_click_days", "f_position_days",
        "f_impressions_early", "f_impressions_late", "f_clicks_early",
        "f_clicks_late", "y_impressions", "y_available_days", "y_active_days",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Snapshot is missing required fields: {sorted(missing)}")
    out = frame.copy()
    out["log_impressions"] = np.log1p(out["f_impressions"])
    out["ctr"] = out["f_clicks"] / out["f_impressions"].replace(0, np.nan)
    out["avg_position"] = out["f_avg_position"].where(out["f_avg_position"] > 0)
    out["active_days"] = out["f_active_days"]
    out["click_days"] = out["f_click_days"]
    out["position_days"] = out["f_position_days"]
    out["impression_momentum"] = (
        (out["f_impressions_late"].fillna(0) + 1)
        / (out["f_impressions_early"].fillna(0) + 1)
    )
    out["click_momentum"] = (
        (out["f_clicks_late"].fillna(0) + 1)
        / (out["f_clicks_early"].fillna(0) + 1)
    )
    out["future_impression_ratio"] = out["y_impressions"] / out["f_impressions"]
    out["target_future_decline"] = (out["future_impression_ratio"] <= 0.80).astype(int)
    out["feature_month"] = feature_month
    out["outcome_month"] = outcome_month
    return out


def baseline_score(frame: pd.DataFrame) -> pd.Series:
    """Transparent demand × recent deterioration × reachable-position rule."""
    demand = np.log1p(frame["f_impressions"])
    deterioration = (1 - frame["impression_momentum"].clip(0, 1)).clip(0, 1)
    position = frame["avg_position"].between(4, 20).astype(float)
    low_ctr = (1 - (frame["ctr"].fillna(0) / 0.10).clip(0, 1))
    return demand * (0.50 * deterioration + 0.30 * position + 0.20 * low_ctr)


def make_model() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return Pipeline([
        ("prepare", ColumnTransformer([("numeric", numeric, FEATURE_COLUMNS)])),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ])


def precision_at_k(y_true: Iterable[int], scores: Iterable[float], k: int) -> float:
    table = pd.DataFrame({"y": y_true, "score": scores}).sort_values(
        "score", ascending=False, kind="mergesort"
    )
    if table.empty:
        return float("nan")
    return float(table.head(min(k, len(table)))["y"].mean())


def ranking_metrics(y_true: pd.Series, scores: np.ndarray, ks=(20, 50)) -> dict:
    y = np.asarray(y_true, dtype=int)
    result = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "average_precision": float(average_precision_score(y, scores)),
    }
    result["roc_auc"] = float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None
    for k in ks:
        result[f"precision_at_{k}"] = precision_at_k(y, scores, k)
    return result


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> tuple[Pipeline, dict, pd.DataFrame]:
    """Fit once and compare model/rule on the identical held-out rows."""
    if set(FEATURE_COLUMNS) & {"y_impressions", "future_impression_ratio", "target_future_decline"}:
        raise AssertionError("Target-derived field entered the feature list.")
    if not (train["outcome_month"].max() <= test["feature_month"].min()):
        raise AssertionError("Training outcomes overlap or follow the test feature window.")
    model = make_model().fit(train[FEATURE_COLUMNS], train["target_future_decline"])
    scored = test.copy()
    scored["model_score"] = model.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
    scored["baseline_score"] = baseline_score(scored)
    metrics = {
        "model": ranking_metrics(scored["target_future_decline"], scored["model_score"]),
        "baseline": ranking_metrics(scored["target_future_decline"], scored["baseline_score"]),
        "client_overlap_count": int(len(set(train.client_hash_id) & set(test.client_hash_id))),
        "train_clients": int(train.client_hash_id.nunique()),
        "test_clients": int(test.client_hash_id.nunique()),
        "feature_columns": FEATURE_COLUMNS,
    }
    return model, metrics, scored


def add_recommendations(scored: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates and attach deterministic, human-readable review reasons."""
    queue = scored.sort_values(
        ["model_score", "f_impressions"], ascending=[False, False], kind="mergesort"
    ).copy()
    queue["rank"] = np.arange(1, len(queue) + 1)
    conditions = [
        queue["impression_momentum"] < 0.8,
        queue["avg_position"].between(4, 20),
        queue["ctr"] < 0.02,
    ]
    queue["reason_code"] = np.select(
        conditions,
        ["RECENT_VISIBILITY_DECAY", "REACHABLE_SEARCH_POSITION", "LOW_CTR_WITH_DEMAND"],
        default="MULTIVARIATE_DECLINE_RISK",
    )
    queue["action_label"] = np.select(
        [conditions[2], conditions[1]],
        ["REVIEW_SNIPPET_AND_INTENT", "REVIEW_CONTENT_COVERAGE"],
        default="REVIEW_FRESHNESS_AND_RELEVANCE",
    )
    queue["limitation"] = (
        "Observed decline risk is not evidence that a refresh will cause recovery."
    )
    return queue


def write_artifacts(
    dev_metrics: dict,
    final_metrics: dict,
    final_queue: pd.DataFrame,
    output_dir: str | Path = "work/outputs",
) -> None:
    """Write receipts plus a private, gitignored row-level queue."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "verified_warehouse_run",
        "release": RELEASE,
        "random_seed": SEED,
        "target": "next-month impressions <= 80% of feature-month impressions",
        "development": dev_metrics,
        "sealed_final": final_metrics,
    }
    (output / "capstone_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    private_columns = [
        "rank", *ID_COLUMNS, "model_score", "baseline_score", "reason_code",
        "action_label", "limitation",
    ]
    final_queue[private_columns].to_csv(output / "ranked_recommendations.csv", index=False)
    public = final_queue.groupby(["reason_code", "action_label"], as_index=False).agg(
        recommendations=("rank", "size"),
        mean_model_score=("model_score", "mean"),
    )
    public.to_json(output / "recommendation_summary.json", orient="records", indent=2)


def run_remote(output_dir: str | Path = "work/outputs") -> dict:
    """Run development once, then the sealed June evaluation once."""
    con = connect_warehouse()
    partition_audit = [verify_partition(con, m) for m in ("2026-03", "2026-06")]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "partition_audit.json").write_text(
        json.dumps(partition_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    conflicts = [
        item for item in partition_audit if item["conflicting_gsc_groups"] > 0
    ]
    if conflicts:
        summary = ", ".join(
            f"{item['month']}: {item['conflicting_gsc_groups']} conflicting groups"
            for item in conflicts
        )
        raise AssertionError(
            "Warehouse page-day grain check found multiple different GSC observations "
            f"for the same valid key ({summary}). Extraction was stopped."
        )
    snapshots = {
        name: extract_snapshot(con, feature, outcome)
        for name, (feature, outcome) in WINDOWS.items()
    }
    train = pd.concat([snapshots["train_2026_01"], snapshots["train_2026_02"]])
    _, dev_metrics, _ = evaluate(train, snapshots["development_2026_03"])
    refit = pd.concat([
        train, snapshots["development_2026_03"], snapshots["refit_2026_04"]
    ])
    _, final_metrics, final_scored = evaluate(refit, snapshots["sealed_test_2026_05"])
    queue = add_recommendations(final_scored)
    write_artifacts(dev_metrics, final_metrics, queue, output_dir)
    return {"partition_audit": partition_audit, "development": dev_metrics, "sealed_final": final_metrics}

