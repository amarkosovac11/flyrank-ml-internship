import unittest
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

import capstone_pipeline
from capstone_pipeline import (
    FEATURE_COLUMNS,
    _aggregate_sql,
    _audit_relation,
    add_recommendations,
    baseline_score,
    engineer_snapshot,
    precision_at_k,
)


class CapstonePipelineTests(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect()
        self.con.execute("""
            CREATE TABLE raw_daily (
                report_date DATE,
                client_hash_id VARCHAR,
                content_hash_id VARCHAR,
                gsc_data_available BOOLEAN,
                gsc_impressions BIGINT,
                gsc_clicks BIGINT,
                gsc_sum_position BIGINT,
                gsc_avg_position DOUBLE
            )
        """)

    def tearDown(self):
        self.con.close()

    def test_precision_at_k(self):
        self.assertEqual(precision_at_k([0, 1, 1], [0.1, 0.9, 0.8], 2), 1.0)

    def test_engineering_uses_future_only_for_target(self):
        raw = pd.DataFrame({
            "client_hash_id": ["c"], "content_hash_id": ["p"],
            "f_impressions": [200], "f_clicks": [10], "f_avg_position": [8.0],
            "f_available_days": [31], "f_active_days": [25], "f_click_days": [8],
            "f_position_days": [25], "f_impressions_early": [120],
            "f_impressions_late": [80], "f_clicks_early": [6], "f_clicks_late": [4],
            "y_impressions": [150], "y_available_days": [30], "y_active_days": [20],
        })
        out = engineer_snapshot(raw, "2026-05", "2026-06")
        self.assertEqual(out.loc[0, "target_future_decline"], 1)
        self.assertTrue(set(FEATURE_COLUMNS).isdisjoint({"y_impressions", "future_impression_ratio"}))

    def test_baseline_and_reasons(self):
        frame = pd.DataFrame({
            "f_impressions": [1000], "impression_momentum": [0.5],
            "avg_position": [8.0], "ctr": [0.01], "model_score": [0.8],
            "baseline_score": [0.0],
        })
        frame["baseline_score"] = baseline_score(frame)
        self.assertGreater(frame.loc[0, "baseline_score"], 0)
        queue = add_recommendations(frame)
        self.assertEqual(queue.loc[0, "reason_code"], "RECENT_VISIBILITY_DECAY")
        self.assertEqual(queue.loc[0, "rank"], 1)

    def test_nullable_keys_are_not_misreported_as_valid_key_conflicts(self):
        self.con.execute("""
            INSERT INTO raw_daily VALUES
              (DATE '2026-06-01', NULL, 'p-null', TRUE, 9, 1, 45, 5.0),
              (DATE '2026-06-01', NULL, 'p-null', TRUE, 9, 1, 45, 5.0),
              (DATE '2026-06-01', 'c1', 'p1', TRUE, 100, 10, 500, 5.0)
        """)
        audit = _audit_relation(self.con, "raw_daily", "2026-06")
        self.assertEqual(audit["invalid_key_rows"], 2)
        self.assertEqual(audit["valid_page_days"], 1)
        self.assertEqual(audit["conflicting_gsc_groups"], 0)

    def test_exact_repeats_are_deduplicated_before_weighted_aggregation(self):
        self.con.execute("""
            INSERT INTO raw_daily VALUES
              (DATE '2026-06-01', 'c1', 'p1', TRUE, 100, 10, 500, 5.0),
              (DATE '2026-06-01', 'c1', 'p1', TRUE, 100, 10, 500, 5.0),
              (DATE '2026-06-02', 'c1', 'p1', TRUE, 50, 5, 500, 10.0)
        """)
        audit = _audit_relation(self.con, "raw_daily", "2026-06")
        self.assertEqual(audit["exact_gsc_duplicate_groups"], 1)
        self.assertEqual(audit["conflicting_gsc_groups"], 0)
        with patch.object(capstone_pipeline, "_month_relation", return_value="raw_daily"):
            monthly = self.con.sql(_aggregate_sql("2026-06", "f")).df().iloc[0]
        self.assertEqual(monthly["f_impressions"], 150)
        self.assertEqual(monthly["f_clicks"], 15)
        self.assertEqual(monthly["f_active_days"], 2)
        self.assertAlmostEqual(monthly["f_avg_position"], 1000 / 150)

    def test_conflicting_valid_page_day_still_fails_grain_audit(self):
        self.con.execute("""
            INSERT INTO raw_daily VALUES
              (DATE '2026-06-01', 'c1', 'p1', TRUE, 100, 10, 500, 5.0),
              (DATE '2026-06-01', 'c1', 'p1', TRUE, 120, 10, 600, 5.0)
        """)
        audit = _audit_relation(self.con, "raw_daily", "2026-06")
        self.assertEqual(audit["conflicting_gsc_groups"], 1)


if __name__ == "__main__":
    unittest.main()
