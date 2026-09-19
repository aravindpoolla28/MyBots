import unittest

import pandas as pd

from session_pattern_match import build_session_history, classify_move, summarize_pattern


class SessionPatternMatchTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            [
                # 2025-01-01
                {"timestamp": "2025-01-01 00:00:00+00:00", "open": 100.0, "close": 101.0},
                {"timestamp": "2025-01-01 01:00:00+00:00", "open": 101.0, "close": 101.5},
                {"timestamp": "2025-01-01 07:00:00+00:00", "open": 101.5, "close": 102.0},
                {"timestamp": "2025-01-01 08:00:00+00:00", "open": 102.0, "close": 101.5},
                {"timestamp": "2025-01-01 15:00:00+00:00", "open": 101.5, "close": 102.3},
                {"timestamp": "2025-01-01 16:00:00+00:00", "open": 102.3, "close": 103.0},
                {"timestamp": "2025-01-01 23:00:00+00:00", "open": 103.0, "close": 104.0},
                # 2025-01-02
                {"timestamp": "2025-01-02 00:00:00+00:00", "open": 104.0, "close": 104.5},
                {"timestamp": "2025-01-02 01:00:00+00:00", "open": 104.5, "close": 105.0},
                {"timestamp": "2025-01-02 07:00:00+00:00", "open": 105.0, "close": 106.0},
                {"timestamp": "2025-01-02 08:00:00+00:00", "open": 106.0, "close": 105.3},
                {"timestamp": "2025-01-02 15:00:00+00:00", "open": 105.3, "close": 106.1},
                {"timestamp": "2025-01-02 16:00:00+00:00", "open": 106.1, "close": 106.7},
                {"timestamp": "2025-01-02 23:00:00+00:00", "open": 106.7, "close": 107.5},
                # 2025-01-03
                {"timestamp": "2025-01-03 00:00:00+00:00", "open": 107.5, "close": 108.2},
                {"timestamp": "2025-01-03 01:00:00+00:00", "open": 108.2, "close": 108.9},
                {"timestamp": "2025-01-03 07:00:00+00:00", "open": 108.9, "close": 109.8},
                {"timestamp": "2025-01-03 08:00:00+00:00", "open": 109.8, "close": 108.6},
                {"timestamp": "2025-01-03 15:00:00+00:00", "open": 108.6, "close": 109.0},
                {"timestamp": "2025-01-03 16:00:00+00:00", "open": 109.0, "close": 109.8},
                {"timestamp": "2025-01-03 23:00:00+00:00", "open": 109.8, "close": 110.6},
                # 2025-01-04
                {"timestamp": "2025-01-04 00:00:00+00:00", "open": 110.6, "close": 110.9},
                {"timestamp": "2025-01-04 01:00:00+00:00", "open": 110.9, "close": 111.4},
                {"timestamp": "2025-01-04 07:00:00+00:00", "open": 111.4, "close": 112.0},
                {"timestamp": "2025-01-04 08:00:00+00:00", "open": 112.0, "close": 111.2},
                {"timestamp": "2025-01-04 15:00:00+00:00", "open": 111.2, "close": 111.8},
                {"timestamp": "2025-01-04 16:00:00+00:00", "open": 111.8, "close": 112.6},
                {"timestamp": "2025-01-04 23:00:00+00:00", "open": 112.6, "close": 113.8},
                # 2025-01-05
                {"timestamp": "2025-01-05 00:00:00+00:00", "open": 113.8, "close": 114.4},
                {"timestamp": "2025-01-05 01:00:00+00:00", "open": 114.4, "close": 115.0},
                {"timestamp": "2025-01-05 07:00:00+00:00", "open": 115.0, "close": 115.7},
                {"timestamp": "2025-01-05 08:00:00+00:00", "open": 115.7, "close": 114.9},
                {"timestamp": "2025-01-05 15:00:00+00:00", "open": 114.9, "close": 115.5},
                {"timestamp": "2025-01-05 16:00:00+00:00", "open": 115.5, "close": 116.1},
                {"timestamp": "2025-01-05 23:00:00+00:00", "open": 116.1, "close": 117.2},
            ]
        )

    def test_classify_move(self):
        self.assertEqual(classify_move(0.015), "up")
        self.assertEqual(classify_move(-0.015), "down")
        self.assertEqual(classify_move(0.001), "flat")

    def test_build_history_matches_pattern(self):
        history = build_session_history(self.df)
        self.assertIn("pattern", history.columns)
        self.assertIn("ny_label", history.columns)
        self.assertGreater(len(history), 0)

    def test_summarize_pattern(self):
        history = build_session_history(self.df)
        result = summarize_pattern(
            history,
            open_label="flat",
            asia_label="up",
            london_label="down",
            flat_tolerance=0.01,
        )
        self.assertIn("historical days", result)
        self.assertIn("NY", result)
        self.assertIn("up", result.lower())


if __name__ == "__main__":
    unittest.main()
