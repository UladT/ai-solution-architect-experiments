"""Unit tests for disaster MCP data functions."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from task5_hybrid_chatbot.mcp_servers import disaster_server


class DisasterServerDataTests(unittest.TestCase):
    """Test core disaster analytics behavior."""

    def test_load_disaster_frames_combines_csv_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir)
            df1 = pd.DataFrame(
                [
                    {
                        "Year": 2000,
                        "Country": "India",
                        "Disaster Type": "Flood",
                        "Total Deaths": 100,
                        "Total Affected": 5000,
                    }
                ]
            )
            df2 = pd.DataFrame(
                [
                    {
                        "Year": 2001,
                        "Country": "Japan",
                        "Disaster Type": "Earthquake",
                        "Total Deaths": 50,
                        "Total Affected": 1200,
                    }
                ]
            )
            df1.to_csv(path / "a.csv", index=False)
            df2.to_csv(path / "b.csv", index=False)

            combined = disaster_server._load_disaster_frames(tmp_dir)

            self.assertEqual(len(combined), 2)
            self.assertIn("_source_file", combined.columns)
            self.assertIn("Total Deaths", combined.columns)

    def test_filter_frame_respects_country_type_and_year(self) -> None:
        frame = pd.DataFrame(
            [
                {"Year": 2000, "Country": "India", "Disaster Type": "Flood"},
                {"Year": 2005, "Country": "India", "Disaster Type": "Earthquake"},
                {"Year": 2004, "Country": "Japan", "Disaster Type": "Flood"},
            ]
        )

        filtered = disaster_server._filter_frame(
            frame,
            country="India",
            disaster_type="flood",
            start_year=1999,
            end_year=2001,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["Country"], "India")
        self.assertEqual(filtered.iloc[0]["Disaster Type"], "Flood")


class DisasterServerToolTests(unittest.TestCase):
    """Test public tool-format helpers without starting MCP runtime."""

    def setUp(self) -> None:
        self.original_loader = disaster_server._load_disaster_frames

    def tearDown(self) -> None:
        disaster_server._load_disaster_frames = self.original_loader

    def test_get_disaster_summary_contains_aggregates(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "Year": 2020,
                    "Country": "India",
                    "Disaster Type": "Flood",
                    "Disaster Subtype": "Riverine flood",
                    "Event Name": "Monsoon Flood",
                    "Total Deaths": 120,
                    "Total Affected": 10000,
                },
                {
                    "Year": 2019,
                    "Country": "India",
                    "Disaster Type": "Flood",
                    "Disaster Subtype": "Flash flood",
                    "Event Name": "Flash Event",
                    "Total Deaths": 30,
                    "Total Affected": 2000,
                },
            ]
        )
        disaster_server._load_disaster_frames = lambda: frame

        result = disaster_server.get_disaster_summary(
            country="India",
            disaster_type="Flood",
            start_year=2018,
            end_year=2021,
            limit=2,
        )

        self.assertIn("events: 2", result)
        self.assertIn("total_deaths: 150", result)
        self.assertIn("total_affected: 12000", result)
        self.assertIn("Monsoon Flood", result)

    def test_top_disasters_by_metric_handles_unknown_metric(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "Year": 2020,
                    "Country": "Japan",
                    "Disaster Type": "Earthquake",
                    "Event Name": "Event A",
                    "Total Deaths": 70,
                }
            ]
        )
        disaster_server._load_disaster_frames = lambda: frame

        result = disaster_server.top_disasters_by_metric(metric="UnknownMetric")
        self.assertIn("is not available", result)


if __name__ == "__main__":
    unittest.main()
