import csv
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

from newsrec.data import Candidate, Impression, NewsArticle
from newsrec.minionerec_adapter import export_catalog, export_item_info, export_split, load_sid_index, load_upstream_predictions


def article(news_id, title):
    return NewsArticle(news_id, "news", "", title, "Abstract", "", "", "")


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.news = {"N1": article("N1", "A title"), "N2": article("N2", "Another title"), "N3": article("N3", "Third title")}
        self.impressions = [Impression("I1", "U1", datetime(2020, 1, 1), ("N1",), (Candidate("N2", 1), Candidate("N3", 0)))]

    def test_export_and_import_are_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            ids = export_catalog(self.news, directory)
            index_path = directory / "MIND.index.json"
            index_path.write_text(json.dumps({str(i): [f"<a_{i}>", "<b_0>", "<c_0>"] for i in range(3)}), encoding="utf-8")
            index = load_sid_index(index_path, 3)
            self.assertEqual(export_split(self.impressions, self.news, ids, directory, "valid", index), 1)
            export_item_info(self.news, ids, index, directory)
            with (directory / "valid/MIND.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["item_sid"], "<a_1><b_0><c_0>")
            result_path = directory / "result.json"
            result_path.write_text(json.dumps([{"output": "<a_1><b_0><c_0>\n", "predict": ["<a_1><b_0><c_0>", "bad", "<a_2><b_0><c_0>"]}]), encoding="utf-8")
            loaded = load_upstream_predictions(result_path, directory / "valid/MIND.queries.jsonl", index, ids)
            self.assertEqual([item.news_id for item in loaded["I1"]], ["N2", "N3"])

    def test_reject_sid_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text(json.dumps({"0": ["<a_1>"], "1": ["<a_1>"], "2": ["<a_2>"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "collision"):
                load_sid_index(path, 3)


if __name__ == "__main__":
    unittest.main()
