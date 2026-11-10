"""Small, dependency-free checks for the MIND exposure and time split contract."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from newsrec.data import (
    Candidate,
    Impression,
    build_retrieval_examples,
    iter_exposed_candidates,
    merge_news_catalogs,
    read_behaviors,
    read_news,
    temporal_split,
)


class MindDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def write(self, name: str, content: str) -> Path:
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_news_and_behavior_parsing(self) -> None:
        news_path = self.write(
            "news.tsv",
            "N1\tnews\tworld\tFirst title\tFirst abstract\thttps://a\t[]\t[]\n"
            "N2\tsports\tfootball\tSecond title\t\thttps://b\t[]\t[]\n",
        )
        behaviors_path = self.write(
            "behaviors.tsv",
            "1\tU1\t11/15/2019 9:55:12 AM\tN1\tN2-1 N1-0\n"
            "2\tU2\t11/15/2019 10:00:00 AM\t\tN1-0 N2-1\n",
        )
        news = read_news(news_path)
        behaviors = read_behaviors(behaviors_path)
        self.assertEqual(set(news), {"N1", "N2"})
        self.assertEqual(news["N2"].abstract, "")
        self.assertEqual(behaviors[0].history, ("N1",))
        self.assertEqual(behaviors[0].candidates, (Candidate("N2", 1), Candidate("N1", 0)))
        self.assertEqual(behaviors[1].history, ())
        self.assertEqual(behaviors[0].timestamp, datetime(2019, 11, 15, 9, 55, 12))

    def test_only_exposed_candidates_have_labels(self) -> None:
        rows = read_behaviors(
            self.write("behaviors.tsv", "1\tU1\t11/15/2019 9:55:12 AM\tN0\tN1-1 N2-0\n")
        )
        labeled = [(candidate.news_id, candidate.label) for _, candidate in iter_exposed_candidates(rows)]
        self.assertEqual(labeled, [("N1", 1), ("N2", 0)])

    def test_retrieval_targets_use_only_pre_impression_history(self) -> None:
        rows = read_behaviors(
            self.write(
                "behaviors.tsv",
                "1\tU1\t11/15/2019 9:55:12 AM\tN0\tN1-1 N2-1 N3-0 N0-1\n"
                "2\tU2\t11/15/2019 10:00:00 AM\t\tN4-1 N5-0\n",
            )
        )
        examples = build_retrieval_examples(rows)
        self.assertEqual([x.positive_news_id for x in examples], ["N1", "N2"])
        self.assertTrue(all(x.history == ("N0",) for x in examples))
        self.assertEqual(
            [x.positive_news_id for x in build_retrieval_examples(rows, min_history=0, exclude_seen_targets=False)],
            ["N1", "N2", "N0", "N4"],
        )

    def test_split_keeps_equal_times_together(self) -> None:
        rows = read_behaviors(
            self.write(
                "behaviors.tsv",
                "4\tU1\t11/15/2019 11:00:00 AM\tN1\tN4-0\n"
                "1\tU1\t11/15/2019 9:00:00 AM\tN1\tN2-1\n"
                "3\tU2\t11/15/2019 10:00:00 AM\tN1\tN3-0\n"
                "2\tU1\t11/15/2019 10:00:00 AM\tN1\tN2-0\n",
            )
        )
        train, valid = temporal_split(rows, val_fraction=0.5)
        self.assertEqual([x.impression_id for x in train], ["1", "2", "3"])
        self.assertEqual([x.impression_id for x in valid], ["4"])
        self.assertLess(max(x.timestamp for x in train), min(x.timestamp for x in valid))

    def test_bad_rows_and_unsplittable_data_fail(self) -> None:
        bad_label = self.write("bad.tsv", "1\tU1\t11/15/2019 9:00:00 AM\t\tN1-2\n")
        with self.assertRaisesRegex(ValueError, "invalid candidate"):
            read_behaviors(bad_label)
        unlabeled = self.write("test.tsv", "1\tU1\t11/15/2019 9:00:00 AM\t\tN1 N2\n")
        with self.assertRaisesRegex(ValueError, "unlabeled candidate"):
            read_behaviors(unlabeled)
        malformed_news = self.write("bad-news.tsv", "N1\tcategory\tTitle\n")
        with self.assertRaisesRegex(ValueError, "expected 8 news columns"):
            read_news(malformed_news)
        same_time = [
            Impression(str(i), "U1", datetime(2019, 11, 15), (), (Candidate("N1", 0),))
            for i in range(2)
        ]
        with self.assertRaisesRegex(ValueError, "distinct timestamps"):
            temporal_split(same_time)

    def test_merge_catalogs_rejects_conflicts(self) -> None:
        a = read_news(self.write("a.tsv", "N1\ta\tb\tTitle\tAbstract\turl\t[]\t[]\n"))
        b = read_news(self.write("b.tsv", "N1\ta\tb\tOther title\tAbstract\turl\t[]\t[]\n"))
        with self.assertRaisesRegex(ValueError, "conflicting metadata"):
            merge_news_catalogs(a, b)


if __name__ == "__main__":
    unittest.main()
