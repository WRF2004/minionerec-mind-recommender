import unittest
from datetime import datetime

from newsrec.content import ContentIndex
from newsrec.data import Candidate, Impression, NewsArticle
from newsrec.metrics import ranking_metrics, recall_at_k
from newsrec.ranking import PairwiseRanker, build_rank_rows


def article(news_id, category, title):
    return NewsArticle(news_id, category, "", title, "", "", "", "")


class RankingTests(unittest.TestCase):
    def test_grouped_metrics_keep_unexposed_items_unlabeled(self):
        records = [("I1", 1, 0.9), ("I1", 0, 0.2), ("I2", 0, 0.1), ("I2", 1, 0.8)]
        metrics = ranking_metrics(records, (2,))
        self.assertEqual(metrics["auc"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(metrics["ndcg@2"], 1.0)
        recall = recall_at_k({"I1": ["N2", "N3"]}, {"I1": ["N3"]}, (1, 2))
        self.assertEqual(recall["recall@1"], 0.0)
        self.assertEqual(recall["recall@2"], 1.0)

    def test_pairwise_ranker_learns_from_logged_candidates(self):
        news = {
            "N1": article("N1", "sports", "football game team"),
            "N2": article("N2", "sports", "football team wins"),
            "N3": article("N3", "finance", "market stocks fall"),
        }
        impressions = [
            Impression("I1", "U1", datetime(2020, 1, 1), ("N1",), (Candidate("N2", 1), Candidate("N3", 0))),
            Impression("I2", "U2", datetime(2020, 1, 2), ("N1",), (Candidate("N3", 0), Candidate("N2", 1))),
        ]
        rows = build_rank_rows(impressions, news, ContentIndex().fit(news))
        scores = PairwiseRanker().fit(rows, epochs=20).predict(rows)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[3], scores[2])


if __name__ == "__main__":
    unittest.main()
