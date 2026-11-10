import unittest

from newsrec.content import CandidateScore, ContentIndex
from newsrec.fusion import fuse_candidates, generation_budget, retrieval_agreement


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.news = {
            "N1": {"title": "Football club wins final", "abstract": "League sports game"},
            "N2": {"title": "Football team signs player", "abstract": "League sports transfer"},
            "N3": {"title": "Stocks rise on earnings", "abstract": "Market business finance"},
            "N4": {"title": "Sports championship results", "abstract": "Football final game"},
        }

    def test_content_retrieval_excludes_history(self):
        index = ContentIndex().fit(self.news)
        result = index.search(["N1"], 3)
        self.assertNotIn("N1", [item.news_id for item in result])
        self.assertIn(result[0].news_id, {"N2", "N4"})
        self.assertGreater(index.score(["N1"], "N2"), index.score(["N1"], "N3"))

    def test_adaptive_budget_responds_to_evidence(self):
        self.assertGreater(generation_budget(50, 20, 0.8), generation_budget(50, 1, 0.1))
        gen = [CandidateScore("N1", 2.0, "generation"), CandidateScore("N2", 1.0, "generation")]
        con = [CandidateScore("N1", 2.0, "content"), CandidateScore("N3", 1.0, "content")]
        self.assertEqual(retrieval_agreement(gen, con, 2), 0.5)
        fused = fuse_candidates(gen, con, 3, 10)
        self.assertEqual(len({item.news_id for item in fused}), 3)
        self.assertEqual(fused[0].news_id, "N1")
        self.assertEqual(set(fused[0].sources), {"generation", "content"})


if __name__ == "__main__":
    unittest.main()
