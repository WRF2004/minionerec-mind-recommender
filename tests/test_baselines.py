from datetime import datetime
import unittest

from newsrec.baselines import ItemCFIndex, PopularityIndex
from newsrec.data import Candidate, Impression


class BaselineTests(unittest.TestCase):
    def test_baselines_use_training_clicks_and_exclude_seen(self):
        train = [
            Impression("I1", "U1", datetime(2020, 1, 1), ("N1",), (Candidate("N2", 1), Candidate("N3", 0))),
            Impression("I2", "U2", datetime(2020, 1, 2), ("N1",), (Candidate("N2", 1), Candidate("N4", 0))),
        ]
        self.assertEqual([x.news_id for x in PopularityIndex().fit(train).search(["N1"], 5)], ["N2"])
        itemcf = ItemCFIndex().fit(train)
        self.assertEqual([x.news_id for x in itemcf.search(["N1"], 5)], ["N2"])
        self.assertEqual(itemcf.search(["N1", "N2"], 5), [])


if __name__ == "__main__":
    unittest.main()
