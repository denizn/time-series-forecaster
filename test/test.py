# Sample unit test to validate non-negativity of output of forecasts.csv

import unittest
import pandas as pd
import numpy as np

from config.config import get_config


class TestForecastNonNegativity(unittest.TestCase):
    def runTest(self):

        yhat = pd.read_csv("models/results/forecasts.csv")["yhat"]
        self.assertEqual(np.all(yhat >= 0), 1, "Not all forecasts are positive")


if __name__ == "__main__":
    unittest.main()
