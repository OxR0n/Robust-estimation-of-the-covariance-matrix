import unittest

import numpy as np

from gaussian_evaluation import evaluate_estimate
from simulation_gaussian import simulate_gaussian


class GaussianSimulationTests(unittest.TestCase):
    def test_simulation_is_reproducible_and_matches_shared_interface(self):
        first = simulate_gaussian(T=100, D=4, seed=7)
        second = simulate_gaussian(T=100, D=4, seed=7)

        self.assertEqual(first["X"].shape, (100, 4))
        self.assertEqual(first["Sigma_true"].shape, (4, 4))
        np.testing.assert_array_equal(first["X"], second["X"])
        np.testing.assert_array_equal(first["Sigma_true"], second["Sigma_true"])
        self.assertGreater(np.linalg.eigvalsh(first["Sigma_true"]).min(), 0.0)

    def test_tuple_output_and_invalid_volatility(self):
        X, covariance = simulate_gaussian(T=5, D=2, output="tuple", seed=1)
        self.assertEqual(X.shape, (5, 2))
        self.assertEqual(covariance.shape, (2, 2))

        with self.assertRaises(ValueError):
            simulate_gaussian(T=5, D=2, volatility=0.0)


class GaussianEvaluationTests(unittest.TestCase):
    def test_true_covariance_has_zero_error_and_oracle_risk(self):
        sigma_true = simulate_gaussian(T=100, D=4, seed=11)["Sigma_true"]
        metrics = evaluate_estimate(sigma_true, sigma_true, w_max=0.5)

        self.assertAlmostEqual(metrics["frob_rel"], 0.0)
        self.assertAlmostEqual(metrics["spectral"], 0.0)
        self.assertAlmostEqual(metrics["corr_frob"], 0.0)
        self.assertAlmostEqual(metrics["mvp_risk_ratio_c"], 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
