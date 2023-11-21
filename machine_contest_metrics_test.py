"""Copyright 2022 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests machine_contest_metrics.
"""

from typing import Any, Callable

from absl.testing import absltest
import numpy as np
import machine_contest_metrics

def format_data(threedlist):
  """Format data in our peculiar format.

  Args:
    threedlist: a list with at least three levels of nesting.

  Returns:
    A list with three levels of nesting, and the remaining levels are
    a numpy array.
  """
  return [np.asarray(t) for t in threedlist]

class MachineContestMetricsTest(absltest.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()

    # Create a few simple datasets.
    self.linear_responses = [
        [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        [[0, 0], [0, 0], [0, 0]],
        [[1, 1], [1, 1], [1, 1]],
    ]
    self.linear_responses = format_data(self.linear_responses)

    self.twod_responses = [
        [
            [[0.1, 0.9], [0.1, 0.8]],
            [[0.2, 0.8], [0.1, 0.7]],
            [[0.3, 0.7], [0.2, 0.6]],
        ],
        [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        [[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]],
    ]
    self.twod_responses = format_data(self.twod_responses)

  def metric_helper(
      self,
      metric: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
      expected_results: list[list[float]],
      response_data: list[list[list[Any]]],
      metric_params: dict[str, Any] | None = None,
  ) -> None:
    """Runs tests on a given metric.

    Args:
      metric: The metric to test.
      expected_results: A list of expected results for the metric.
      response_data: A list of input data 3-tuples.
      metric_params: The params needed by the metric func.
    """

    if metric_params is None:
      metric_params = {}
    for (human, machine1, machine2), expected_results in zip(
        response_data, expected_results
    ):
      results = metric(
          np.array(human),
          np.array(machine1),
          np.array(machine2),
          **metric_params
      )
      for result, expected in zip(results, expected_results):
        self.assertAlmostEqual(result, expected, 4)

  def test_accuracy(self):
    linear_responses = format_data([
        [[0.1, 0.1], [0.2, 0.8], [0.3, 0.7]],
        [[0, 0.1], [1.0, 1.0], [0.6, 0.5]],
        [[1, 0], [1, 0], [0, 1]],
    ])

    expected_linear_results = [[0.5, 0.5], [0, 0], [1, 0]]
    self.metric_helper(
        machine_contest_metrics.accuracy,
        expected_linear_results,
        linear_responses,
    )

  def test_accuracy_with_thresholds(self):
    linear_responses = format_data([
        [[0.1, 0.1], [0.2, 0.8], [0.3, 0.7]],
        [[0, 0.1], [1.0, 1.0], [0.6, 0.5]],
        [[1, 0], [1, 0], [0, 1]],
    ])

    expected_linear_results = [[1.0, 0.0], [0, 0], [1, 0]]
    self.metric_helper(
        machine_contest_metrics.accuracy,
        expected_linear_results,
        linear_responses,
        {'ht': 0.5, 'mt1': 0.9, 'mt2': 0.3},
    )

  def test_auc(self):
    linear_responses = format_data([
        [[0.6, 0.6, 0.3], [0.8, 0.4, 0.8], [0.3, 0.3, 0.8]],
        [[0, 1], [0.1, 1], [0.6, 0.5]],
        [[1, 0], [1, 0], [0, 1]],
    ])

    expected_linear_results = [[0.25, 0.0], [1.0, 0.5], [1.0, 0.0]]
    self.metric_helper(
        machine_contest_metrics.auc,
        expected_linear_results,
        linear_responses,
        {'ht': 0.5, 'mt1': 0.5, 'mt2': 0.5},
    )

  def test_cos_distance(self):
    expected_linear_results = [[0.00900, 0.04297], [0, 0], [0, 0]]
    self.metric_helper(
        machine_contest_metrics.cos_distance,
        expected_linear_results,
        self.linear_responses,
    )

  def test_emd_aggregated(self):
    expected_linear_results = [[0.4, 0.4], [0.0, 0.0], [0.4, 0.4]]

    linear_responses = format_data([
        [[0.1, 0.1], [0.2, 0.8], [0.3, 0.7]],
        [[0, 0.1], [1.0, 1.0], [0.5, 0.5]],
        [[1, 0], [0, 0], [1, 1]],
    ])
    self.metric_helper(
        lambda x, y, z: machine_contest_metrics.emd_aggregated(x, y, z, 5),
        expected_linear_results,
        linear_responses,
    )

  def test_mean_of_emds(self):
    expected_linear_results = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    linear_responses = format_data([
        [[0.1, 0.1], [0.2, 0.8], [0.3, 0.7]],
        [[0, 0.1], [1.0, 1.0], [0.5, 0.5]],
        [[1, 0], [0, 0], [1, 1]],
    ])
    self.metric_helper(
        lambda x, y, z: machine_contest_metrics.mean_of_emds(x, y, z, 5),
        expected_linear_results,
        linear_responses,
    )

  def test_mean(self):
    expected_linear_results = [[0.5, 0.5], [1.0, 0.5], [0.0, 1.0]]

    linear_responses = format_data([
        [[0.1, 0.1], [0.2, 0.8], [0.3, 0.7]],
        [[0, 0.1], [1.0, 1.0], [0.5, 0.5]],
        [[1.0, 0], [0, 0], [1.0, 1.0]],
    ])
    self.metric_helper(
        machine_contest_metrics.mean,
        expected_linear_results,
        linear_responses,
    )

  def test_f1_score(self):
    linear_responses = format_data([
        [[0.6, 0.6, 0.3], [0.8, 0.4, 0.8], [0.3, 0.3, 0.8]],
        [[0, 1], [0.1, 1], [0.6, 0.5]],
        [[1, 0], [1, 0], [0, 1]],
    ])
    expected_linear_results = [[0.5, 0], [1, 0.6666666], [1, 0]]
    self.metric_helper(
        machine_contest_metrics.f1_score,
        expected_linear_results,
        linear_responses,
        {'ht': 0.5, 'mt1': 0.5, 'mt2': 0.5},
    )

  def test_root_mean_squared_error(self):
    expected_linear_results = [
        [0.1, 0.2],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    self.metric_helper(
        machine_contest_metrics.root_mean_squared_error,
        expected_linear_results,
        self.linear_responses,
    )

  def test_precision(self):
    linear_responses = format_data([
        [[0.6, 0.6, 0.3], [0.8, 0.4, 0.8], [0.3, 0.3, 0.8]],
        [[0, 1], [0.1, 1], [0.6, 0.5]],
        [[1, 0], [1, 0], [0, 1]],
    ])

    expected_linear_results = [[0.5, 0], [1, 0.5], [1, 0]]
    self.metric_helper(
        machine_contest_metrics.precision,
        expected_linear_results,
        linear_responses,
        {'ht': 0.5, 'mt1': 0.5, 'mt2': 0.5},
    )

  def test_wins_mae(self):
    expected_linear_results = [[2, 0], [0, 0], [0, 0]]
    self.metric_helper(
        machine_contest_metrics.wins_mae,
        expected_linear_results,
        self.linear_responses,
    )

  def test_mean_absolute_error(self):
    expected_linear_results = [[0.1, 0.2], [0, 0], [0, 0]]
    self.metric_helper(
        machine_contest_metrics.mean_absolute_error,
        expected_linear_results,
        self.linear_responses,
    )

  def test_max_absolute_error(self):
    expected_linear_results = [[0.1, 0.2], [0, 0], [0, 0]]
    self.metric_helper(
        machine_contest_metrics.max_absolute_error,
        expected_linear_results,
        self.linear_responses,
    )

  def test_kld(self):
    """Test the case where at least one of the inputs is nonpositive."""
    expected = 46.06817
    self.assertAlmostEqual(machine_contest_metrics.kld(0.2, -0.3), expected, 5)

  def test_kld_of_means(self):
    linear_responses = format_data([
        [[0.1, 0.9], [0.5, 0.8], [0.3, 0.7]],
        [[0.1, 0.9], [0.2, 0.9], [0.1, 0.9]],
        [[0.1, 0.9], [0.2, 0.95], [0.7, 0.6]],
    ])

    expected_results = [[0.03763, 0.0], [0.00468, 0.0], [0.01023, 0.03763]]
    self.metric_helper(
        machine_contest_metrics.kld_of_means, expected_results, linear_responses
    )

  def test_mean_kld(self):
    linear_responses = format_data([
        [[0.1, 0.9], [0.5, 0.8], [0.3, 0.7]],
        [[0.1, 0.9], [0.2, 0.9], [0.1, 0.9]],
        [[0.1, 0.9], [0.2, 0.95], [0.7, 0.6]],
    ])

    expected_results = [[1.19861, 0.46523], [0.15342, 0.0], [0.15417, 2.06311]]

    self.metric_helper(
        machine_contest_metrics.mean_kld, expected_results, linear_responses
    )

  def test_recall(self):
    linear_responses = format_data([
        [[0.6, 0.6, 0.3], [0.8, 0.4, 0.8], [0.3, 0.3, 0.8]],
        [[0, 1], [0.1, 1], [0.6, 0.5]],
        [[1, 0], [1, 0], [0, 1]],
    ])

    expected_linear_results = [[0.5, 0], [1, 1], [1, 0]]
    self.metric_helper(
        machine_contest_metrics.recall,
        expected_linear_results,
        linear_responses,
        {'ht': 0.5, 'mt1': 0.5, 'mt2': 0.5},
    )

  def test_spearmanr(self):
    expected_linear_results = [[1, 1], [1, 1], [1, -1]]
    linear_responses = format_data([
        [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]],
        [[0.1, 0.9], [0.2, 0.95], [0.7, 0.6]],
    ])
    self.metric_helper(
        machine_contest_metrics.spearmanr,
        expected_linear_results,
        linear_responses,
    )

  def test_higher_wins(self):
    machine1 = np.arange(10)
    machine2 = np.ones(10) * 5
    results = machine_contest_metrics.higher_wins(machine1, machine2)
    self.assertTupleEqual(results, (4, 5))

  def test_lower_wins(self):
    machine1 = np.arange(10)
    machine2 = np.ones(10) * 5
    results = machine_contest_metrics.lower_wins(machine1, machine2)
    self.assertTupleEqual(results, (5, 4))

  def test_calculate_p_value_balanced(self):
    """Test the case with perfect balanced between `s_null` and `s_alt`."""
    s_null = np.random.permutation(10)  # order doesn't matter
    s_alt = np.repeat(4.5, 10)
    p_value = machine_contest_metrics.calculate_p_value(s_null, s_alt)
    self.assertAlmostEqual(p_value, 0.5, places=2)

  def test_calculate_p_value_null_all_greater(self):
    """Test the case where `s_alt` is no better than `s_null` for all scores."""
    s_null = np.arange(10)
    s_alt = np.zeros_like(s_null)
    p_value = machine_contest_metrics.calculate_p_value(s_null, s_alt)
    self.assertAlmostEqual(p_value, 0.0, places=2)

  def test_calculate_p_value_alt_all_greater(self):
    """Test the case where `s_alt` is always better than `s_null`."""
    s_null = np.repeat(-1, 10)
    s_alt = np.random.permutation(10)
    p_value = machine_contest_metrics.calculate_p_value(s_null, s_alt)
    self.assertAlmostEqual(p_value, 0.9, places=2)

  def test_calculate_p_value_same_items(self):
    """Test when `s_null` and `s_alt` have the same items."""
    s_null = np.random.permutation(10)
    s_alt = np.random.permutation(10)
    p_value = machine_contest_metrics.calculate_p_value(s_null, s_alt)
    self.assertAlmostEqual(p_value, 0.45, places=2)

  def test_mean_confidence_bounds(self):
    scores = np.arange(1000)
    (lower_quantile, mean, upper_quantile) = (
        machine_contest_metrics.mean_and_confidence_bounds(scores)
    )
    self.assertAlmostEqual(mean, 999.0 / 2.0, places=2)
    self.assertEqual(lower_quantile, 25)
    self.assertEqual(upper_quantile, 975)

if __name__ == '__main__':
  absltest.main()
