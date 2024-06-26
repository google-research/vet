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
from absl.testing import parameterized
import numpy as np
import datatypes
import machine_contest_metrics

def format_data(
    input_data_list: list[list[list[float]]],
) -> list[datatypes.ResponseData]:
  """Format data in our data classes.

  Args:
    input_data_list: a list response data. Each response data is in the format
      of [gold_array, machine1_array, machine2_array].

  Returns:
    A list of ResponseData.
  """
  output = []
  for gold, machine1, machine2 in input_data_list:
    response_data = datatypes.ResponseData(
        np.asarray(gold), np.asarray(machine1), np.asarray(machine2)
    )
    output.append(response_data)
  return output

class MachineContestMetricsTest(absltest.TestCase):

  def metric_helper(
      self,
      metric: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
      expected_results: list[list[float]],
      response_data_list: list[datatypes.ResponseData],
      metric_params: dict[str, Any] | None = None,
  ) -> None:
    """Runs tests on a given metric.

    Args:
      metric: The metric to test.
      expected_results: A list of expected results for the metric.
      response_data_list: A list of input data 3-tuples.
      metric_params: The params needed by the metric func.
    """

    if metric_params is None:
      metric_params = {}
    for response_data, expected_results in zip(
        response_data_list, expected_results
    ):
      results = metric(
          response_data.gold,
          response_data.preds1,
          response_data.preds2,
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
    linear_responses = format_data([
        [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        [[0, 0], [0, 0], [0, 0]],
        [[1, 1], [1, 1], [1, 1]],
    ])
    expected_linear_results = [[0.00900, 0.04297], [0, 0], [0, 0]]
    self.metric_helper(
        machine_contest_metrics.cos_distance,
        expected_linear_results,
        linear_responses,
    )

  def test_emd_aggregated(self):
    linear_responses = format_data([
        [[0.1, 0.1], [0.2, 0.8], [0.3, 0.7]],
        [[0, 0.1], [1.0, 1.0], [0.5, 0.5]],
        [[1, 0], [0, 0], [1, 1]],
    ])
    expected_linear_results = [[0.4, 0.4], [0.0, 0.0], [0.4, 0.4]]
    self.metric_helper(
        lambda x, y, z: machine_contest_metrics.emd_aggregated(x, y, z, 5),
        expected_linear_results,
        linear_responses,
    )

  def test_mean_of_emds(self):
    linear_responses = format_data([
        [[0.1, 0.1], [0.2, 0.8], [0.3, 0.7]],
        [[0, 0.1], [1.0, 1.0], [0.5, 0.5]],
        [[1, 0], [0, 0], [1, 1]],
    ])
    expected_linear_results = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    self.metric_helper(
        lambda x, y, z: machine_contest_metrics.mean_of_emds(x, y, z, 5),
        expected_linear_results,
        linear_responses,
    )

  def test_mean_relative_entropy(self):
    linear_responses = format_data([
        [[[0.1, 0.2, 0.2, 0.3, 0.3, 0.3]],
         [[0.1, 0.2, 0.2, 0.3, 0.3, 0.3]],
         [[0.1, 0.1, 0.2, 0.3, 0.3, 0.3]]],
        [[[0.1, 0.2, 0.2, 0.3, 0.3, 0.3]],
         [[0.1, 0.1, 0.2, 0.2, 0.3, 0.3]],
         [[0.1, 0.2, 0.2, 0.3, 0.3, 0.3]]],
    ])
    expected_linear_results = [[0.0, 0.16666], [0.13835, 0.0]]
    self.metric_helper(
        machine_contest_metrics.mean_relative_entropy,
        expected_linear_results,
        linear_responses,
    )

  def test_mean(self):
    linear_responses = format_data([
        [[0.1, 0.1], [0.2, 0.8], [0.3, 0.7]],
        [[0, 0.1], [1.0, 1.0], [0.5, 0.5]],
        [[1.0, 0], [0, 0], [1.0, 1.0]],
    ])
    expected_linear_results = [[0.5, 0.5], [1.0, 0.5], [0.0, 1.0]]
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
    linear_responses = format_data([
        [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        [[0, 0], [0, 0], [0, 0]],
        [[1, 1], [1, 1], [1, 1]],
    ])
    expected_linear_results = [
        [0.1, 0.2],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    self.metric_helper(
        machine_contest_metrics.root_mean_squared_error,
        expected_linear_results,
        linear_responses,
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
    linear_responses = format_data([
        [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        [[0, 0], [0, 0], [0, 0]],
        [[1, 1], [1, 1], [1, 1]],
    ])
    expected_linear_results = [[2, 0], [0, 0], [0, 0]]
    self.metric_helper(
        machine_contest_metrics.wins_mae,
        expected_linear_results,
        linear_responses,
    )

  def test_mean_absolute_error(self):
    linear_responses = format_data([
        [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        [[0, 0], [0, 0], [0, 0]],
        [[1, 1], [1, 1], [1, 1]],
    ])
    expected_linear_results = [[0.1, 0.2], [0, 0], [0, 0]]
    self.metric_helper(
        machine_contest_metrics.mean_absolute_error,
        expected_linear_results,
        linear_responses,
    )

  def test_max_absolute_error(self):
    linear_responses = format_data([
        [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        [[0, 0], [0, 0], [0, 0]],
        [[1, 1], [1, 1], [1, 1]],
    ])
    expected_linear_results = [[0.1, 0.2], [0, 0], [0, 0]]
    self.metric_helper(
        machine_contest_metrics.max_absolute_error,
        expected_linear_results,
        linear_responses,
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
    linear_responses = format_data([
        [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]],
        [[0.1, 0.9], [0.2, 0.95], [0.7, 0.6]],
    ])
    expected_linear_results = [[1, 1], [1, 1], [1, -1]]
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
    """Test the case with perfect balance between `s_null` and `s_alt`."""
    s_null = np.random.permutation(10)  # order doesn't matter
    s_alt = np.repeat(4.5, 10)
    p_value = machine_contest_metrics.calculate_p_value(s_null, s_alt)
    self.assertAlmostEqual(p_value, 0.5, places=2)

  def test_calculate_p_value_null_all_greater(self):
    """Test the case where `s_alt` is no better than `s_null` for all scores."""
    s_null = np.arange(1, 11)
    s_alt = np.zeros_like(s_null)
    p_value = machine_contest_metrics.calculate_p_value(
        s_null, s_alt, two_sided_test=False)
    self.assertAlmostEqual(p_value, 1.0, places=2)

  def test_calculate_p_value_alt_all_greater(self):
    """Test the case where `s_alt` is always better than `s_null`."""
    s_null = np.repeat(-1, 10)
    s_alt = np.random.permutation(10)
    p_value = machine_contest_metrics.calculate_p_value(
        s_null, s_alt, two_sided_test=False)
    self.assertAlmostEqual(p_value, 0.0, places=2)

  def test_calculate_p_value_same_items(self):
    """Test when `s_null` and `s_alt` have the same set of (unequal) scores."""
    s_null = np.random.permutation(10)
    s_alt = np.random.permutation(10)
    p_value = machine_contest_metrics.calculate_p_value(s_null, s_alt)
    self.assertAlmostEqual(p_value, 0.55, places=2)

  def test_calculate_p_value_same_items_all_equal(self):
    """Test when `s_null` and `s_alt` have equal-valued items."""
    s_null = np.repeat(0.5, 10)
    s_alt = np.repeat(0.5, 10)
    p_value = machine_contest_metrics.calculate_p_value(s_null, s_alt)
    self.assertAlmostEqual(p_value, 1.0, places=2)

  def test_calculate_p_value_with_ties(self):
    s_null = np.concatenate([np.zeros(50), np.ones(50)])
    s_alt = np.concatenate([np.zeros(60), np.ones(40)])
    p_value1 = machine_contest_metrics.calculate_p_value(
        s_null, s_alt, two_sided_test=False)
    p_value2 = machine_contest_metrics.calculate_p_value(
        s_null, s_alt, two_sided_test=True)
    self.assertAlmostEqual(p_value1, 0.8, places=2)
    self.assertAlmostEqual(p_value2, 0.7, places=2)

  def test_calculate_p_value_null_outlier(self):
    s_null = np.asarray([100, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    s_alt = np.repeat(10, 10)
    p_value = machine_contest_metrics.calculate_p_value(s_null, s_alt)
    self.assertAlmostEqual(p_value, 0.1, places=2)

  def test_calculate_p_value_alt_outlier(self):
    s_null = np.repeat(10, 10)
    s_alt = np.asarray([100, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    p_value1 = machine_contest_metrics.calculate_p_value(
        s_null, s_alt, two_sided_test=False)
    p_value2 = machine_contest_metrics.calculate_p_value(
        s_null, s_alt, two_sided_test=True)
    self.assertAlmostEqual(p_value1, 0.9, places=2)
    self.assertAlmostEqual(p_value2, 0.1, places=2)

  def test_mean_confidence_bounds(self):
    scores = np.arange(1000)
    (lower_quantile, mean, upper_quantile) = (
        machine_contest_metrics.mean_and_confidence_bounds(scores)
    )
    self.assertAlmostEqual(mean, 999.0 / 2.0, places=2)
    self.assertEqual(lower_quantile, 25)
    self.assertEqual(upper_quantile, 975)

class MetricsTests(parameterized.TestCase):
  @parameterized.named_parameters(
      dict(testcase_name='mean_relative_entropy:machine1_zero_entropy',
           fn=machine_contest_metrics.mean_relative_entropy,
           gold=[[0.1, 0.2, 0.2, 0.3, 0.3, 0.3]],
           machine1=[[0.1, 0.2, 0.2, 0.3, 0.3, 0.3]],
           machine2=[[0.1, 0.1, 0.2, 0.3, 0.3, 0.3]],
           expected=[0.0, 0.16666]),
      dict(testcase_name='mean_relative_entropy:machine2_zero_entropy',
           fn=machine_contest_metrics.mean_relative_entropy,
           gold=[[0.1, 0.2, 0.2, 0.3, 0.3, 0.3]],
           machine1=[[0.1, 0.1, 0.2, 0.2, 0.3, 0.3]],
           machine2=[[0.1, 0.2, 0.2, 0.3, 0.3, 0.3]],
           expected=[0.13835, 0.0]),
  )
  def test_metric_function(
      self,
      fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
      gold: list[list[float]],
      machine1: list[list[float]],
      machine2: list[list[float]],
      expected: list[float],
  ):
    result = fn(np.asarray(gold), np.asarray(machine1), np.asarray(machine2))
    self.assertTrue(np.allclose(result, expected, atol=10e-04))

if __name__ == '__main__':
  absltest.main()
