"""Copyright 2023 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for response_resampler_lib.
"""

import os
from unittest import mock

from absl.testing import absltest
import numpy as np
import datatypes
import response_resampler_lib as resampler_lib
import parameterized_sample_lib as psample
import pandas as pd

class ResponseResamplerLibTest(absltest.TestCase):
  """Tests for class ExperimentManager and Experiment."""

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    # Create a response data file.
    self.n_items = 10
    self.k_responses = 5
    self.out_dir = self.create_tempdir()
    self.datasets = psample.simulate_response_tables(10, 5, 0.3, 2)
    self.output_file_path = os.path.join(self.out_dir, 'output.csv')

    # Create the experiment config file.
    self.config_filename = self.out_dir.create_file(
        os.path.join(self.out_dir, 'config', 'config.csv')
    )

  def create_experiments(self, line: int = -1):
    # Create an experiment manager and an experiment.
    self.experiments_manager = resampler_lib.ExperimentsManager(
        self.datasets,
        self.config_filename,
        line,
        self.k_responses,
        self.output_file_path,
    )

    config_row = self.experiments_manager.e_grid.iloc[0]
    self.experiment = resampler_lib.Experiment(config_row, self.k_responses)
    self.metrc_func = self.experiment.metric

  def write_config_file(self, metric_spec: str) -> None:
    """Writes the config file csv content for testing."""
    # A string template with metric name to be specified.
    config_lines = f"""
        line,agg_over_trials,num_trials,sampler,shaper,comparison_metric,agg_over_responses
        1,mean,10,"(all_items,bootstrap_responses)",noop,"{metric_spec}",mean
      """
    with open(self.config_filename, 'w') as f:
      f.writelines(config_lines)

  def test_run_experiments(self):
    """Test no exception when running the experiments."""
    self.write_config_file(metric_spec='accuracy')
    self.create_experiments()
    self.experiments_manager.run_experiments()
    result_df = pd.read_csv(self.output_file_path)
    self.assertTupleEqual((1, 37), result_df.shape)

  def test_run_single_row_experiment(self):
    """Test no exception when running a single row experiment."""
    self.write_config_file(metric_spec='accuracy')
    # Test running a single row experiment.
    self.create_experiments(line=0)
    self.experiments_manager.run_experiments()

    result_df = pd.read_csv(self.output_file_path)
    self.assertTupleEqual((1, 37), result_df.shape)

  def test_parse_metric(self):
    self.write_config_file(metric_spec='accuracy')
    self.create_experiments()

    human_data, machine1_data, machine2_data = np.array(
        [[0.1, 0.6], [0, 0.1], [0.7, 0.7]]
    )
    # The default score thresholds for accuracy are all 0.5.
    results = self.metrc_func(human_data, machine1_data, machine2_data)
    self.assertAlmostEqual(results, (0.5, 0.5), 4)

  @mock.patch('machine_contest_metrics.recall')
  def test_parse_recall_with_thresholds(self, mock_recall):
    self.write_config_file(metric_spec='recall(ht=0.5, mt1=0.6, mt2=0.7)')

    self.create_experiments()
    human_data, machine1_data, machine2_data = np.array([[0.5], [0.6], [0.7]])
    self.metrc_func(human_data, machine1_data, machine2_data)
    mock_recall.assert_called_once_with(
        human_data, machine1_data, machine2_data, ht=0.5, mt1=0.6, mt2=0.7
    )

  @mock.patch('machine_contest_metrics.mean_of_emds')
  def test_parse_mean_of_emds(self, mock_mean_of_emds):
    self.write_config_file(metric_spec='mean_of_emds(bins=6)')

    self.create_experiments()
    human_data, machine1_data, machine2_data = np.array([[0.5], [0.6], [0.7]])

    self.metrc_func(human_data, machine1_data, machine2_data)
    mock_mean_of_emds.assert_called_once_with(
        human_data, machine1_data, machine2_data, bins=6
    )

  def test_bad_metric_spec_with_positional_args(self):
    # Positional args are not supported.
    self.write_config_file(metric_spec='mean_of_emds(6)')
    with self.assertRaises(ValueError):
      self.create_experiments()

  # Test that resetting to same seed value yields same sample.
  def test_samplers_seeding(self):
    item_samplers = resampler_lib.ItemSamplers(seed=19)
    response_samplers = resampler_lib.ResponseSamplers(seed=19)

    human_data, machine1_data, machine2_data = np.array(
        [[0.1, 0.6], [0, 0.1], [0.7, 0.7]])
    response_data = datatypes.ResponseData(
        human_data, machine1_data, machine2_data)

    item_sample = item_samplers.resample_items(response_data)
    item_samplers.reset_seed(seed=19)
    new_item_sample = item_samplers.resample_items(response_data)
    self.assertDictEqual(item_sample.to_dict(), new_item_sample.to_dict())

    human_data = np.ndarray(shape=[1, human_data.shape[0]], buffer=human_data)
    response_sample = response_samplers.resample_responses(
        n_items=human_data.shape[0], k_responses=human_data.shape[1],
        domain_size=human_data.shape[1], matrix=human_data)
    response_samplers.reset_seed(seed=19)
    new_response_sample = response_samplers.resample_responses(
        n_items=human_data.shape[0], k_responses=human_data.shape[1],
        domain_size=human_data.shape[1], matrix=human_data)
    self.assertListEqual(response_sample.flatten().tolist(),
                         new_response_sample.flatten().tolist())

  def test_item_samplers(self):
    item_samplers = resampler_lib.ItemSamplers(seed=19)
    human_data, machine1_data, machine2_data = np.array(
        [[0.1, 0.6], [0, 0.1], [0.7, 0.7]])
    response_data = datatypes.ResponseData(
        human_data, machine1_data, machine2_data)

    resample = item_samplers.resample_items(response_data)
    self.assertEqual(resample.gold.shape[0], human_data.shape[0])

  def test_response_sampler_output_shape(self):
    response_samplers = resampler_lib.ResponseSamplers(seed=19)
    data_array = np.arange(self.n_items * self.k_responses)
    data_matrix = np.reshape(data_array,
                             newshape=[self.n_items, self.k_responses])

    for response_sampler in [
        response_samplers.resample_responses,
        response_samplers.sample_responses,
        response_samplers.take_all_responses,
        response_samplers.sample_all_responses,
    ]:
      resample = response_sampler(
          n_items=data_matrix.shape[0], k_responses=data_matrix.shape[1],
          domain_size=data_matrix.shape[1], matrix=data_matrix)
      self.assertEqual(resample.shape, data_matrix.shape)

    for response_sampler in [
        response_samplers.take_first_response,
        response_samplers.sample_one_response,
    ]:
      resample = response_sampler(
          n_items=data_matrix.shape[0], k_responses=data_matrix.shape[1],
          domain_size=data_matrix.shape[1], matrix=data_matrix)
      self.assertEqual(resample.shape, (data_matrix.shape[0], 1))

  def test_response_sampling_without_replacement(self):
    response_samplers = resampler_lib.ResponseSamplers(seed=19)
    data_array = np.arange(self.n_items * self.k_responses)
    data_matrix = np.reshape(data_array,
                             newshape=[self.n_items, self.k_responses])

    for response_sampler in [
        response_samplers.sample_responses,
        response_samplers.take_all_responses,
        response_samplers.sample_all_responses,
    ]:
      output = response_sampler(
          n_items=data_matrix.shape[0], k_responses=data_matrix.shape[1],
          domain_size=data_matrix.shape[1], matrix=data_matrix)
      sorted_output = np.sort(output, axis=1)
      self.assertTrue(np.array_equal(sorted_output, data_matrix))

if __name__ == '__main__':
  absltest.main()
