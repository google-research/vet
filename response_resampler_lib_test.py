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
import pickle
from unittest import mock

from absl.testing import absltest
import numpy as np
import response_resampler_lib as resampler_lib
import parameterized_sample_lib as psample
import pandas as pd

class ResponseResamplerLibTest(absltest.TestCase):
  """Tests for class ExperimentManager and Experiment."""

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    # Create a response data file.
    self.out_dir = self.create_tempdir()
    self.data_filename = self.out_dir.create_file('output.pkl')
    datasets = psample.generate_response_tables(10, 5, 0.3, 2)
    with open(self.data_filename, 'wb') as f:
      pickle.dump(datasets, f)

    # Create the experiment config file.
    self.config_filename = self.out_dir.create_file(
        os.path.join(self.out_dir, 'config', 'config.csv')
    )

  def create_experiments(self, line: int = -1):
    # Create an experiment manager and an experiment.
    self.experiments_manager = resampler_lib.ExperimentsManager(
        path_name=self.out_dir.full_path,
        in_data_file=self.data_filename,
        use_pickle=True,
        line=line,
        config_file_name=self.config_filename,
        n_items=20,
        k_responses=5,
    )

    config_row = self.experiments_manager.e_grid.iloc[0]
    self.experiment = resampler_lib.Experiment(config_row)
    self.metrc_func = self.experiment.metric

  def write_config_file(self, metric_spec: str) -> None:
    """Writes the config file csv content for testing."""
    # A string template with metric name to be specified.
    config_lines = f"""
        ,Agg over Trials,# trials,Sampler,Shaper,Comparison(Metric),Agg_over_votes
        1,mean,1000,"(all_items,sample(5))",noop,"{metric_spec}",mean
      """
    with open(self.config_filename, 'w') as f:
      f.writelines(config_lines)

  def test_run_experiments(self):
    """Test no exception when running the experiments."""
    self.write_config_file(metric_spec='accuracy')
    self.create_experiments()
    self.experiments_manager.run_experiments()
    output_csv_file = os.path.join(
        self.experiments_manager.path_name,
        self.experiments_manager.out_file_name,
    )
    result_df = pd.read_csv(output_csv_file)
    self.assertTupleEqual((1, 37), result_df.shape)

  def test_run_single_row_experiment(self):
    """Test no exception when running a single row experiment."""
    self.write_config_file(metric_spec='accuracy')
    # Test running a single row experiment.
    self.create_experiments(line=0)
    self.experiments_manager.run_experiments()

    output_csv_file = os.path.join(
        self.experiments_manager.path_name,
        self.experiments_manager.out_file_name,
    )
    result_df = pd.read_csv(output_csv_file)
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
    self.write_config_file(metric_spec='recall(ht=0.5,mt1=0.6,mt2=0.7)')

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

if __name__ == '__main__':
  absltest.main()
