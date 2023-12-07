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

Resampling library for the P-Value Experimentation Suite.

Assesses various p-value estimators for annotators and stochastic machines
based on bootstrapping.

The responses for each item are drawn from a unique, characteristic distribution
with 1-2 shaping params - each item has a different distribution. This is meant
to capture, e.g., that some items have high agreement and some have high
disagreement. To generate the random responses, we first randomly generate the
shaping params for each item (N x 2 shape params), and then draw (k) random
responses from that distribution.

For example, we could say that the individual responses per item are normally
distributed, and therefore each item has a mean and standard deviation. We
first generate a distribution of means and stdevs (one pair per item), then
for each item we sample k responses from the distribution with that mean, stdev.
"""

import collections
import datetime
import enum
import functools
import json
import os
import pickle
import random as rand
import re
from typing import Any, Callable, Mapping, Tuple

from absl import logging
import numpy as np
import datatypes
import machine_contest_metrics as mcm
import pandas as pd

class TypesEnumBase(enum.Enum):
  """A base enum class that can return the list of enum values."""

  @classmethod
  def value_list(cls):
    return [c.value for c in cls]

class ParameterTypes(TypesEnumBase):
  """Lists of parameters and stats we keep track of in experiments."""

  AGG_OVER_TRIALS = "agg_over_trials"
  NUM_TRIALS = "num_trials"
  SAMPLER = "sampler"
  GT_SAMPLER = "GT_sampler"
  SHAPER = "shaper"
  COMPARISON_METRIC = "comparison_metric"
  AGG_OVER_RESPONSES = "agg_over_responses"

class TestStatTypes(TypesEnumBase):
  """Types for test stats."""

  M1_SCORE = "M1_Score"
  M1_SCORE_STD = "M1_Score_Std"
  M2_SCORE = "M2_Score"
  M2_SCORE_STD = "M2_Score_Std"
  M1_TRIAL_WINS = "M1_Trial_Wins"
  M2_TRIAL_WINS = "M2_Trial_Wins"
  EST_P_SCORE = "Est_Pvalue"
  ALT_SCORE_DIFFS_TEST_MEAN = "Alt_Score_Diffs_Test_Mean"
  NULL_SCORE_DIFFS_TEST_MEAN = "Null_Score_Diffs_Test_Mean"
  ALT_SCORE_DIFFS_TEST_STD = "Alt_Score_Diffs_Test_Std"
  NULL_SCORE_DIFFS_TEST_STD = "Null_Score_Diffs_Test_Std"

class GroundStatTypes(TypesEnumBase):
  """Types for ground stats."""

  M1_GT_NULL = "M1 GT Null"
  M1_GT_NULL_STD = "M1 GT Null Std"
  M2_GT_NULL = "M2 GT Null"
  M2_GT_NULL_STD = "M2 GT Null Std"
  M1_GT_ALT = "M1 GT Alt"
  M1_GT_ALT_STD = "M1 GT Alt Std"
  M2_GT_ALT = "M2 GT Alt"
  M2_GT_ALT_STD = "M2 GT Alt Std"
  ALT_SCORE_DIFFS_GT = "Alt Score Diffs GT"
  ALT_SCORE_DIFFS_GT_STD = "Alt Score Diffs GT Std"
  NULL_SCORE_DIFFS_GT = "Null Score Diffs GT"
  NULL_SCORE_DIFFS_GT_STD = "Null Score Diffs GT Std"

  @classmethod
  def basic_stat_values(cls):
    return [c.value for c in cls if not c.name.endswith("STD")]

class AggStatTypes(TypesEnumBase):
  """Types for aggregated stats."""

  GT_P_SCORE = "GT_Pvalue"
  GT_M1_ALT_WINS = "GT_M1_Alt_Wins"
  GT_M2_ALT_WINS = "GT_M2_Alt_Wins"
  GT_M1_NULL_WINS = "GT_M1_Null_Wins"
  GT_M2_NULL_WINS = "GT_M2_Null_Wins"

main_stats = TestStatTypes.value_list() + GroundStatTypes.basic_stat_values()

def noop(x: Any) -> Any:
  """Return input.

  It is is a useful placeholder for various samplers, shapers, and aggregators
  that perform no operation on the input.

  Args:
    x: The input variable.

  Returns:
    x
  """
  return x

##############################################
# Shaper functions

def vectorize(matrix: np.ndarray[Any, np.dtype]) -> np.ndarray[int, np.dtype]:
  """Return a flattened matrix."""
  return matrix.flatten()

##############################################
# Sampler functions: for responses

def first_response(responses: np.ndarray) -> np.ndarray:
  """Returns the first item of a vector or list as a one-element list."""
  return np.array([responses[0]])

def sample_responses(k_responses: int) -> Callable[[np.ndarray], np.ndarray]:
  """Bootstrap sample from a set of responses."""
  func = lambda x: (rand.choices(x, k=k_responses))
  return func

def sample_all(responses: np.ndarray) -> np.ndarray:
  """Bootstrap sample from all responses per item."""
  return np.array(rand.sample(responses, k=len(responses)))

def sample_ground_responses(
    k_responses: int,
) -> Callable[[np.ndarray], np.ndarray]:
  """Sample without replacement from ground truth data."""
  func = lambda x: (rand.choices(list(x), k=k_responses))
  return func

################################################
# item_samplers

def all_items(response_data: datatypes.ResponseData) -> datatypes.ResponseData:
  """Take all items from each dataset."""
  return response_data

def bootstrap_items(
    response_data: datatypes.ResponseData,
) -> datatypes.ResponseData:
  """Bootstrap sample from the items in each dataset."""
  human = response_data.gold
  indices = rand.choices(range(len(human)), k=len(human))
  human_scores_t = np.take(human, indices, axis=0)
  machine1_scores_t = np.take(response_data.preds1, indices, axis=0)
  machine2_scores_t = np.take(response_data.preds2, indices, axis=0)
  return datatypes.ResponseData(
      human_scores_t, machine1_scores_t, machine2_scores_t
  )

def resample_items_and_responses(
    response_data: datatypes.ResponseData,
    item_sampler: Callable[[datatypes.ResponseData], datatypes.ResponseData],
    response_sampler: Callable[[np.ndarray], np.ndarray],
) -> datatypes.ResponseData:
  """Construct a 2D bootstrap sample across three datasets.

  Args:
    response_data: Contains responses for human, machine1 and machine2.
    item_sampler: The method for sampling items
    response_sampler: The method for sampling responses.

  Returns:
    A sample of the inputs, according to the provided methods,
    where the sampled items across all three datasets are the same.
  """
  sampled_response_data = item_sampler(response_data)
  human = sampled_response_data.gold
  machine1 = sampled_response_data.preds1
  machine2 = sampled_response_data.preds2
  if len(machine1[0]) > len(human[0]):
    machine1 = [sample_responses(len(human[0]))(x) for x in machine1]
    machine2 = [sample_responses(len(human[0]))(x) for x in machine2]

  human = [response_sampler(x) for x in human]
  machine1 = [response_sampler(x) for x in machine1]
  machine2 = [response_sampler(x) for x in machine2]

  return datatypes.ResponseData(
      np.array(human), np.array(machine1), np.array(machine2)
  )

def resample_items_and_responses_factory(
    item_sampler: Callable[[datatypes.ResponseData], datatypes.ResponseData],
    response_sampler: Callable[[np.ndarray], np.ndarray],
) -> Callable[[datatypes.ResponseData], datatypes.ResponseData]:
  """Construct a 2D bootstrap sampler across three datasets.

  Args:
    item_sampler: The method for sampling items.
    response_sampler: The method for sampling responses.

  Returns:
    A curried instance of resample_items_and_responses, with the item_sampler
    and response_sampler arguments fixed.
  """
  # pylint: disable=g-long-lambda
  return lambda response_data: resample_items_and_responses(
      response_data, item_sampler, response_sampler
  )

class Experiment:
  """Manages a single experiment."""

  def __init__(self, config_row: dict[str, Any], k_responses: int):
    """Initializer for Experiment class.

    Args:
      config_row: a DataFrame row of parameters from the experiments
        configuration file.
      k_responses: Number of responses per item. Must be no greater than number
        of responses per items in the input_data dataset.
    """
    self.k_responses = k_responses
    exp_config = self.setup_experiment(config_row)
    self.metric = exp_config[ParameterTypes.COMPARISON_METRIC.value]
    self.item_level_aggregator = exp_config[
        ParameterTypes.AGG_OVER_RESPONSES.value
    ]
    self.num_trials = exp_config[ParameterTypes.NUM_TRIALS.value]
    self.sampler = exp_config[ParameterTypes.SAMPLER.value]
    self.ground_sampler = exp_config[ParameterTypes.GT_SAMPLER.value]
    self.shaper = exp_config[ParameterTypes.SHAPER.value]
    self.trial_aggregator = exp_config[ParameterTypes.AGG_OVER_TRIALS.value]

  def run_trial(
      self,
      response_data: datatypes.ResponseData,
      sampler: Callable[[datatypes.ResponseData], np.ndarray],
  ) -> Tuple[float, float]:
    """Count the item wins for each system.

      I.e., whichever system is closer to the gold value.

    Args:
      response_data: A ResponseData that contains gold, machine1 and machine2
        responses.
      sampler: Determines how items from the args above are sampled.

    Returns:
      A set of pairs for every horizontal metric with wins/ties/losses
    """
    sampled_response_data = sampler(response_data)
    human_scores_t = sampled_response_data.gold
    machine1_scores_t = sampled_response_data.preds1
    machine2_scores_t = sampled_response_data.preds2
    human_scores_t = np.array(
        [self.item_level_aggregator(x) for x in self.shaper(human_scores_t)]
    )
    machine1_scores_t = np.array(
        [self.item_level_aggregator(x) for x in self.shaper(machine1_scores_t)]
    )
    machine2_scores_t = np.array(
        [self.item_level_aggregator(x) for x in self.shaper(machine2_scores_t)]
    )

    return self.metric(human_scores_t, machine1_scores_t, machine2_scores_t)

  def _get_ground_trial_results(
      self,
      alt_sample: datatypes.ResponseData,
      null_sample: datatypes.ResponseData,
  ):
    """Get the results of a ground truth trial.

    Args:
      alt_sample: A ResponseData that contains responses for the alternative
        hypothesis.
      null_sample: A ResponseData that contains responses for the null
        hypothesis.
    """

    null_machine1_ground, null_machine2_ground = self.run_trial(
        null_sample,
        self.ground_sampler,
    )

    alt_machine1_ground, alt_machine2_ground = self.run_trial(
        alt_sample,
        self.ground_sampler,
    )

    self.sample_results[GroundStatTypes.M1_GT_NULL.value].append(
        null_machine1_ground
    )
    self.sample_results[GroundStatTypes.M2_GT_NULL.value].append(
        null_machine2_ground
    )
    self.sample_results[GroundStatTypes.M1_GT_ALT.value].append(
        alt_machine1_ground
    )
    self.sample_results[GroundStatTypes.M2_GT_ALT.value].append(
        alt_machine2_ground
    )

  def _get_test_set_results(
      self,
      alt_sample: datatypes.ResponseData,
  ):
    """Run tests and add results to _sample_results.

    Args:
      alt_sample: A ResponseData that contains responses for the alternative
        hypothesis.
    """

    machine1_wins_per_trial, machine2_wins_per_trial = np.transpose([
        self.run_trial(alt_sample, self.sampler) for _ in range(self.num_trials)
    ])

    # Now construct a null hypothesis and test
    null_responses = np.concatenate(
        [alt_sample.preds1, alt_sample.preds2], axis=1
    )
    null_test = datatypes.ResponseData(
        alt_sample.gold, null_responses, null_responses
    )
    null1_score, null2_score = np.transpose([
        self.run_trial(null_test, self.sampler) for _ in range(self.num_trials)
    ])

    alt_test_diff = machine1_wins_per_trial - machine2_wins_per_trial
    null_test_diff = null1_score - null2_score
    if np.median(null_test_diff) > np.median(alt_test_diff):
      alt_test_diff = -alt_test_diff
      null_test_diff = -null_test_diff

    machine1_trial_wins, machine2_trial_wins = mcm.higher_wins(
        machine1_wins_per_trial, machine2_wins_per_trial
    )

    self.sample_results[TestStatTypes.M1_SCORE.value].append(
        self.trial_aggregator(machine1_wins_per_trial)
    )
    self.sample_results[TestStatTypes.M1_SCORE_STD.value].append(
        np.std(machine1_wins_per_trial)
    )
    self.sample_results[TestStatTypes.M2_SCORE.value].append(
        self.trial_aggregator(machine2_wins_per_trial)
    )
    self.sample_results[TestStatTypes.M2_SCORE_STD.value].append(
        np.std(machine2_wins_per_trial)
    )
    self.sample_results[TestStatTypes.M1_TRIAL_WINS.value].append(
        machine1_trial_wins
    )
    self.sample_results[TestStatTypes.M2_TRIAL_WINS.value].append(
        machine2_trial_wins
    )
    self.sample_results[TestStatTypes.EST_P_SCORE.value].append(
        mcm.calculate_p_value(null_test_diff, alt_test_diff)
    )
    self.sample_results[TestStatTypes.ALT_SCORE_DIFFS_TEST_MEAN.value].append(
        np.mean(alt_test_diff)
    )
    self.sample_results[TestStatTypes.NULL_SCORE_DIFFS_TEST_MEAN.value].append(
        np.mean(null_test_diff)
    )
    self.sample_results[TestStatTypes.ALT_SCORE_DIFFS_TEST_STD.value].append(
        np.std(alt_test_diff)
    )
    self.sample_results[TestStatTypes.NULL_SCORE_DIFFS_TEST_STD.value].append(
        np.std(null_test_diff)
    )

  def _aggregate_experiment_results(self) -> Mapping[str, float]:
    """Aggregate results from multiple trials and report summary statistics.

    Returns:
      Summary statistics of the results.
    """
    alt_ground_diff = np.array(
        self.sample_results[GroundStatTypes.M1_GT_ALT.value]
    ) - np.array(self.sample_results[GroundStatTypes.M2_GT_ALT.value])
    null_ground_diff = np.array(
        self.sample_results[GroundStatTypes.M1_GT_NULL.value]
    ) - np.array(self.sample_results[GroundStatTypes.M2_GT_NULL.value])

    if np.median(null_ground_diff) > np.median(alt_ground_diff):
      null_ground_diff = -null_ground_diff
      alt_ground_diff = -alt_ground_diff

    self.sample_results[GroundStatTypes.ALT_SCORE_DIFFS_GT.value] = (
        alt_ground_diff
    )
    self.sample_results[GroundStatTypes.NULL_SCORE_DIFFS_GT.value] = (
        null_ground_diff
    )
    results = {}
    results[AggStatTypes.GT_P_SCORE.value] = mcm.calculate_p_value(
        null_ground_diff, alt_ground_diff
    )

    ground_alt_machine1_wins, ground_alt_machine2_wins = mcm.higher_wins(
        self.sample_results[GroundStatTypes.M1_GT_ALT.value],
        self.sample_results[GroundStatTypes.M2_GT_ALT.value],
    )
    ground_null_machine1_wins, ground_null_machine2_wins = mcm.higher_wins(
        self.sample_results[GroundStatTypes.M1_GT_NULL.value],
        self.sample_results[GroundStatTypes.M2_GT_NULL.value],
    )

    results[AggStatTypes.GT_M1_ALT_WINS.value] = ground_alt_machine1_wins
    results[AggStatTypes.GT_M2_ALT_WINS.value] = ground_alt_machine2_wins
    results[AggStatTypes.GT_M1_NULL_WINS.value] = ground_null_machine1_wins
    results[AggStatTypes.GT_M2_NULL_WINS.value] = ground_null_machine2_wins

    for val in TestStatTypes.value_list():
      _, local_mean, _ = mcm.mean_and_confidence_bounds(
          self.sample_results[val]
      )
      results[val] = local_mean
    for val in GroundStatTypes.basic_stat_values():
      lower, local_mean, upper = mcm.mean_and_confidence_bounds(
          self.sample_results[val]
      )
      results[val] = local_mean
      results[f"{val} Lower"] = lower
      results[f"{val} Upper"] = upper
      results[f"{val} Std"] = np.std(self.sample_results[val])
    return results

  def parse_metric_func(
      self, func_spec: str
  ) -> Tuple[str, list[Any], dict[str, Any]]:
    """Parses the function spec and returns the function name and params.

    Args:
      func_spec: The function with optional args, e.g. "accuracy",
        "accuracy(ht=0.5)" or "mean_of_emds(bins=5)".

    Returns:
      A tuple of
      1) function name
      2) the function list args or an empty list
      3) the function dictionary args or an empty dict
    """

    def parse_numeric(num_str: str) -> float | int:
      """Parses and returns a numeric value."""
      if "." in num_str:
        return float(num_str)
      else:
        return int(num_str)

    func_spec = func_spec.replace(" ", "")
    if "(" in func_spec:
      m = re.match("(.+)\\((.*)\\)", func_spec)
      if not m:
        raise ValueError(f"Cannot parse func_spec: {func_spec}")
      func_name, func_params = m.groups()
      # Parse key-value pairs separated by comma.
      matches = re.findall("([^=]+?)(?:=(.+?))?,", func_params + ",")
      arg_list = [parse_numeric(key) for key, val in matches if not val]
      arg_dict = {key: parse_numeric(val) for key, val in matches if val}
      if arg_list:
        raise ValueError(
            f"Only dictionary arguments are allowed. func_spec={func_spec}"
        )
      return func_name, arg_list, arg_dict
    else:
      return func_spec, [], {}

  def setup_experiment(self, config_row: dict[str, Any]) -> Mapping[str, Any]:
    """Unpacks a row from the experiment specification grid.

    Also rehydrates all functions used in experiments, and includes a bit of
    string hacking to make the ground truth experiments comparable with the
    test experiments.

    Args:
      config_row: The row of the current experiment grid.

    Returns:
      experiment_config: A copy of the input config_row with some cells
        replaced by functions.
    """
    metrics = {
        "mean_absolute_error": mcm.mean_absolute_error,
        "root_mean_squared_error": mcm.root_mean_squared_error,
        "max_absolute_error": mcm.max_absolute_error,
        "wins_mae": mcm.wins_mae,
        "spearmanr": mcm.spearmanr,
        "emd_agg": mcm.emd_aggregated,
        "mean_of_emds": mcm.mean_of_emds,
        "mean": mcm.mean,
        "cos_distance": mcm.cos_distance,
        "accuracy": mcm.accuracy,
        "precision": mcm.precision,
        "recall": mcm.recall,
        "f1_score": mcm.f1_score,
        "auc": mcm.auc,
    }
    response_aggregators = {"mean": np.mean, "noop": noop}
    shapers = {
        "noop": noop,
        "vectorize": vectorize,
    }
    samplers = {
        "(all_items,bootstrap_responses)": resample_items_and_responses_factory(
            all_items, sample_responses(self.k_responses)
        ),
        "(all_items,one_response)": resample_items_and_responses_factory(
            all_items, sample_responses(1)
        ),
        "(all_items,all_responses)": resample_items_and_responses_factory(
            all_items, noop
        ),
        "(bootstrap_items,bootstrap_responses)": (
            resample_items_and_responses_factory(
                bootstrap_items, sample_responses(self.k_responses)
            )
        ),
        "(bootstrap_items,all_responses)": resample_items_and_responses_factory(
            bootstrap_items, noop
        ),
        "(bootstrap_items,one_response)": resample_items_and_responses_factory(
            bootstrap_items, sample_responses(1)
        ),
        "(bootstrap_items,first_response)": (
            resample_items_and_responses_factory(
                bootstrap_items, first_response
            )
        ),
    }
    ground_samplers = {
        "(all_items,bootstrap_responses)": resample_items_and_responses_factory(
            all_items, sample_ground_responses(self.k_responses)
        ),
        "(all_items,one_response)": resample_items_and_responses_factory(
            all_items, sample_ground_responses(1)
        ),
        "(all_items,all_responses)": resample_items_and_responses_factory(
            all_items, noop
        ),
        "(bootstrap_items,bootstrap_responses)": (
            resample_items_and_responses_factory(
                bootstrap_items, sample_ground_responses(self.k_responses)
            )
        ),
        "(bootstrap_items,one_response)": resample_items_and_responses_factory(
            bootstrap_items, sample_ground_responses(1)
        ),
        "(bootstrap_items,all_responses)": resample_items_and_responses_factory(
            bootstrap_items, noop
        ),
        "(bootstrap_items,first_response)": (
            resample_items_and_responses_factory(
                bootstrap_items, first_response
            )
        ),
    }
    parameters_dict = {
        ParameterTypes.AGG_OVER_RESPONSES.value: response_aggregators,
        ParameterTypes.SHAPER.value: shapers,
        ParameterTypes.SAMPLER.value: samplers,
        ParameterTypes.GT_SAMPLER.value: ground_samplers,
    }
    experiment_config = config_row.copy()
    for param_name in parameters_dict:
      experiment_config[param_name] = parameters_dict[param_name][
          config_row[param_name]
      ]

    experiment_config[ParameterTypes.AGG_OVER_TRIALS.value] = np.mean

    metric_spec = config_row[ParameterTypes.COMPARISON_METRIC.value]
    metric_name, _, dict_args = self.parse_metric_func(metric_spec)
    experiment_config[ParameterTypes.COMPARISON_METRIC.value] = (
        functools.partial(metrics[metric_name], **dict_args)
    )

    return experiment_config

  def run_experiment(
      self,
      alt_samples: list[datatypes.ResponseData],
      null_samples: list[datatypes.ResponseData],
  ) -> Mapping[str, float]:
    """Executes an experiment given sample data.

    Variant of run_v2_ground_experiments that uses the first ground
    truth sample as the test sample.

    Args:
      alt_samples: A list of trial samples for the alternative hypothesis.
      null_samples: A list of trial samples for the null hypothesis.

    Returns:
      The results of the experiment, in terms of the score and number of wins,
      over the samples.
    """
    self.sample_results = collections.defaultdict(list)

    count = 0
    for alt_sample, null_sample in zip(alt_samples, null_samples):
      if count == 0:
        self._get_test_set_results(alt_sample)
        count += 1

      self._get_ground_trial_results(alt_sample, null_sample)

    return self._aggregate_experiment_results()

class ExperimentsManager:
  """Manages experiments for testing resampling-based p-value estimators."""

  def __init__(
      self,
      exp_dir: str,
      input_response_file: str,
      use_pickle: bool,
      line: int,
      config_file_name: str,
      n_items: int,
      k_responses: int,
  ):
    """Initializer for ExperimentsManager class.

    Args:
      exp_dir: The base path for the input and configuration files.
      input_response_file: The input file of response data from the true
        distribution.
      use_pickle: If true decode the data file using pickle format.
      line: The line of the configuration file to run (to facilitate parallel
        execution), or -1 to run all of them.
      config_file_name: The name of the configuration file.
      n_items: Number of items per response set. Must be no greater than the
        number of items in the input_data datasets.
      k_responses: Number of responses per item. Must be no greater than number
        of responses per items in the input_data dataset.
    """
    self.exp_dir = exp_dir
    self.n_items = n_items
    self.k_responses = k_responses
    line_str = "" if line == -1 else f"_line={line}"
    self.output_file_name = (
        f"results_N={n_items}_K={k_responses}_{input_response_file}"
        f"{line_str}.csv"
    )
    data_file = os.path.join(exp_dir, input_response_file)
    logging.info("Opening data file %s", data_file)
    start_time = datetime.datetime.now()
    open_mode = "rb" if use_pickle else "r"
    with open(data_file, open_mode) as f:
      response_sets_dict = pickle.load(f) if use_pickle else json.load(f)

    response_sets = datatypes.ResponseSets.from_dict(response_sets_dict)
    logging.info("finished loading")
    loading_time = datetime.datetime.now() - start_time
    logging.info("File loading time=%f", loading_time.total_seconds())

    conversion_start_time = datetime.datetime.now()
    # Reduce the size for all the data arrays in response_data when the
    # response examples are over-generated.
    response_sets.truncate(n_items, k_responses)

    conversion_time = datetime.datetime.now() - conversion_start_time
    logging.info("Data conversion time=%f", conversion_time.total_seconds())

    config_file = os.path.join(exp_dir, "config/", config_file_name)
    logging.info("Opening config file %s", config_file)
    with open(config_file, "r") as f:
      self.e_grid = pd.read_csv(f)

    # The grouth-truth sampler is set to be same as the regular sampler,
    # although it may be mapped to a slightly different function later.
    self.e_grid[ParameterTypes.GT_SAMPLER.value] = self.e_grid[
        ParameterTypes.SAMPLER.value
    ]
    self.alt_samples = response_sets.alt_data_list
    self.null_samples = response_sets.null_data_list
    self.line = line
    self.elapsed_t = datetime.timedelta(0)

  def run_experiments(self):
    """Runs a collection of experiments."""

    for row_idx, config_row in self.e_grid.iterrows():
      if self.line != -1 and row_idx != self.line:
        continue

      experiment = Experiment(config_row, self.k_responses)
      start_time = datetime.datetime.now()
      results = experiment.run_experiment(self.alt_samples, self.null_samples)
      self.save_experiment_results(results, row_idx)

      compute_time = datetime.datetime.now() - start_time
      logging.info(
          "Compute time for config row %d is %f",
          row_idx,
          compute_time.total_seconds(),
      )
      self.elapsed_t += compute_time

    logging.info("Total compute time: %f", self.elapsed_t.total_seconds())
    # Write out the experiment results.
    with open(
        os.path.join(self.exp_dir, self.output_file_name), "wb"
    ) as f:
      self.e_grid.to_csv(f)

  def save_experiment_results(
      self, results: Mapping[str, float], row_idx: int
  ) -> None:
    """Append experiment results to the row of the experiment config DataFrame.

    Args:
      results: The results to write out.
      row_idx: The row of the configuration grid corresponding to the results.
    """
    for val in (
        TestStatTypes.value_list()
        + GroundStatTypes.value_list()
        + AggStatTypes.value_list()
    ):
      self.e_grid.at[row_idx, val] = results[val]

    logging.info(self.e_grid.loc[[row_idx]][ParameterTypes.value_list()])
