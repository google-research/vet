r"""Copyright 2022 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Code support for parameterized stochastic models for probabilistic responses
mixed with the toxicity data:
https://data.esrg.stanford.edu/study/toxicity-perspectives

This binary first generates the standard probabilistic responses using
distribution models fit to the toxicity data. Then it replaces the generated
gold data with the real human-annotated toxicity data, for both null and
alternative hypothesis datasets.

Example usage:

python toxicity_parameterized_sample --exp_dir=/data_dir/path --distortion=.02 \
    --input=toxicity_ratings_sample.csv
"""

import datetime
import os
import random as rand
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import numpy as np
import datatypes
import parameterized_sample_lib as psample
import pandas as pd

_DISTORTION = flags.DEFINE_float(
    "distortion", 0.3, "Amount of distortion between machines."
)
_EXP_DIR = flags.DEFINE_string(
    "exp_dir", "/tmp/ptest/", "The data directory path."
)
_INPUT = flags.DEFINE_string(
    "input", "toxicity_ratings_sample.csv", "The gold standard input file."
)

_GENERATOR = flags.DEFINE_enum_class(
    "generator",
    psample.GeneratorType.TOXICITY_DISTR_GEN,
    psample.GeneratorType,
    "A function that generates distributions.",
)
_N_ITEMS = flags.DEFINE_integer(
    "n_items", 1000, "Number of rows in the input response dataset."
)
_K_RESPONSES = flags.DEFINE_integer(
    "k_responses", 5, "Number of responses per item."
)
_NUM_SAMPLES = flags.DEFINE_integer(
    "num_samples", 1000, "Number of rows to sample for generator."
)
_USE_PICKLE = flags.DEFINE_boolean(
    "use_pickle",
    False,
    "If true use pickle to save data. Otherwise use json."
    "Pickle is much faster as it saves the data in binary format.",
)
_RANDOM_SEED = flags.DEFINE_integer(
    "random_seed",
    None,
    "When set, it generates the data in deterministically across runs.",
)

def generate_empirical_toxicity_data_responses(
    file: str, response_sets: datatypes.ResponseSets
) -> datatypes.ResponseSets:
  """Generate data that is consistent with the Toxicity dataset.

  This is used only to generate data based this dataset:
  https://data.esrg.stanford.edu/study/toxicity-perspectives

  This function is needed because, unlike simulated data that is purely
  synthetic, here we have actual data that we should run bootstrapping on.
  The responses that we run non-parametric bootstraping on are always the first
  sample (at index 0), and so this function replaces the model and gold sets of
  simulated responses with this more empirical data.

  Args:
    file: The path and name of the datafile (if downloaded from the website).
    response_sets: the response sets generated.

  Returns:
    A new set of response sets, with all the gold collection of response sets
    based directly on the toxicity dataset.
  """
  # We will open this file and use it to populate the first set of gold data
  # produced by the generator. We will also use the parameters learned from it
  # to produce the corresponding machine responses.
  with open(file, "rt") as f:
    all_df = pd.read_csv(f)

  df = all_df.sample(_N_ITEMS.value)
  # Values here are divided by 5 because responses were originally from 0 to 4.
  # This scales them to 0 to .8, which makes them easier to compare to the
  # datasets generated without the toxicity dataset.
  toxicity_data = (
      df[["score_0", "score_1", "score_2", "score_3", "score_4"]].to_numpy() / 5
  )
  human_means = np.mean(toxicity_data, axis=1)
  human_stdev = np.std(toxicity_data, axis=1, ddof=1)
  mac1_h_distrs = [
      psample.norm_distr_factory(mean, dev, psample.likert_norm_dist)
      for mean, dev in zip(human_means, human_stdev)
  ]
  machine2_means = [
      psample.distort_shape(s, _DISTORTION.value) for s in human_means
  ]
  mac2_h_distrs = [
      psample.norm_distr_factory(mean, dev, psample.likert_norm_dist)
      for mean, dev in zip(machine2_means, human_stdev)
  ]

  _, preds1_alt, preds2_alt = psample.sample_h(
      mac1_h_distrs, mac1_h_distrs, mac2_h_distrs, resps_per_item=5
  )

  mach_null_h_distrs = [
      psample.null_hypothesis_generator(mach1_h_distr, mach2_h_distr)
      for mach1_h_distr, mach2_h_distr in zip(mac1_h_distrs, mac2_h_distrs)
  ]

  _, preds1_null, preds2_null = psample.sample_h(
      mac1_h_distrs, mach_null_h_distrs, mach_null_h_distrs, resps_per_item=5
  )

  response_sets.alt_data_list[0].gold = toxicity_data
  response_sets.alt_data_list[0].preds1 = preds1_alt
  response_sets.alt_data_list[0].preds2 = preds2_alt
  response_sets.null_data_list[0].gold = toxicity_data
  response_sets.null_data_list[0].preds1 = preds1_null
  response_sets.null_data_list[0].preds2 = preds2_null
  return response_sets

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  input_filename = os.path.join(_EXP_DIR.value, "inputs", _INPUT.value)
  if not os.path.exists(input_filename):
    raise ValueError(f"Path {input_filename} does not exist!")

  # Set random seeds for deterministic data generation.
  if _RANDOM_SEED.value:
    rand.seed(_RANDOM_SEED.value)
    np.random.seed(_RANDOM_SEED.value)

  generation_start_time = datetime.datetime.now()
  response_sets = psample.simulate_response_tables(
      _N_ITEMS.value,
      _K_RESPONSES.value,
      _DISTORTION.value,
      _NUM_SAMPLES.value,
      _GENERATOR.value,
  )
  elapsed_time = datetime.datetime.now() - generation_start_time
  logging.info("Regular data generation time=%f", elapsed_time.total_seconds())

  # For the Toxicity data, in addition to the call to simulate_response_tables
  # above for simulated data, here we have actual data that we should run
  # bootstrapping on. The responses that we bootstrap from are always the first
  # set of responses generated (at index 0), and so we now replace the first
  # set of synthetically generated responses with the real data.
  toxicity_start_time = datetime.datetime.now()
  response_sets = generate_empirical_toxicity_data_responses(input_filename,
                                                             response_sets)
  elapsed_time = datetime.datetime.now() - toxicity_start_time
  logging.info("Toxicity data generation time=%f", elapsed_time.total_seconds())

  file_extension = "pkl" if _USE_PICKLE.value else "json"
  output_filename = os.path.join(
      _EXP_DIR.value,
      f"responses_simulated_distr_dist={_DISTORTION.value}_gen_N="
      f"{_N_ITEMS.value}_K={_K_RESPONSES.value}"
      f"_num_samples={_NUM_SAMPLES.value}.{file_extension}",
  )
  psample.write_samples_to_file(
      response_sets, output_filename, _USE_PICKLE.value
  )

if __name__ == "__main__":
  app.run(main)
