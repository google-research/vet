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

Code support for parameterized stochastic models for probabilistic response.

Use of this binary is based on two functions that work together. The main
function is generate_response_tables. It creates a collection of triple-wise
samples of responses for the three stochastic models. The caller specifies the
number of samples, the number of  items in each sample, and the number of
responses per item that each of the three machines provides.

The second function is passed as an argument to generate_response_tables.
It generates for each item and each of the three responses (human and machines
1 and 2) a probability distribution function, which generate_response tables
uses to generate the samples.

Example usage:

python toxicity_parameterized_sample --path_name=/data_dir/path --distortion=.02
"""
import datetime
import os
import random as rand
from typing import List, Sequence

from absl import app
from absl import flags
from absl import logging
import numpy as np
import parameterized_sample_lib as psample
import pandas as pd

_DISTORTION = flags.DEFINE_float(
    "distortion", 0.3, "Amount of distortion between machines."
)
_PATH_NAME = flags.DEFINE_string(
    "path_name", "/cns/is-d/home/homanc/ptest/", "The data directory path."
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
    "n_items", 1000, "Number of items per response set."
)
_K_RESPONSES = flags.DEFINE_integer(
    "k_responses", 5, "Number of responses per item."
)
_NUM_TRIALS = flags.DEFINE_integer(
    "num_trials", 20, "Number of trial per experiment."
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

def generate_toxicity_data_responses(
    file: str, response_sets: dict[str, List[dict[str, np.ndarray]]]
) -> dict[str, List[dict[str, np.ndarray]]]:
  """Generate ground truth data that is consistent with toxicity dataset.

  This is used only to generate data based this dataset:
  https://data.esrg.stanford.edu/study/toxicity-perspectives

  This function is needed because, unlike data that is generated purely
  synthetically, here we have actual data that we should run bootstrapping on.
  The responses that we bootstrap from are always the first set of responses
  generated, and so this function replaces the first set of synthetic responses
  with the real data.

  Args:
    file: The path and name of the datafile (if downloaded from the website).
    response_sets: the responses sets generated.

  Returns:
    A new set of response sets, with the first collection of response sets
    based directly on the toxicity dataset.
  """
  # We will open this file and use it to populate the first set of gold data
  # produced by the generator. We will also use the parameters learned from it
  # to produce the corresponding machine responses.
  with open(file, "rt") as f:
    all_df = pd.read_csv(f)

  df = all_df.sample(_NUM_SAMPLES.value)
  # Values here are divided by 5 because responses were originally from 0 to 4.
  # This scales them to 0 to .8, which makes them easier to compare to the
  # datasets generated without the toxicity dataset.
  toxicity_data = (
      df[["score_0", "score_1", "score_2", "score_3", "score_4"]].to_numpy() / 5
  )

  human_means = df["mean"] / 5
  human_stdev = df["stdev"] / 5
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
  response_sets["alt"][0]["gold"] = toxicity_data
  response_sets["alt"][0]["preds1"] = preds1_alt
  response_sets["alt"][0]["preds2"] = preds2_alt
  response_sets["null"][0]["gold"] = toxicity_data
  response_sets["null"][0]["preds1"] = preds1_null
  response_sets["null"][0]["preds2"] = preds2_null
  return response_sets

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  # Set random seeds for deterministic data generation.
  if _RANDOM_SEED.value:
    rand.seed(_RANDOM_SEED.value)
    np.random.seed(_RANDOM_SEED.value)

  generation_start_time = datetime.datetime.now()
  response_sets = psample.generate_response_tables(
      _N_ITEMS.value,
      _K_RESPONSES.value,
      _DISTORTION.value,
      _NUM_TRIALS.value,
      _GENERATOR.value,
  )
  elapsed_time = datetime.datetime.now() - generation_start_time
  logging.info("Regular data generation time=%f", elapsed_time.total_seconds())

  # For the toxicity data, in addition to the call to generate_response_tables
  # above, because unlike data that is generated purely synthetically, here we
  # have actual data that we should run bootstrapping on. The responses that we
  # bootstrap from are always the first set of responses generated, and so
  # generate_toxicity_data_responses replaces the first set of synthetically
  # generated responses with the real data.
  #
  # Note that even though the remaining response sets are generated
  # synthetically, we use a special generator tailored to the dataset to
  # generate them.
  toxicity_start_time = datetime.datetime.now()
  response_sets = generate_toxicity_data_responses(
      os.path.join(_PATH_NAME.value, "inputs", _INPUT.value), response_sets
  )
  elapsed_time = datetime.datetime.now() - toxicity_start_time
  logging.info("Toxicity data generation time=%f", elapsed_time.total_seconds())

  file_extension = "pkl" if _USE_PICKLE.value else "json"
  output_filename = os.path.join(
      _PATH_NAME.value,
      f"responses_simulated_distr_dist={_DISTORTION.value}_gen_N="
      f"{_N_ITEMS.value}_K={_K_RESPONSES.value}"
      f"_n_samples={_NUM_TRIALS.value}.{file_extension}",
  )
  psample.write_samples_to_file(
      response_sets, output_filename, _USE_PICKLE.value
  )

if __name__ == "__main__":
  app.run(main)
