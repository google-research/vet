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
function is simulate_response_tables. It creates a collection of triple-wise
samples of responses for the three stochastic models. The caller specifies the
number of samples, the number of  items in each sample, and the number of
responses per item that each of the three machines provides.

The second function is passed as an argument to simulate_response_tables.
It generates for each item and each of the three responses (human and machines
1 and 2) a probability distribution function, which simulate_response_tables
uses to generate the samples.

Example usage:

python parameterized_sample --exp_dir=/data_dir/path --distortion=.02
"""
import datetime
import os
import random as rand
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import numpy as np
import parameterized_sample_lib as psample

_DISTORTION = flags.DEFINE_float(
    "distortion", 0.3, "Amount of distortion between machines."
)
_EXP_DIR = flags.DEFINE_string(
    "exp_dir", "/tmp/ptest/",
    "The file path where the experiment input and output data are located."
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
_NUM_SAMPLES = flags.DEFINE_integer(
    "num_samples", 20, "Number of sample sets per experiment."
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

# for how to use this library.
def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
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
  logging.info("Data generation time=%f", elapsed_time.total_seconds())

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
