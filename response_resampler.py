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

Annotator P-Value Experimentation Suite.

Run args:
  --n_items: Number of items.
  --k_responses: Number of responses per item.
  --line: Line of configuration file to run, or -1 for all lines.
  --config_file: Config file for running experiments.
  --in_data_file: File containing response data over which to run simulation.
  --path: path of cns directory for input, configuration, and output files.
"""

import random as rand

from absl import app
from absl import flags
from absl import logging

import response_resampler_lib as resampler_lib

_N_ITEMS = flags.DEFINE_integer("n_items", 100, "Number of items.")

_K_RESPONSES = flags.DEFINE_integer("k_responses", 5, "Number of responses.")
_LINE = flags.DEFINE_integer("line", -1, "Line of experiment file.")
_CONFIG_FILE = flags.DEFINE_string(
    "config_file",
    "config_N=1000_K=5_n_trials=1000.csv",
    "Config file for running experiments.",
)
_IN_DATA_FILE = flags.DEFINE_string(
    "in_data_file",
    "responses_simulated_distr_dist=0.3_gen_N=1000_K=5_n_samples=1000.json",
    "Function that generates norm.",
)
_PATH_NAME = flags.DEFINE_string(
    "path", "/cns/is-d/home/homanc/ptest/", "Path name of the cns directory."
)
_USE_PICKLE = flags.DEFINE_boolean(
    "use_pickle",
    False,
    "If true load the data using pickle. Otherwise load using json."
    "Pickle is much faster as it saves the data in binary format.",
)
_RANDOM_SEED = flags.DEFINE_integer(
    "random_seed",
    None,
    "When set, it generates the data in deterministically across runs.",
)

def main(_):
  logging.info(
      "Running ptest experiments with command line arguments:"
      "n_items = %d, k_responses = %d, "
      "line = %d, config = %s,"
      " input_data = %s",
      _N_ITEMS.value,
      _K_RESPONSES.value,
      _LINE.value,
      _CONFIG_FILE.value,
      _IN_DATA_FILE.value,
  )

  # Set random seeds for deterministic data generation.
  if _RANDOM_SEED.value:
    rand.seed(_RANDOM_SEED.value)

  experiments_manager = resampler_lib.ExperimentsManager(
      path_name=_PATH_NAME.value,
      in_data_file=_IN_DATA_FILE.value,
      use_pickle=_USE_PICKLE.value,
      line=_LINE.value,
      config_file_name=_CONFIG_FILE.value,
      n_items=_N_ITEMS.value,
      k_responses=_K_RESPONSES.value,
  )

  logging.info("Experiments set up. Getting ready to run.")
  experiments_manager.run_experiments()

if __name__ == "__main__":
  app.run(main)
