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

This library supports the simulation of machine learning experiments in order
to improve experimental design. The tools presented here can help answer
questions such as:
    + How many human annotators are needed in order to create
      reliable gold standard test data?
    + How many samples per prompt are needed in order to reliably distinguish
      the performance of one generative model from another? How many prompts?
    + What is the best way to measure "reliability"?

This library provides stochastic parameterized **response grids** for
    + Two stochastic response models, such as two machine-learning-based models,
        say machine 1 and machine 2.
    + One "true" stochastic model, say a crowd of human annotators.
A typical use case involves evaluating the machine responses are against the
human responses over a common set of items, so in simulations we need sample
from all three models at the same time. This library does exactly that.
"""
import datetime
import enum
import functools
import json
import pickle
import random as rand
import scipy
from typing import Any, Callable, List, Tuple

from absl import logging
import numpy as np

import datatypes

def norm_bounded_sample(
  mean: float,
  std: float,
  bound: Tuple[float, float]
):
    """Sample from [0,1] via a normal distribution.

    Vars:
      mean: The mean of the normal distribution to sample from.
      std: The standard deviation of the distribution to sample from.
      range: an tuple containing upper and lower bounds within which
        the sample must be contained.
    Returns:
        A size "size" sample, as a list.
    """
    lower, upper = bound
    sample_of_one = scipy.stats.norm.rvs(loc=mean, scale=std, size=1)
    while ((sample_of_one < lower) or (sample_of_one > upper)):
      sample_of_one = scipy.stats.norm.rvs(loc=mean, scale=std, size=1)
    return sample_of_one[0]
 
def bi_norm_bounded_sample(
  mean1: float,
  std1: float,
  mean2: float,
  std2: float,
  mixture: float,
  bounds: Tuple[float, float]
):
    """Sample from [0,1] via a binormal distribution.

    Vars:
      mean1: The mean of the first normal distribution to sample from.
      std1: The standard deviation of the first distribution to sample from.
      mean2: The mean of the second normal distribution to sample from.
      std2: The standard deviation of the second distribution to sample from.
      mixture: The likelihood of choosing the first distirbution.
      bounds: an tuple containing upper and lower bounds within which
        the sample must be contained.
    Returns:
        A size "size" sample, as a list.
    """
    mixture_sample = rand.random()
    if mixture_sample < mixture:
      return norm_bounded_sample(mean1, std1, bounds)
    else:
      return norm_bounded_sample(mean2, std2, bounds)
 
def toxicity_mean_dist() -> float:
  """Return the mean of a toxicity human rater distribution.

  For use with the toxicity dataset.

  Returns:
    The value accoring to the clamped normalvariate with the parameters shown
    below.
  """
  return clamp(abs(np.random.default_rng().normal(0, 0.28)), max_value=0.8)

def toxicity_stdev_dist() -> float:
  """Return the standard deviation of a toxicity human rater distribution.

  For use with the toxicity dataset.

  Returns:
    The value accoring to the triagular distribution with the parameters shown
    below.
  """
  return clamp(np.random.default_rng().triangular(-0.06, 0.21, 0.45))

def clamp(num: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
  """Clamp clamp num to be between min/max by clipping.

  Args:
    num: The number to clamp.
    min_value: The minimum value clamped to.
    max_value: The maximum value clamped to.

  Returns:
    The number after clamping.
  """

  return np.clip(num, min_value, max_value)

def distort_shape(s_param: float, diff: float) -> float:
  """Randomly distort a parameter.

  Args:
    s_param: A scalar parameter.
    diff: The maximum amount of distortion.

  Returns:
    The scalar s_param plus a random amount, determined by diff.
  """
  return clamp(s_param + rand.uniform(-diff, diff))

def sample_from(distr: Callable[[], float], num: int) -> List[float]:
  """Draw a sample of size num from dist.

  Args:
    distr: A probability distribution function. Each call draws a random sample.
    num: The number of samples to draw.

  Returns:
    A list of random values
  """
  return [distr() for _ in range(num)]

def norm_distr_factory(
    mean: float,
    stdev: float,
    h_dist: Callable[[float, float], float] = rand.normalvariate,
) -> Callable[[], float]:
  """Helper function for gen_alt_h_distrs_norm.

  Args:
    mean: The mean parameter for the distribution.
    stdev: The standard deviation parameter distribution.
    h_dist: A norm-based probability distribution function.

  Returns:
    A function that samples from h_dist whenever called.
  """
  return lambda: h_dist(mean, stdev)

def gen_alt_h_distrs_norm(
    mean_distr: Callable[[], float],
    stdev_distr: Callable[[], float],
    n: int,
    alt_distortion: float = 0.1,
    h_dist: Callable[[float, float], float] = rand.normalvariate,
) -> Tuple[
    List[Callable[[], float]],
    List[Callable[[], float]],
    List[Callable[[], float]],
]:
  """Create parameterized normal probability distributions for each item.

  Args:
    mean_distr: A generating function for the mean parameter of each item's
      response.
    stdev_distr: A generating function for the standard deviation parameter of
      each item's response distribution.
    n: number of items.
    alt_distortion: A parameter determining the bias of the second model's
      response distribution
    h_dist: the distribution for each human or machine responder.

  Returns:
    A triple of human, machine 1 and machine 2 response distributions.
  """

  human_means = sample_from(mean_distr, n)
  human_stdevs = sample_from(stdev_distr, n)

  machine2_means = [distort_shape(s, alt_distortion) for s in human_means]

  human_item_distrs = [
      norm_distr_factory(mean, dev, h_dist)
      for mean, dev in zip(human_means, human_stdevs)
  ]
  machine1_item_distrs = [
      norm_distr_factory(mean, dev, h_dist)
      for mean, dev in zip(human_means, human_stdevs)
  ]
  machine2_item_distrs = [
      norm_distr_factory(mean, dev, h_dist)
      for mean, dev in zip(machine2_means, human_stdevs)
  ]
  return human_item_distrs, machine1_item_distrs, machine2_item_distrs

def sample_h(
    hum_h_distrs: List[Callable[[], float]],
    mach1_h_distrs: List[Callable[[], float]],
    mach2_h_distrs: List[Callable[[], float]],
    resps_per_item: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Sample a number of responses in the "horizontal" direction for each item.

  Args:
    hum_h_distrs: List of human response distributions.
    mach1_h_distrs: List of machine 1 response distributions.
    mach2_h_distrs: List of machine 2 response distributions.
    resps_per_item: The number of responses per item.

  Returns:
    Three response tables for humans, machine1, and machine2 responses,
    respectively, as numpy arrays.
  """
  gold = [sample_from(hdistr, resps_per_item) for hdistr in hum_h_distrs]
  preds1 = [sample_from(hdistr, resps_per_item) for hdistr in mach1_h_distrs]
  preds2 = [sample_from(hdistr, resps_per_item) for hdistr in mach2_h_distrs]

  gold = np.array(gold)
  preds1 = np.array(preds1)
  preds2 = np.array(preds2)

  return gold, preds1, preds2

def null_hypothesis_generator(
    distr1: Callable[[], float], distr2: Callable[[], float]
) -> Callable[[], float]:
  """Create a null hypthesis generator.

  Args:
    distr1: One distribution, representing the response distribution of one
      machine.
    distr2: Another distribution, representing the response distribution of
      another machine.

  Returns:
    A new distribution, which chooses uniformly from the two distributions
    given.
  """

  def null_dist():
    f = rand.choice([distr1, distr2])
    return f()

  return null_dist

def alt_distr_gen(
    n: int, distortion: float = 0.3
) -> Tuple[
    List[Callable[[], float]],
    List[Callable[[], float]],
    List[Callable[[], float]],
]:
  """Generates a number of alternative distribution triples.

  Args:
    n: The number of triples to generate. It is the number of items in a
      simulated response set.
    distortion: the distortion value of the mean/variance.

  Returns:
    A 3-tuple of lists of distribution functions. Each list is of length
    n, and corresponds to 1 of 2 machine responses, or a human response.
  """
  return gen_alt_h_distrs_norm(
      uniform_dist_factory(0, 1),
      uniform_dist_factory(0, 0.3),
      n,
      alt_distortion=distortion,
  )

def likert_norm_dist(mean: float, std: float, rate: int = 5) -> float:
  """Sample from a distribution over a likert-like domain.

  Args:
    mean: the mean of the generating normal distribution
    std: the standard deviation of the generating normal distribution
    rate: the number of levels in the likert-like domain

  Returns:
    A value between 0 and 1, generated by clamping the generating normal
    distribution between 0 and 1 and then breaking it into rate evenly
    distributed intervals. The value returned is the minimum value of
    all values in the iterval. E.g., for the default value of rate=5, the
    return values are {0, 0.2, 0.4, 0.6, 0.8}
  """
  x = clamp(np.random.default_rng().normal(mean, std))
  x = int(x * rate) / rate
  return x if x < 1 else (rate - 1) / rate

def binary_norm_dist(mean: float, std: float) -> float:
  """Sample from a distribution over a binary domain.

  Args:
    mean: the mean of the generating normal distribution
    std: the standard deviation of the generating normal distribution
    rate: the number of levels in the likert-like domain

  Returns:
    A value between 0 and 1, generated by clamping the generating normal
    distribution between 0 and 1 and then breaking it into rate evenly
    distributed intervals. The value returned is the minimum value of
    all values in the iterval. E.g., for the default value of rate=5, the
    return values are {0, 0.2, 0.4, 0.6, 0.8}
  """
  x = np.random.default_rng().normal(mean, std)
  return 0 if x < .5 else 1

def toxicity_distr_gen(
    n: int, distortion: float
) -> Tuple[
    List[Callable[[], float]],
    List[Callable[[], float]],
    List[Callable[[], float]],
]:
  """Generates a number alternative distribution triples.

  This specific generator is based on the toxicity dataset from:
  https://data.esrg.stanford.edu/study/toxicity-perspectives

  Args:
    n: The number of triples to generate. It is the number of items in a
      simulated response set.
    distortion: the distortion value.

  Returns:
    A 3-tuple of lists of distribution functions. Each list is of length
    n, and corresponds to 1 of 2 machine responses, or a human response.
  """
  return gen_alt_h_distrs_norm(
      toxicity_mean_dist,
      toxicity_stdev_dist,
      n,
      alt_distortion=distortion,
      h_dist=likert_norm_dist,
  )

def amazon_distr_gen(
    n: int, distortion: float
) -> Tuple[
    List[Callable[[], float]],
    List[Callable[[], float]],
    List[Callable[[], float]],
]:
  """Generates a number alternative distribution triples.

  This specific generator is based on the amazon ratings dataset.

  Args:
    n: The number of triples to generate. It is the number of items in a
      simulated response set.
    distortion: the distortion value.

  Returns:
    A 3-tuple of lists of distribution functions. Each list is of length
    n, and corresponds to 1 of 2 machine responses, or a human response.
  """
  return gen_alt_h_distrs_norm(
      lambda: norm_bounded_sample(0.552121, 0.032093, (0,1)),
      lambda: norm_bounded_sample(0.318177, 0.018281, (0,1)),
      n,
      alt_distortion=distortion,
      h_dist=likert_norm_dist,
  )

def armis_distr_gen(
    n: int, distortion: float
) -> Tuple[
    List[Callable[[], float]],
    List[Callable[[], float]],
    List[Callable[[], float]],
]:
  """Generates a number alternative distribution triples.

  This specific generator is based on the ArMIS dataset, part
  of the Learning with Disagreement challenge (le-wi-di),
  https://le-wi-di.github.io/

  Data is located here:
  https://github.com/Le-Wi-Di/le-wi-di.github.io/blob/main/data_post-competition.zip
  Args:
    n: The number of triples to generate. It is the number of items in a
      simulated response set.
    distortion: the distortion value.

  Returns:
    A 3-tuple of lists of distribution functions. Each list is of length
    n, and corresponds to 1 of 2 machine responses, or a human response.
  """
  return gen_alt_h_distrs_norm(
      lambda: bi_norm_bounded_sample(-0.430701, 0.418148, 1.194010, 0.525248, 0.652561, (0,1)),
      lambda: bi_norm_bounded_sample(-0.264113, 0.530150, 0.362404, 0.632262, 0.766390, (0,1)),
      n,
      alt_distortion=distortion,
      h_dist=binary_norm_dist,
  )

def convabuse_distr_gen(
    n: int, distortion: float
) -> Tuple[
    List[Callable[[], float]],
    List[Callable[[], float]],
    List[Callable[[], float]],
]:
  """Generates a number alternative distribution triples.

  This specific generator is based on the ConvAbuse dataset, part
  of the Learning with Disagreement challenge (le-wi-di),
  https://le-wi-di.github.io/

  Data is located here:
  https://github.com/Le-Wi-Di/le-wi-di.github.io/blob/main/data_post-competition.zip
  Args:
    n: The number of triples to generate. It is the number of items in a
      simulated response set.
    distortion: the distortion value.

  Returns:
    A 3-tuple of lists of distribution functions. Each list is of length
    n, and corresponds to 1 of 2 machine responses, or a human response.
  """
  return gen_alt_h_distrs_norm(
      lambda: norm_bounded_sample(1.124694, 0.512993, (0,1)),
      lambda: norm_bounded_sample(-0.324344, 0.417337, (0,1)),
      n,
      alt_distortion=distortion,
      h_dist=binary_norm_dist,
  )

def hs_brexit_distr_gen(
    n: int, distortion: float
) -> Tuple[
    List[Callable[[], float]],
    List[Callable[[], float]],
    List[Callable[[], float]],
]:
  """Generates a number alternative distribution triples.

  This specific generator is based on the HS_Brexit dataset, part
  of the Learning with Disagreement challenge (le-wi-di),
  https://le-wi-di.github.io/

  Data is located here:
  https://github.com/Le-Wi-Di/le-wi-di.github.io/blob/main/data_post-competition.zip
  Args:
    n: The number of triples to generate. It is the number of items in a
      simulated response set.
    distortion: the distortion value.

  Returns:
    A 3-tuple of lists of distribution functions. Each list is of length
    n, and corresponds to 1 of 2 machine responses, or a human response.
  """
  return gen_alt_h_distrs_norm(
      lambda: norm_bounded_sample(-0.338555, 0.281887, (0,1)),
      lambda: norm_bounded_sample(-0.341252, 0.409315, (0,1)),
      n,
      alt_distortion=distortion,
      h_dist=binary_norm_dist,
  )

def md_agreement_distr_gen(
    n: int, distortion: float
) -> Tuple[
    List[Callable[[], float]],
    List[Callable[[], float]],
    List[Callable[[], float]],
]:
  """Generates a number alternative distribution triples.

  This specific generator is based on the ConvAbuse dataset, part
  of the Learning with Disagreement challenge (le-wi-di),
  https://le-wi-di.github.io/

  Data is located here:
  https://github.com/Le-Wi-Di/le-wi-di.github.io/blob/main/data_post-competition.zip
  Args:
    n: The number of triples to generate. It is the number of items in a
      simulated response set.
    distortion: the distortion value.

  Returns:
    A 3-tuple of lists of distribution functions. Each list is of length
    n, and corresponds to 1 of 2 machine responses, or a human response.
  """
  return gen_alt_h_distrs_norm(
      lambda: norm_bounded_sample(-0.500000, 1.000000, (0,1)),
      lambda: norm_bounded_sample(-0.392347, 0.850241, (0,1)),
      n,
      alt_distortion=distortion,
      h_dist=binary_norm_dist,
  )

def generate_response_tables(
    n_items: int = 1000,
    k_responses: int = 5,
    distortion: float = 0.3,
    num_samples: int = 1000,
    alt_distr_generator: Callable[
        [int, float],
        Tuple[
            List[Callable[[], float]],
            List[Callable[[], float]],
            List[Callable[[], float]],
        ],
    ] = alt_distr_gen,
) -> datatypes.ResponseSets:
  """Generates a collection of human and machine responses.

  Generates tables ("sets"), for null and alternate hypotheses

  Args:
    n_items: Number of items per set.
    k_responses: Number of responses/set
    distortion: Mean/variance distortion value.
    num_samples: Number of samples of size n_items x k_responses.
    alt_distr_generator: Function that generates one <gold, machine1, machine2>
      response table set.

  Returns:
    A dictionary organized by null and alt hypothesis, with list of
    dictionaries, with each dictionary containing one <gold, machine1, machine2>
    response.
  """

  responses_alt = []
  responses_null = []

  for _ in range(num_samples):
    # Obtain response tables and results
    hum_h_distrs, mach1_h_distrs, mach2_h_distrs = alt_distr_generator(
        n_items, distortion
    )

    gold_alt, preds1_alt, preds2_alt = sample_h(
        hum_h_distrs, mach1_h_distrs, mach2_h_distrs, resps_per_item=k_responses
    )

    mach_null_h_distrs = [
        null_hypothesis_generator(mach1_h_distr, mach2_h_distr)
        for mach1_h_distr, mach2_h_distr in zip(mach1_h_distrs, mach2_h_distrs)
    ]

    gold_null, preds1_null, preds2_null = sample_h(
        hum_h_distrs, mach_null_h_distrs, mach_null_h_distrs,
        resps_per_item=k_responses
    )

    responses_alt.append(
        datatypes.ResponseData(
            gold=gold_alt, preds1=preds1_alt, preds2=preds2_alt
        )
    )
    responses_null.append(
        datatypes.ResponseData(
            gold=gold_null, preds1=preds1_null, preds2=preds2_null
        )
    )

  response_sets = datatypes.ResponseSets(
      alt_data_list=responses_alt, null_data_list=responses_null
  )

  return response_sets

def uniform_dist_factory(minimum: float, maximum: float) -> Callable[[], float]:
  """Helper function for passing to gen_alt_h_distrs_norm.

  Args:
    minimum: The min parameter for the distribution.
    maximum: The max parameter for the distribution.

  Returns:
    A function that samples from uniform[min, max] whenever called.
  """
  return lambda: rand.uniform(minimum, maximum)

def norm_generator(
    min_mean: float,
    max_mean: float,
    min_std: float,
    max_std: float,
    dist: float,
) -> Callable[[Any], Any]:
  """Helper function for generating triples of related norm distributions.

  (I.e., related to human and machs 1 and 2's responses to the same data item.)
  The mean and std_dev parameters are drawn from uniform intervals. The human
  and mach 1 have the same amount of distortion. An amount of distortion can be
  added to mach 2, it is also drawn from a uniform interval.

  Args:
    min_mean: the minimum value the mean may take.
    max_mean: the maximum value the mean may take.
    min_std: the minimum value the std_dev may take.
    max_std: the maximum value the std may take.
    dist: The distortion interval.

  Returns:
    The norm generators (as functions)
  """

  def fn(x):
    return gen_alt_h_distrs_norm(
        (lambda: rand.uniform(min_mean, max_mean)),
        (lambda: rand.uniform(min_std, max_std)),
        x,
        alt_distortion=dist,
    )

  # ugly, but we must save space
  min_mean_str = f"{min_mean}".replace("0.", ".")
  max_std_str = f"{max_std}".replace("0.", ".")
  dist_str = f"{dist}".replace("0.", ".")

  fn.__name__ = f"gen_alt_h_distrs_norm({min_mean_str},{max_mean},{min_std},{max_std_str},{dist_str})"
  return fn

def write_samples_to_file(
    response_sets: datatypes.ResponseSets,
    output_filename: str,
    use_pickle: bool,
) -> None:
  """Outputs the sample data to a file.

  Args:
    response_sets: The sample datasets to output.
    output_filename: The output filename.
    use_pickle: If true use pickle to serialize data. Otherwise use json. Pickle
      serialization is in binary format so it is more efficient.
  """
  write_start_time = datetime.datetime.now()
  open_mode = "wb" if use_pickle else "w"
  with open(output_filename, open_mode) as f:
    if use_pickle:
      pickle.dump(response_sets.to_dict(), f)
    else:
      json.dump(response_sets.to_dict(), f)

  elapsed_time = datetime.datetime.now() - write_start_time
  logging.info("File writing time=%f", elapsed_time.total_seconds())

@enum.unique
class GeneratorType(enum.Enum):
  """Types of generator functions."""

  def __call__(self, *args):
    return self.value(*args)

  ALT_DISTR_GEN: Callable[
      [int],
      Tuple[
          List[Callable[[], float]],
          List[Callable[[], float]],
          List[Callable[[], float]],
      ],
  ] = functools.partial(alt_distr_gen)
  TOXICITY_DISTR_GEN: Callable[
      [int],
      Tuple[
          List[Callable[[], float]],
          List[Callable[[], float]],
          List[Callable[[], float]],
      ],
  ] = functools.partial(toxicity_distr_gen)
  AMAZON_DISTR_GEN: Callable[
      [int],
      Tuple[
          List[Callable[[], float]],
          List[Callable[[], float]],
          List[Callable[[], float]],
      ],
  ] = functools.partial(amazon_distr_gen)
  HS_BREXIT_DISTR_GEN:  Callable[
      [int],
      Tuple[
          List[Callable[[], float]],
          List[Callable[[], float]],
          List[Callable[[], float]],
      ],
  ] = functools.partial(hs_brexit_distr_gen)
  ArMIS_DISTR_GEN:  Callable[
      [int],
      Tuple[
          List[Callable[[], float]],
          List[Callable[[], float]],
          List[Callable[[], float]],
      ],
  ] = functools.partial(armis_distr_gen)
  ConvAbuse_GEN:  Callable[
      [int],
      Tuple[
          List[Callable[[], float]],
          List[Callable[[], float]],
          List[Callable[[], float]],
      ],
  ] = functools.partial(convabuse_distr_gen)
  MD_Agreement_GEN:  Callable[
      [int],
      Tuple[
          List[Callable[[], float]],
          List[Callable[[], float]],
          List[Callable[[], float]],
      ],
  ] = functools.partial(md_agreement_distr_gen)
