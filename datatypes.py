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

import dataclasses
from typing import Any
import numpy as np

@dataclasses.dataclass
class ResponseData:
  """Human and machine results for comparison.

  gold:
    A n*k array for ground truth human responses.
  preds1:
    A n*k array for machine_1 responses.
  preds2:
    A n*k array for machine_2 responses.
  """

  gold: np.ndarray
  preds1: np.ndarray
  preds2: np.ndarray

  def truncate(self, n_items: int, k_responses: int):
    """Trims the arrays down to size n*k."""
    self.gold = self.gold[:n_items, :k_responses]
    self.preds1 = self.preds1[:n_items, :k_responses]
    self.preds2 = self.preds2[:n_items, :k_responses]

  def to_dict(self) -> dict[str, Any]:
    """Converts the data fields to a dictionary."""
    return {
        'gold': self.gold.tolist(),
        'preds1': self.preds1.tolist(),
        'preds2': self.preds2.tolist(),
    }

  @classmethod
  def from_dict(cls, input_dict: dict[str, Any]):
    """Constructs TripleData from a dictionary of arrays.

    Args:
      input_dict: Input arrays keyed by 'gold', 'preds1' and 'preds2'

    Returns:
      The constructed object.
    Raises:
      ValueError: when the shapes of the array are not the same.
    """
    response_data = ResponseData(
        gold=np.asarray(input_dict['gold']),
        preds1=np.asarray(input_dict['preds1']),
        preds2=np.asarray(input_dict['preds2']),
    )
    if (
        response_data.gold.shape != response_data.preds1.shape
        or response_data.gold.shape != response_data.preds2.shape
    ):
      raise ValueError(
          f'Array shapes do not match: gold={response_data.gold.shape},'
          f' preds1={response_data.preds1.shape},'
          f' preds2={response_data.preds2.shape}'
      )
    return response_data

@dataclasses.dataclass
class ResponseSets:
  """Contains multiple trials of alternative and null hypothesis data.

  alt_data_list:
    Contains experiment data for the alternative hypothesis.
  null_data_list:
    Contains experiment data for the null hypothesis.

  The two lists should be of the same size. The size of the list represents
  the number of trials of the experiment. Each trial should contain a variation
  of the alernative and null hypothesis experiment data. When using simulated
  data, gold, machine1 and machine2 should be from the same distribution across
  trials.

  The alternative hypothesis assumes that machine2 responses have a different
  performance than machine1 responses w.r.t. gold data on some metric, while
  the null hypothesis assumes that machine2 has the same performance as
  machine1.
  """

  alt_data_list: list[ResponseData]
  null_data_list: list[ResponseData]

  def to_dict(self) -> dict[str, Any]:
    """Converts the data fields to a dictionary."""
    return {
        'alt': [x.to_dict() for x in self.alt_data_list],
        'null': [x.to_dict() for x in self.null_data_list],
    }

  @classmethod
  def from_dict(cls, input_dict: dict[str, Any]):
    """Constructs ResponseSets from a dictionary.

    Args:
      input_dict: The input dictionary of data.

    Returns:
      The constructed object.
    Raises:
      ValueError: When the sizes of the two data arrays are not the same.
    """
    response_sets = ResponseSets(
        alt_data_list=[ResponseData.from_dict(d) for d in input_dict['alt']],
        null_data_list=[ResponseData.from_dict(d) for d in input_dict['null']],
    )
    if len(response_sets.alt_data_list) != len(response_sets.null_data_list):
      raise ValueError(
          'Array list sizes do not match: '
          f'alt={len(response_sets.alt_data_list)},'
          f'null={len(response_sets.null_data_list)}'
      )

    return response_sets

  def truncate(self, n_items: int, k_responses: int):
    """Trims the ResponseData in the data lists down to size n*k."""
    for response_data in self.alt_data_list:
      response_data.truncate(n_items, k_responses)
    for response_data in self.null_data_list:
      response_data.truncate(n_items, k_responses)
