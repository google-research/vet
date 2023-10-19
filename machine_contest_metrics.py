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

Metrics used for machine vs. machine tests.

Contains metrics that compare the performance of one machine to another based on
their responses relative to a human-labeled response set. They were created
to support a simulator used to model response variance in machine learning
testing, but may be useful other settings where two sets of responses are
compared to a third.
"""

import math
from typing import Tuple
import numpy as np
from scipy import spatial
from scipy import stats
from sklearn import metrics

def binarize(scores: np.ndarray, threshold: float) -> np.ndarray:
  return np.where(scores < threshold, 0, 1)

def accuracy(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
    ht: float = 0.5,
    mt1: float = 0.5,
    mt2: float = 0.5,
) -> Tuple[float, float]:
  """Compute cosine similarities relative to human labels.

  The params ht, mt1 and mt2 can be specified via a config string,
  so we use short names for them.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.
    ht: The positive score threshold for human.
    mt1: The positive score threshold for machine1.
    mt2: The positive score threshold for machine2.

  Returns:
    A pair of cosine similarity scores, for machines 1 and 2, relative to
    human scores.
  """

  human = binarize(human, ht)

  return (
      metrics.accuracy_score(human, binarize(machine1, mt1)),
      metrics.accuracy_score(human, binarize(machine2, mt2)),
  )

def auc(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
    ht: float = 0.5,
    mt1: float = 0.5,
    mt2: float = 0.5,
) -> Tuple[float, float]:
  """Compute ROC AUC relative to human labels.

  The params ht, mt1 and mt2 can be specified via a config string,
  so we use short names for them.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.
    ht: The positive score threshold for human.
    mt1: The positive score threshold for machine1.
    mt2: The positive score threshold for machine2.

  Returns:
    A pair of receiver operater characteristic (ROC) area under the curve (AUC)
    scores, for machines 1 and 2, relative to human scores.
  """
  human = binarize(human, ht)

  return (
      metrics.roc_auc_score(human, binarize(machine1, mt1)),
      metrics.roc_auc_score(human, binarize(machine2, mt2)),
  )

def cos_distance(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> Tuple[float, float]:
  """Compute cosine similarities relative to human labels.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.

  Returns:
    A pair of cosine similarity scores, for machines 1 and 2, relative to
    human scores.
  """
  human = human.flatten()
  machine1 = machine1.flatten()
  machine2 = machine2.flatten()
  return (
      spatial.distance.cosine(human, machine1),
      spatial.distance.cosine(human, machine2),
  )

def emd_aggregated(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray, bins: int
) -> Tuple[float, float]:
  """Takes earth-mover's distance (EMD) relative to human labels.

  Distributions must first be aggregated across responses per item.

  Args:
    human: An array of human responses.
    machine1: An array of machine responses.
    machine2: An array of responses from another machine.
    bins: The number of bins to use.

  Returns:
    A 2-tuple of the vertically aggregated earth-movers distance between one
    machine and the human responses, and of the other machine at the human
    responses.
  """
  bins = [x / bins for x in range(bins + 1)]
  freq_human, freq_machine1, freq_machine2 = (
      np.histogram(vals, bins=bins)[0] for vals in (human, machine1, machine2)
  )

  return (
      stats.wasserstein_distance(freq_machine1, freq_human),
      stats.wasserstein_distance(freq_machine2, freq_human),
  )

def inverse_mean_squared_error(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> Tuple[float, float]:
  """Compute inverse mean squared error relative to human labels.

  Args:
    human: An array of human responses.
    machine1: An array of machine responses.
    machine2: An array of responses from another machine.

  Returns:
    A 2-tuple of the inverse mean squared error between one machine and the
    human responses, and of the other machine at the human responses.
  """
  return (
      1 / metrics.mean_squared_error(human, machine1),
      1 / metrics.mean_squared_error(human, machine2),
  )

def f1_score(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
    ht: float = 0.5,
    mt1: float = 0.5,
    mt2: float = 0.5,
) -> Tuple[float, float]:
  """Compute f1-score relative to human labels.

  The params ht, mt1 and mt2 can be specified via a config string,
  so we use short names for them.

  Args:
    human: An array of human responses.
    machine1: An array of machine responses.
    machine2: An array of responses from another machine.
    ht: The positive score threshold for human.
    mt1: The positive score threshold for machine1.
    mt2: The positive score threshold for machine2.

  Returns:
    A 2-tuple of the f-score, i.e., the harmonic mean of precision and recall,
    between one machine and the human responses, and of the other machine at the
    human responses.
  """
  human = binarize(human, ht)
  return (
      metrics.f1_score(human, binarize(machine1, mt1)),
      metrics.f1_score(human, binarize(machine2, mt2)),
  )

def precision(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
    ht: float = 0.5,
    mt1: float = 0.5,
    mt2: float = 0.5,
) -> Tuple[float, float]:
  """Compute precision relative to human labels.

  The params ht, mt1 and mt2 can be specified via a config string,
  so we use short names for them.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.
    ht: The positive score threshold for human.
    mt1: The positive score threshold for machine1.
    mt2: The positive score threshold for machine2.

  Returns:
    A pair of precision scores, for machines 1 and 2, relative to
    human scores.
  """
  human = binarize(human, ht)

  return (
      metrics.precision_score(human, binarize(machine1, mt1)),
      metrics.precision_score(human, binarize(machine2, mt2)),
  )

def recall(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
    ht: float = 0.5,
    mt1: float = 0.5,
    mt2: float = 0.5,
) -> Tuple[float, float]:
  """Compute recall relative to human labels.

  The params ht, mt1 and mt2 can be specified via a config string,
  so we use short names for them.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.
    ht: The positive score threshold for human.
    mt1: The positive score threshold for machine1.
    mt2: The positive score threshold for machine2.

  Returns:
    A pair of recall scores, for machines 1 and 2, relative to
    human scores.
  """
  human = binarize(human, ht)
  return (
      metrics.recall_score(human, binarize(machine1, mt1)),
      metrics.recall_score(human, binarize(machine2, mt2)),
  )

def wins_mae(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> Tuple[float, float]:
  """Compute number of wins relative to distance from human labels.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.

  Returns:
    A 2-tuple of the itemwise distance wins between one machine and the human
    responses, and of the other machine and the human responses.
  """

  dist_pairs = list(
      zip(
          (abs(x - y) for x, y in zip(human, machine1)),
          (abs(x - y) for x, y in zip(human, machine2)),
      )
  )

  return (
      sum(1 for a, b in dist_pairs if a < b),
      sum(1 for a, b in dist_pairs if a > b),
  )

def hist_frequency_2d(plot_scores: np.ndarray, bins: int) -> np.ndarray:
  """Constructs a 2-D histogram of scores.

  Args:
    plot_scores: a 2D array of plot scores.
    bins: the number of bins for computing the histogram.

  Returns:
    A 1D histogram of the row-major marginal distribution.
  """
  bins = [x / bins for x in range(bins + 1)]

  output = np.array([np.histogram(x, bins=bins)[0] for x in plot_scores])
  return output

def mean_average_error(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> Tuple[float, float]:
  """Compute itemwise distance mean.

  Args:
    human: An array of human responses.
    machine1: An array of machine responses.
    machine2: An array of responses from another machine.

  Returns:
    A 2-tuple of the itemwise distance mean between one machine and the human
    responses, and of the other machine and the human responses.
  """
  return (
      np.mean(list(abs(x - y) for x, y in zip(human, machine1))),
      np.mean(list(abs(x - y) for x, y in zip(human, machine2))),
  )

def itemwise_emds(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> Tuple[list[float], list[float]]:
  """Compute itemise earth movers distance between machine and human responses.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: A 2D array of responses from another machine.

  Returns:
    A tuple of earth movers distance (emd) scores, one for each human, machhine
    pair of response sets (for machines 1 and 2), with one emd score for each
    item in each of the pairs.

  Constructs a histogram for each item in each of the three arrays, then
  compares earth movers distance (emd) between corresponding items in
  the human responses and each of machine 1 and 2's responses.
  """
  scores1 = [stats.wasserstein_distance(x, y) for x, y in zip(human, machine1)]
  scores2 = [stats.wasserstein_distance(x, y) for x, y in zip(human, machine2)]
  return scores1, scores2

def kld(observed_mean: float, predicted_mean: float) -> float:
  """Compute the kl-divergence between two normal distributions.

  Used as a helper function for kld_of_means and mean_kld.

  If greater than 0, then return KLD.
  Else: Take the min of observed_mean and predicted_mean:
  If min < 0, add that plus 0.x to the min add the abs value of the min plus
  0.x to both.

  Args:
    observed_mean: The (true) mean of the observed normal distribution.
    predicted_mean: The (predicted) model mean.

  Returns:
    The kl divergence KL(ws || pl). Note: only valid if ws and pl are taken
    to represent the means of normal distributions.
  """
  if observed_mean > 0 and predicted_mean > 0:
    return (
        math.log(1 / observed_mean)
        - math.log(1 / predicted_mean)
        + (observed_mean / predicted_mean)
        - 1.0
    )
  else:
    add_score = abs(min(observed_mean, predicted_mean)) + 0.01
    observed_mean += add_score
    predicted_mean += add_score
    return (
        math.log(1 / observed_mean)
        - math.log(1 / predicted_mean)
        + (observed_mean / predicted_mean)
        - 1.0
    )

def kld_of_means(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> Tuple[float, float]:
  """Compute Kulback-Liebler Divergence (KLD) over all items.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.

  Returns:
    A pair of kld-of-means scores, for machines 1 and 2, relative to
    human scores.
  """
  human_mean = np.mean(human)
  return (
      kld(np.mean(machine1), human_mean),
      kld(np.mean(machine2), human_mean),
  )

def mean_kld(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> Tuple[float, float]:
  """Compute the mean of klds over all responses.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.

  Returns:
    A pair of mean-of-kld scores, for machines 1 and 2, relative to
    human scores.
  """
  scores1 = np.mean(
      list((kld(np.mean(x), np.mean(y)) for x, y in zip(machine1, human)))
  )
  scores2 = np.mean(
      list((kld(np.mean(x), np.mean(y)) for x, y in zip(machine2, human)))
  )
  return scores1, scores2

def mean_of_emds(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray, bins: int
) -> Tuple[float, float]:
  """Takes mean of earth-mover's distance (EMD) relative to human labels.

  Mean is over all items, EMD is across responses per item.

  Args:
    human: An array of human responses.
    machine1: An array of machine responses.
    machine2: An array of responses from another machine.
    bins: The number of bins to use.

  Returns:
    A 2-tuple of the vertically aggregated earth-movers distance between one
    machine and the human responses, and of the other machine at the human
    responses.
  """
  freq_scores1 = hist_frequency_2d(np.array(machine1), bins)
  freq_scores2 = hist_frequency_2d(np.array(machine2), bins)
  freq_human = hist_frequency_2d(np.array(human), bins)
  scores1, scores2 = itemwise_emds(freq_human, freq_scores1, freq_scores2)
  return np.mean(scores1), np.mean(scores2)

def spearmanr(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> Tuple[float, float]:
  """Compute spearman ranking correlation.

  Args:
    human: An array of human responses.
    machine1: An array of machine responses.
    machine2: An array of responses from another machine.

  Returns:
    A 2-tuple of the spearman ranking between one machine and the human
    responses, and of the other machine at the human responses.
  """
  return (
      stats.spearmanr(human, machine1)[0],
      stats.spearmanr(human, machine2)[0],
  )

def higher_wins(machine1: np.ndarray, machine2: np.ndarray) -> Tuple[int, int]:
  """Count the number of times each machine's score is greater than the other's.

  Used for scores over multiple trials where greater is better.

  Args:
    machine1: The scores for machine1 on some metric over multiple trials.
    machine2: The scores for machine2 on the same metric as machine1 on multiple
      trials.

  Returns:
    A tuple,
      The first argument is the mumber of trials machine1 won.
      The second argument is the mumber of trials machine2 won.
  """
  machine1_wins = 0
  machine2_wins = 0
  for j, _ in enumerate(machine1):
    if machine1[j] > machine2[j]:
      machine1_wins += 1
    elif machine2[j] > machine1[j]:
      machine2_wins += 1
  return machine1_wins, machine2_wins

def lower_wins(machine1: np.ndarray, machine2: np.ndarray) -> Tuple[int, int]:
  """Count the number of times each machine's score is lesser than the other's.

  Used for scores over multiple trials where lesser is better.

  Args:
    machine1: The scores for machine1 on some metric over multiple trials.
    machine2: The scores for machine2 on the same metric as machine1 on multiple
      trials.

  Returns:
    A tuple,
      The first argument is the mumber of trials machine1 won.
      The second argument is the mumber of trials machine2 won.
  """

  # pylint: disable=arguments-out-of-order
  return higher_wins(machine2, machine1)

def calculate_p_value(s_null: np.ndarray, s_alt: np.ndarray) -> float:
  """Compute the p-score for null and alternative hypothesis results.

  Provides a one-sided test with the assumption that alt distribution is biased
  greater than the null distribution.

  Args:
    s_null: A sequence of null hypothesis results.
    s_alt: A sequence of alt hypothesis results.

  Returns:
    The p-score for the results.
  """
  s_null = sorted(s_null, reverse=True)
  s_alt = sorted(s_alt, reverse=True)
  p_counts = i = j = 0
  while i < len(s_null) and j < len(s_alt):
    if s_null[i] >= s_alt[j]:
      i += 1
    else:
      p_counts += i
      j += 1
  if j < len(s_alt):
    p_counts += i * (len(s_alt) - j)

  return p_counts / (len(s_null) * len(s_alt))

def mean_and_confidence_bounds(
    scores: np.ndarray,
) -> Tuple[float, float, float]:
  """Compute mean and (empirical) confidence bounds for a list of numbers.

  Args:
    scores: A list of numbers.

  Returns:
    The mean and the 2.5th and 97.5th percentile numbers.
  """
  scores.sort()
  lower_index = int(len(scores) * 0.025)
  return scores[lower_index], np.mean(scores), scores[-lower_index]
