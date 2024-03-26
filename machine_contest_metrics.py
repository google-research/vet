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

import numpy as np
import scipy.spatial
import scipy.stats
import sklearn.metrics

def binarize(scores: np.ndarray, threshold: float) -> np.ndarray:
  return np.where(scores < threshold, 0, 1)

# the others below.
def accuracy(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
    ht: float = 0.5,
    mt1: float = 0.5,
    mt2: float = 0.5,
) -> tuple[float, float]:
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
      sklearn.metrics.accuracy_score(human, binarize(machine1, mt1)),
      sklearn.metrics.accuracy_score(human, binarize(machine2, mt2)),
  )

def auc(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
    ht: float = 0.5,
    mt1: float = 0.5,
    mt2: float = 0.5,
) -> tuple[float, float]:
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
      sklearn.metrics.roc_auc_score(human, binarize(machine1, mt1)),
      sklearn.metrics.roc_auc_score(human, binarize(machine2, mt2)),
  )

def cos_distance(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
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
  return (
      scipy.spatial.distance.cosine(human, machine1.flatten()),
      scipy.spatial.distance.cosine(human, machine2.flatten()),
  )

def emd_aggregated(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray, bins: int
) -> tuple[float, float]:
  """Takes earth-mover's distance (EMD) relative to human labels.

  Distributions must first be aggregated across responses per item.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: A 2D array of responses from another machine.
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
      scipy.stats.wasserstein_distance(freq_machine1, freq_human),
      scipy.stats.wasserstein_distance(freq_machine2, freq_human),
  )

def mean(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute means of machine1 and machine2.

  Note: human scores is not being used.

  Args:
    human: A 2D array of human scores.
    machine1: A 2D array of machine scores.
    machine2: A 2D array of machine scores.

  Returns:
    A pair of mean scores for machines 1 and 2 respectively.
  """
  del human
  return (np.mean(machine1), np.mean(machine2))

def root_mean_squared_error(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute (L2) root mean squared error relative to human labels.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: A 2D array of responses from another machine.

  Returns:
    A 2-tuple of the root mean squared error between one machine and the
    human responses, and of the other machine at the human responses.
  """
  return (
      sklearn.metrics.mean_squared_error(human, machine1, squared=False),
      sklearn.metrics.mean_squared_error(human, machine2, squared=False),
  )

def f1_score(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
    ht: float = 0.5,
    mt1: float = 0.5,
    mt2: float = 0.5,
) -> tuple[float, float]:
  """Compute f1-score relative to human labels.

  The params ht, mt1 and mt2 can be specified via a config string,
  so we use short names for them.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: A 2D array of responses from another machine.
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
      sklearn.metrics.f1_score(human, binarize(machine1, mt1)),
      sklearn.metrics.f1_score(human, binarize(machine2, mt2)),
  )

def precision(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
    ht: float = 0.5,
    mt1: float = 0.5,
    mt2: float = 0.5,
) -> tuple[float, float]:
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
      sklearn.metrics.precision_score(human, binarize(machine1, mt1)),
      sklearn.metrics.precision_score(human, binarize(machine2, mt2)),
  )

def recall(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
    ht: float = 0.5,
    mt1: float = 0.5,
    mt2: float = 0.5,
) -> tuple[float, float]:
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
      sklearn.metrics.recall_score(human, binarize(machine1, mt1)),
      sklearn.metrics.recall_score(human, binarize(machine2, mt2)),
  )

def wins_mae(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute number of wins relative to distance from human labels.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.

  Returns:
    A 2-tuple of the itemwise distance wins between one machine and the human
    responses, and of the other machine and the human responses.
  """

  machine1_results = abs(human - machine1)
  machine2_results = abs(human - machine2)

  return (
      np.sum(machine1_results < machine2_results),
      np.sum(machine1_results > machine2_results),
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

def mean_absolute_error(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute (L1) itemwise distance mean.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: A 2D array of responses from another machine.

  Returns:
    A 2-tuple of the itemwise distance mean between one machine and the human
    responses, and of the other machine and the human responses.
  """
  return (np.mean(abs(human - machine1)), np.mean(abs(human - machine2)))

def max_absolute_error(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute (L_infinity) itemwise maximum distance.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: A 2D array of responses from another machine.

  Returns:
    A 2-tuple of the itemwise max distance between one machine and the human
    responses, and of the other machine and the human responses.
  """
  return (np.max(abs(human - machine1)), np.max(abs(human - machine2)))

def itemwise_emds(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[list[float], list[float]]:
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
  scores1 = [scipy.stats.wasserstein_distance(x, y)
             for x, y in zip(human, machine1)]
  scores2 = [scipy.stats.wasserstein_distance(x, y)
             for x, y in zip(human, machine2)]
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
) -> tuple[float, float]:
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
) -> tuple[float, float]:
  """Compute the mean of klds over all responses.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.

  Returns:
    A pair of mean-of-kld scores, for machines 1 and 2, relative to
    human scores.
  """
  kld_vfunc = np.vectorize(kld)
  scores1 = np.mean(kld_vfunc(machine1, human))
  scores2 = np.mean(kld_vfunc(machine2, human))

  return scores1, scores2

def mean_of_emds(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray, bins: int
) -> tuple[float, float]:
  """Takes mean of earth-mover's distance (EMD) relative to human labels.

  Mean is over all items, EMD is across responses per item.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: A 2D array of responses from another machine.
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

def mean_relative_entropy(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray,
    bins: int = 100,
) -> tuple[float, float]:
  """Computes the average per-item KL divergence of machine to human labels.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: A 2D array of responses from another machine.
    bins: The number of bins to use.

  Returns:
    A 2-tuple of the D_KL(machine1, human), D_KL(machine2, human).
  """
  def histogram1d(x):
    return np.histogram(x, bins=bins, range=[0, 1])[0]

  human_freqs = np.apply_along_axis(func1d=histogram1d, axis=1, arr=human)
  machine1_freqs = np.apply_along_axis(func1d=histogram1d, axis=1, arr=machine1)
  machine2_freqs = np.apply_along_axis(func1d=histogram1d, axis=1, arr=machine2)
  machine1_kl = scipy.stats.entropy(
      pk=machine1_freqs, qk=human_freqs, base=2, axis=1)
  machine2_kl = scipy.stats.entropy(
      pk=machine2_freqs, qk=human_freqs, base=2, axis=1)
  return np.mean(machine1_kl), np.mean(machine2_kl)

def spearmanr(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute spearman ranking correlation.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: A 2D array of responses from another machine.

  Returns:
    A 2-tuple of the spearman ranking between one machine and the human
    responses, and of the other machine at the human responses.
  """
  return (
      scipy.stats.spearmanr(human, machine1)[0],
      scipy.stats.spearmanr(human, machine2)[0],
  )

def higher_wins(machine1: np.ndarray, machine2: np.ndarray) -> tuple[int, int]:
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
  machine1 = np.asarray(machine1)
  machine2 = np.asarray(machine2)
  return np.sum(machine1 > machine2), np.sum(machine2 > machine1)

def lower_wins(machine1: np.ndarray, machine2: np.ndarray) -> tuple[int, int]:
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

def calculate_p_value(
    s_null: np.ndarray,
    s_alt: np.ndarray,
    two_sided_test: bool = True,
) -> float:
  """Compute the p-value given null and alternative hypothesis results.

  Provides either a one-sided test, with the assumption that alt distribution is
  biased greater than the null distribution, or a two-sided test.

  Args:
    s_null: A sequence of null hypothesis results.
    s_alt: A sequence of alt hypothesis results.
    two_sided_test: Whether to test as two-sided/bidirectional (that is,
        M1 > M2 OR M2 > M1)

  Returns:
    The p-value for the results.
  """
  # Preprocessing for two-sided hypothesis test (ie, M1 > M2 OR M2 > M1).
  if two_sided_test and (np.median(s_null) > np.median(s_alt)):
    s_null, s_alt = -s_null, -s_alt

  s_null = sorted(s_null, reverse=True)
  s_alt = sorted(s_alt, reverse=True)

  # If all null hypothesis results are smaller than all alternative hypothesis
  # results then the p-value is 0 and, hence, H_0 is rejected.
  if s_null[0] < s_alt[-1]:
    return 0.0
  # If all null hypothesis results are larger than all alternative hypothesis
  # results then the p-value is 1 and, hence, H_0 clearly cannot be rejected.
  elif s_alt[0] < s_null[-1]:
    return 1.0

  p_count = 0  # numerator of p-value
  i = j = 0
  while i < len(s_null) and j < len(s_alt):
    if s_null[i] >= s_alt[j]:
      i += 1
    else:
      p_count += i
      j += 1

  # There are more s_alt items bigger than the last one in s_null.
  # They should all be counted.
  if j < len(s_alt) - 1 and s_alt[j] > s_null[-1]:
    p_count += (len(s_alt) - j - 1) * (len(s_null) - 1)

  p_value = p_count / (len(s_null) * len(s_alt))
  return 2 * p_value if two_sided_test else p_value

def mean_and_confidence_bounds(
    scores: np.ndarray,
) -> tuple[float, float, float]:
  """Compute mean and (empirical) confidence bounds for a list of numbers.

  Args:
    scores: A list of numbers.

  Returns:
    The mean and the 2.5th and 97.5th percentile numbers.
  """
  scores.sort()
  lower_index = int(len(scores) * 0.025)
  return scores[lower_index], np.mean(scores), scores[-lower_index]
