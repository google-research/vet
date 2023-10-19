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

Tests for parameterized_sample_lib.
"""

import random

from absl.testing import absltest
import parameterized_sample_lib as psample

class ParameterizedSampleTest(absltest.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    random.seed(0)

  def test_clamp_inbounds(self):
    """Test the clamp function for inbounds.

    Assumes _SCORE_MIN == 0 and _SCORE_MAX == 1
    """
    self.assertEqual(psample.clamp(0.5), 0.5)

  def test_clamp_over(self):
    """Test the clamp function for exceed _SCORE_MAX.

    Assumes _SCORE_MIN == 0 and _SCORE_MAX == 1
    """
    self.assertEqual(psample.clamp(2), 1)

  def test_clamp_under(self):
    """Test the clamp function for exceed _SCORE_MIN.

    Assumes _SCORE_MIN == 0 and _SCORE_MAX == 1
    """
    self.assertEqual(psample.clamp(-1), 0)

  def test_distort_shape(self):
    """Distort shape should only return values within bounds."""
    true_values = [psample.distort_shape(0.5, 0.5) for _ in range(100)]
    self.assertAlmostEqual(true_values[0], 0.84442, 5)
    self.assertAlmostEqual(true_values[3], 0.25892, 5)
    self.assertAlmostEqual(true_values[-1], 0.48644, 5)

  def test_sample_from(self):
    """Sample from should return an array of length 8."""
    true_value = psample.sample_from(lambda: 0, 8)
    self.assertCountEqual(true_value, [0] * 8)

  def test_gen_alt_h_distrs_norm(self):
    """Should return 3 lists of same length (10)."""
    h, m1, m2 = psample.gen_alt_h_distrs_norm(
        lambda: 0, lambda: 0, 10, alt_distortion=0
    )

    self.assertCountEqual([f() for f in h], [0] * 10)
    self.assertCountEqual([f() for f in m1], [0] * 10)
    self.assertCountEqual([f() for f in m2], [0] * 10)

  def test_sample_h(self):
    """Sample_h should provide three 2-D tables of results."""
    sampler_stub_h = lambda: 0
    sampler_stub_m1 = lambda: 1
    sampler_stub_m2 = lambda: 2
    n_samples = 3
    h, m1, m2 = psample.sample_h(
        [sampler_stub_h] * 2,
        [sampler_stub_m1] * 2,
        [sampler_stub_m2] * 2,
        n_samples,
    )
    for item in h:
      for response in item:
        self.assertEqual(response, 0)
    for item in m1:
      for response in item:
        self.assertEqual(response, 1)
    for item in m2:
      for response in item:
        self.assertEqual(response, 2)

  def test_generate_response_tables(self):
    """Should generate 2 sets of 2-dimensional tables."""

    def alt_distr_gen(n, distortion):
      return psample.gen_alt_h_distrs_norm(
          (lambda: 0.5), (lambda: 0), n, alt_distortion=distortion
      )

    samples = psample.generate_response_tables(100, 5, 0.3, 100, alt_distr_gen)
    for sample_set in samples["alt"]:
      for item in sample_set["gold"]:
        for response in item:
          self.assertEqual(response, 0.5)
      for item in sample_set["preds1"]:
        for response in item:
          self.assertEqual(response, 0.5)

if __name__ == "__main__":
  absltest.main()
