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

Tests datatypes.
"""

import json

from absl.testing import absltest
import numpy as np
import datatypes

class DatatypesTest(absltest.TestCase):

  def test_response_data_json_conversion(self):
    numbers = np.array([[1, 2, 3]])
    response_data = datatypes.ResponseData(
        gold=numbers,
        preds1=numbers + 10,
        preds2=numbers + 100,
    )
    decoded_dict = json.loads(json.dumps(response_data.to_dict()))
    decoded_data = datatypes.ResponseData.from_dict(decoded_dict)
    self.assertDictEqual(response_data.to_dict(), decoded_data.to_dict())

  def test_response_sets_json_conversion(self):
    array_1 = np.array([[1, 2, 3]])
    response_data_1 = datatypes.ResponseData(
        gold=array_1,
        preds1=array_1 + 10,
        preds2=array_1 + 100,
    )
    array_2 = np.array([4, 5, 6])
    response_data_2 = datatypes.ResponseData(
        gold=array_2,
        preds1=array_2 + 10,
        preds2=array_2 + 100,
    )
    response_sets = datatypes.ResponseSets(
        alt_data_list=[response_data_1, response_data_1],
        null_data_list=[response_data_2, response_data_2],
    )
    decoded_dict = json.loads(json.dumps(response_sets.to_dict()))
    decoded_response_sets = datatypes.ResponseSets.from_dict(decoded_dict)
    self.assertDictEqual(
        response_sets.to_dict(), decoded_response_sets.to_dict()
    )

  def test_response_data_check_size(self):
    with self.assertRaises(ValueError):
      datatypes.ResponseData.from_dict(
          {'gold': [[1, 2, 3]], 'preds1': [[4, 5]], 'preds2': [[7, 8]]}
      )

  def test_response_sets_check_size(self):
    response_data_dict = {'gold': [[1]], 'preds1': [[2]], 'preds2': [[3]]}
    with self.assertRaises(ValueError):
      datatypes.ResponseSets.from_dict({
          'alt': [response_data_dict],
          'null': [response_data_dict, response_data_dict],
      })

  def test_response_data_truncate(self):
    response_data = datatypes.ResponseData.from_dict({
        'gold': [[1, 2], [3, 4]],
        'preds1': [[4, 5], [6, 7]],
        'preds2': [[6, 7], [8, 9]],
    })
    response_data.truncate(n_items=1, k_responses=1)
    self.assertEqual(response_data.gold, [[1]])
    self.assertEqual(response_data.preds1, [[4]])
    self.assertEqual(response_data.preds2, [[6]])

  def test_response_sets_truncate(self):
    response_data_1 = datatypes.ResponseData.from_dict({
        'gold': [[1, 2]],
        'preds1': [[3, 4]],
        'preds2': [[5, 6]],
    })
    response_data_2 = datatypes.ResponseData.from_dict({
        'gold': [[10, 20]],
        'preds1': [[30, 40]],
        'preds2': [[50, 60]],
    })
    response_sets = datatypes.ResponseSets([response_data_1], [response_data_2])
    response_sets.truncate(n_items=1, k_responses=1)
    self.assertEqual(
        response_sets.to_dict(),
        {
            'alt': [{'gold': [[1]], 'preds1': [[3]], 'preds2': [[5]]}],
            'null': [{'gold': [[10]], 'preds1': [[30]], 'preds2': [[50]]}],
        },
    )

if __name__ == '__main__':
  absltest.main()
