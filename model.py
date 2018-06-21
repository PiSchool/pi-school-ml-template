# -*- coding: utf-8 -*-

# Copyright 2018. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains all the logic related to the actual model.
In this example, we use a random forest regressor.

Note the logic: a general class for building several types of models,
and one or more functions to build "standard" configurations. The functions
can be used to define common choices for the hyper-parameters and to make
experiments more scalable / repeatable.
"""

from sklearn.ensemble import RandomForestRegressor


class SimpleModel(RandomForestRegressor):

    def __init__(self, random_state=42,  n_estimators=10, max_features='auto'):
        super(SimpleModel, self).__init__(
            n_estimators=10, max_features='auto')
        self.name = 'RandomForestRegressor'


def build_simple_model():
    regressor = SimpleModel()
    return regressor
