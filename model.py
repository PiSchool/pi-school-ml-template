# -*- coding: utf-8 -*-

# Copyright 2018 Simone Scardapane. All Rights Reserved.
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
In this example, we use a neural network with one hidden layer.

Note the logic: a general class for building several types of networks,
and one or more functions to build "standard" configurations. The functions
can be used to define common choices for the hyper-parameters and to make
experiments more scalable / repeatable.
"""

import tensorflow as tf

class SimpleNetwork(tf.keras.Model):
    """
    We use tf.layers here.
    """
    def __init__(self, hidden_size=10, output_size=3):
        super(SimpleNetwork, self).__init__()
        self.hid = tf.layers.Dense(hidden_size, activation=tf.nn.relu)
        self.out = tf.layers.Dense(output_size)
    
    def call(self, x):
        return self.out(self.hid(x))

def build_simple_model():
    
    return SimpleNetwork()
