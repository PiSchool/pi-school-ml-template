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
This script should contain the actual training logic.
"""

import data, model
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

def loss(model, xb, yb):
    """
    General function to compute the loss.
    """
    
    return tf.losses.sparse_softmax_cross_entropy(logits=model(xb), labels=yb)

def train(model, train_it, val_it, epochs=1000):
    """
    All the training logic should go here.
    """
    
    # Define the optimization algorithm
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    for epoch in range(epochs):
        for xb, yb in train_it.shuffle(1000).batch(32):
            
            optimizer.minimize(lambda: loss(model, xb, yb), global_step=tf.train.get_or_create_global_step())
    
        # Validation step (generally goes at the end of every epoch)
        if epoch % 100 == 0:
            accuracy = tfe.metrics.Accuracy()
            for xb, yb in val_it.batch(32):
                # Accumulate accuracy over the validation set
                accuracy(tf.argmax(model(tf.constant(xb)), axis=1), tf.constant(yb))
            print('Epoch {}, validation accuracy is {}'.format(epoch, accuracy.result()))


def test(model, test_it):
    """
    All test logic should go here.
    """
    
    accuracy = tfe.metrics.Accuracy()
    for xb, yb in test_it.batch(32):
        # Accumulate accuracy over the test set
        accuracy(tf.argmax(model(tf.constant(xb)), axis=1), tf.constant(yb))
    print('Final test accuracy is {}'.format(accuracy.result()))

if __name__ == "__main__":
    
    # Run everything in eager execution
    tfe.enable_eager_execution()
    
    # Important: set all PRNGs
    np.random.seed(0)
    tf.set_random_seed(0)
    
    # Build the model
    model = model.build_simple_model()
    
    # Get iterators over data
    train_it, val_it, test_it = data.load_data()
    
    # Training logic
    train(model, train_it, val_it)
    
    # Test logic
    test(model, test_it)