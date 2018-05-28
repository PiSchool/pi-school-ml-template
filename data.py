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
This module contains all the code related to input loading and preprocessing.

Exchange of data is done via three different iterators for:
    * Training.
    * Validation.
    * Testing.
    
We use RandomState to control the split in a fine way.    
"""

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(rs=np.random.RandomState(0)):
    
    # Load the data (from disk)
    data = load_iris()
    
    # Preprocessing
    X, y = preprocess_data(data)
    
    # Splitting
    Xtrain, Xval, Xtest, ytrain, yval, ytest = split_data(X, y, rs=rs)
    
    # Returning the iterators
    return tf.data.Dataset.from_tensor_slices((Xtrain, ytrain)), \
            tf.data.Dataset.from_tensor_slices((Xval, yval)), \
            tf.data.Dataset.from_tensor_slices((Xtest, ytest))
    
def preprocess_data(data):
    """
    This should contain all the code for the data preprocessing.
    In this case, we only apply a feature normalization.
    """
    
    return MinMaxScaler().fit_transform(data['data']), data['target']


def split_data(X, y, rs=np.random.RandomState(0)):
    """
    This should contain all the code for splitting the data into train / validation / split.
    Note the use of the RandomState to control the PNRG.
    """
    
    # Important: use stratify to get balanced cuts
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, stratify=y, random_state=rs)
    Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, stratify=ytrain, random_state=rs)
    
    return Xtrain, Xval, Xtest, ytrain, yval, ytest
