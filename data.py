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
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split

import s3_helper
import logger

log = logger.getLogger()

# Create a class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

# from sklearn.datasets import load_iris
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split


data_identifier = {}


def get_id():
    id = f'{data_identifier["hash"]}_{data_identifier["seed"]}_{data_identifier["test_set_ratio"]}'
    return id


def load_data(seed=0, validation_set=True):
    data_identifier["hash"] = s3_helper.hash_data()
    data_identifier["seed"] = seed
    data_identifier["test_set_ratio"] = 0.2
    rs = np.random.RandomState(seed)
    downloaded_file_name = s3_helper.get_data()
    data = pd.read_csv(downloaded_file_name)

    # Preprocessing
    X, y = preprocess_data(data)
    log.info(["X shape :", X.shape, "y shape:",
              y.shape, "y first vals:", y[:5]])
    # Splitting
    if (validation_set):
        Xtrain, Xval, Xtest, ytrain, yval, ytest = split_data(
            X, y, rs=rs, validation_set=validation_set)
        return Xtrain, Xval, Xtest, ytrain, yval, ytest
    else:
        Xtrain, Xtest, ytrain, ytest = split_data(
            X, y, rs=rs, validation_set=validation_set)
        return Xtrain, Xtest, ytrain, ytest


def preprocess_data(data):
    """
    This should contain all the code for the data preprocessing.
    In this case, we only apply a feature normalization.
    """
    dfNumeric = data.select_dtypes(include=['int64', 'float64'])
    labels = dfNumeric["SalePrice"].copy().values
    transformed = dfNumeric.drop("SalePrice", axis=1)
    # transformation pipeline
    column_names = list(transformed.columns.values)
    pipeline = Pipeline([
        ('selector', DataFrameSelector(column_names)),
        #
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    transformations = [("numeric_pipeline", pipeline)]

    full_pipeline = FeatureUnion(transformer_list=transformations)
    train_set_prepared = full_pipeline.fit_transform(transformed)
    return train_set_prepared, labels


def split_data(X, y, rs=np.random.RandomState(0), validation_set=True):
    """
    This should contain all the code for splitting the data into train / validation / split.
    Note the use of the RandomState to control the PNRG.
    We are not using it as it is a regression exemple to keep code simple
    """
    # Important: use stratify to get balanced cuts
    if not(validation_set):
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, random_state=rs, test_size=0.2)

        return Xtrain, Xtest, ytrain, ytest
    else:
        # 20% of the set for validation and 20% for test
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, random_state=rs, test_size=0.4)
        Xtest, Xval, ytest, yval = train_test_split(
            Xtest, ytest, random_state=rs, test_size=0.5)

        return Xtrain, Xval, Xtest, ytrain, yval, ytest
