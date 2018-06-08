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

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_squared_log_error

import data
import model
import s3_helper
import logger

log = logger.getLogger()


def train():
    """
    All the training logic should go here.
    """
    log.info("start training model")
    # Get data
    X, Xtest, y, ytest = data.load_data(
        seed=42, validation_set=False)

    # Build the model
    m = model.build_simple_model()
    m.fit(X, y)
    log.info("model trained")

    log.info("save trained model and push to s3")
    s3_helper.save_and_push_model(m)


def test():
    """
    All test logic should go here. This function is called by the continuous integration tool
    to build a performance report per project
    """
    log.info("start testing model")
    # get data
    Xtrain, X, ytrain, y = data.load_data(
        seed=42, validation_set=False)
    # Build the model
    m = model.build_simple_model()

    test_metrics = {}
    log.info("download model")
    local_model = s3_helper.get_model(m.name)
    m = joblib.load(local_model)
    predicted_y = m.predict(X)
    mse = mean_squared_error(y, predicted_y)

    # add identifier of data used for the test
    test_metrics["dataHash"] = data.get_id()

    # gather the different metrics
    test_metrics["rmse"] = np.sqrt(mse)
    test_metrics["msle"] = mean_squared_log_error(y, predicted_y)
    test_metrics["evs"] = explained_variance_score(y, predicted_y)

    log.info(f"test metrics: {test_metrics}")
    return test_metrics


def proof_of_training():
    """
    This function is used by the continuous integration tool to build the performance report.
    The idea is to make a quick train ( not complete and not optimized) to check that the trainning process is working
    This shouldn't take more than 5 minute
    """
    log.info("start training model")
    try:
        # Get data
        Xtrain, Xtest, ytrain, ytest = data.load_data(
            seed=42, validation_set=False)
        # shorten the training set to speed up the trainning process
        X = Xtrain[:100, :]
        y = ytrain[:100]
        log.info([Xtrain.shape, ytrain.shape, X.shape, y.shape])
        # Build the model
        m = model.build_simple_model()
        m.fit(X, y)
        log.info("model trained")
        return True
    except Exception as e:
        log.exception(f'there was an error during the training: {e}')
        return False


if __name__ == "__main__":
    log.info('start main')

    # Training logic
    train()
    # Test logic
    test()
