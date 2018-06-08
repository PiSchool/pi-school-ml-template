
import numpy as np
import pandas as pd
import project_configuration
import data
import model
from sklearn.model_selection import cross_val_score


# def testGetData():
#     data.load_data(np.random.RandomState(42))
#     # If the file cannot be retrieved, an exception is raised.
#     assert True


def testPreprocessData():
    config = project_configuration.get_config()
    df = pd.read_csv(config["localDataFile"])
    X, y = data.preprocess_data(df)

    meanX = X.mean(axis=0)
    print(meanX, np.sum(meanX[meanX < -0.01]), np.sum(meanX[meanX > 0.01]))
    assert np.sum(meanX[meanX < -0.01]) == np.sum(meanX[meanX > 0.01]) == 0


def testSliptData():
    # test split data and if sets respect 60% 20%, 20% ratio
    X = np.arange(1000).reshape((1000, 1))
    y = np.random.randint(0, 100, (1000, 1))
    Xtrain, Xval, Xtest, ytrain, yval, ytest = data.split_data(
        X, y, rs=np.random.RandomState(42))
    print("testSliptData :  ", Xtrain.shape, Xval.shape, Xtest.shape,
          ytrain.shape, yval.shape, ytest.shape)
    assert Xtrain.shape[0] == 600 == ytrain.shape[0] and Xval.shape[
        0] == 200 == yval.shape[0] and Xtest.shape[0] == 200 == ytest.shape[0]


def testLoadData():
    Xtrain, Xval, Xtest, ytrain, yval, ytest = data.load_data(
        seed=42)
    print(Xtrain.shape, Xval.shape, Xtest.shape,
          ytrain.shape, yval.shape, ytest.shape)
    assert Xtrain.shape[0] == ytrain.shape[0] and Xval.shape[
        0] == yval.shape[0] and Xtest.shape[0] == ytest.shape[0] and round(Xtrain.shape[0] / Xtest.shape[0], 0) == 3


def testBuildModel():
    m = model.build_simple_model()

    Xtrain, Xtest, ytrain, ytest = data.load_data(
        rs=np.random.RandomState(42), validation_set=False)
    m.fit(Xtrain, ytrain)
    scores = cross_val_score(m, Xtrain, ytrain, cv=2,
                             scoring="neg_mean_squared_error")
    sqrt_scroes = np.sqrt(-scores)
    assert sqrt_scroes[0] > 0
