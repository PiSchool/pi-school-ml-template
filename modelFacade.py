

"""
This module contains the functions that need to be implemented for the performance report to work correctly
"""


def predict():
    """test the current model vs test set regarding a given performance metrix."""
    # call your model to check performance vs a given test set
    return 99


def getAuxiliaxyMetrics():
    """return additionnal data you would like to see in your performance report"""
    # call your model to get additionnal data you want to mesure.
    return ''


def getSingleMetricHash():
    """return an identifier of the single performance metric for exemple respecting the patern <metricType><TestSetIdentifier>"""
    return 'accuracyTestSetV1'
